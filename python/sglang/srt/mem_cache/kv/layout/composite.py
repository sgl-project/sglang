"""A sub-pool whose unit holds several parts: the layout the plan gives a
lifetime group that declares more than one geometry.

A unit is one page of ``page_size`` tokens. Its parts come in two kinds:

  * per-token parts, laid side by side inside every token entry of the page
    (the page starts with its ``page_size`` token entries);
  * per-page parts, one block each after the page's token entries.

The page is padded to a stride every part keeps its grid in (whole token
entries, and whole rows of each part's rows). Every part addresses its data by
physical page ``p`` at ``p * page_bytes``: a reader indexes a page, then a row
within it. A part whose readers assume one uniform token stride across pages
(token-major rows read as a flat run) cannot sit in a unit whose pages are not
plain runs of token entries, so such a composition is refused.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Protocol, Tuple, runtime_checkable

import torch

from sglang.srt.mem_cache.kv.layout.base import (
    SubPoolSpec,
    check_physical_id_bound,
    dense_rows,
)
from sglang.srt.mem_cache.layout.token_major import ROW_ALIGN_BYTES

PER_TOKEN = "token"
PER_PAGE = "page"


@dataclass(frozen=True, kw_only=True)
class UnitPart(ABC):
    """One part of a composite unit, ``layer_num`` layers of it."""

    placement: ClassVar[str]
    # Whether the part's readers want its unit zeroed when handed out.
    zero_on_handout: ClassVar[bool] = False

    name: str
    layer_num: int

    @property
    @abstractmethod
    def page_addressable(self) -> bool:
        """Whether the part's readers address its data by page, a page stride
        apart, rather than as one run at a uniform token stride."""

    @abstractmethod
    def nbytes(self, page_size: int) -> int:
        """Bytes per token entry (per-token parts) or per page (per-page)."""

    def grid(self) -> int:
        """A byte multiple the page stride keeps, for the part's readers."""
        return ROW_ALIGN_BYTES

    @abstractmethod
    def views(
        self,
        raw: torch.Tensor,
        *,
        offset: int,
        spec: "CompositeSubPoolSpec",
        pages: int,
    ) -> List[torch.Tensor]:
        """Per layer, the part's data in every page, starting at byte
        ``offset`` of page 0 (page ``p`` at ``p * spec.page_bytes()``)."""


@dataclass(frozen=True, kw_only=True)
class LayerMajorRows(UnitPart):
    """Rows of ``row_bytes`` per token per layer, each layer's rows of the page
    contiguous: layer ``l``'s view is ``(pages, page_size, 1, row_bytes)`` with
    a row-sized token stride inside a page."""

    placement: ClassVar[str] = PER_PAGE
    zero_on_handout: ClassVar[bool] = True

    row_bytes: int

    @property
    def page_addressable(self) -> bool:
        return True

    def nbytes(self, page_size: int) -> int:
        return self.layer_num * page_size * self.row_bytes

    def grid(self) -> int:
        # Readers take the page stride as a whole number of rows.
        return math.lcm(ROW_ALIGN_BYTES, self.row_bytes)

    def views(self, raw, *, offset, spec, pages):
        p, row = spec.page_size, self.row_bytes
        return [
            torch.as_strided(
                raw,
                (pages, p, 1, row),
                (spec.page_bytes(), row, row, 1),
                offset + l * p * row,
            )
            for l in range(self.layer_num)
        ]


@dataclass(frozen=True, kw_only=True)
class PageBlobs(UnitPart):
    """One blob per page per layer, ``bytes_per_token`` per token of the page:
    layer ``l``'s view is ``(pages, page_size * bytes_per_token)``."""

    placement: ClassVar[str] = PER_PAGE

    bytes_per_token: int

    @property
    def page_addressable(self) -> bool:
        return True

    def nbytes(self, page_size: int) -> int:
        return self.layer_num * page_size * self.bytes_per_token

    def views(self, raw, *, offset, spec, pages):
        blob = spec.page_size * self.bytes_per_token
        return [
            torch.as_strided(
                raw, (pages, blob), (spec.page_bytes(), 1), offset + l * blob
            )
            for l in range(self.layer_num)
        ]


@dataclass(frozen=True, kw_only=True)
class TokenRows(UnitPart):
    """Rows of ``row_bytes`` per layer inside every token entry: layer ``l``'s
    view is ``(pages, page_size, 1, row_bytes)`` with the token entry as the
    token stride. ``readers_page_address`` says whether its readers take the
    page stride; when not, they read the rows as one run at the token entry
    stride, which only a unit of plain token entries keeps."""

    placement: ClassVar[str] = PER_TOKEN

    row_bytes: int
    readers_page_address: bool = False

    @property
    def page_addressable(self) -> bool:
        return self.readers_page_address

    def nbytes(self, page_size: int) -> int:
        return self.layer_num * self.row_bytes

    def views(self, raw, *, offset, spec, pages):
        p, entry, row = spec.page_size, spec.token_bytes(), self.row_bytes
        return [
            torch.as_strided(
                raw,
                (pages, p, 1, row),
                (spec.page_bytes(), entry, row, 1),
                offset + l * row,
            )
            for l in range(self.layer_num)
        ]


@runtime_checkable
class UnitPartGeometry(Protocol):
    """A part kind declared outside the built-in geometries that lays itself
    out in a composite unit: the part name its layers are routed by, whether it
    is laid out only in a unit (even alone), its bytes in the unit, and the
    view its readers take over them."""

    part_name: str

    def requires_unit_layout(self) -> bool: ...

    def unit_part(self, *, name: str, layer_num: int) -> UnitPart: ...

    def unit_view(self, *, arena, spec, part, page_size): ...


@dataclass(frozen=True, kw_only=True)
class CompositeSubPoolSpec(SubPoolSpec):
    """A sub-pool whose page holds ``parts``: per-token parts side by side in
    each token entry, then per-page parts in order, padded to every part's
    grid. ``layer_num`` counts the first part's layers (the sub-pool's token
    KV)."""

    page_size: int
    parts: Tuple[UnitPart, ...]

    def __post_init__(self):
        super().__post_init__()
        names = [part.name for part in self.parts]
        assert len(set(names)) == len(names), f"duplicate unit parts {names}"
        assert self.draft_region is None, (
            f"sub-pool {self.name!r}: a fused draft region rides in a composite "
            "unit as one of its per-token parts"
        )
        if not self.uniform():
            flat = [part.name for part in self.parts if not part.page_addressable]
            if flat:
                raise ValueError(
                    f"sub-pool {self.name!r} cannot hold parts {flat} beside "
                    f"{[n for n in names if n not in flat]}: their readers take "
                    "rows at one token stride across pages, and these pages "
                    "also hold per-page parts or padding. Only parts read by "
                    "page can share a unit."
                )

    # -- the unit's bytes --

    def token_bytes(self) -> int:
        """One token entry's per-token parts."""
        raw = sum(p.nbytes(self.page_size) for p in self._of(PER_TOKEN))
        return -(-raw // ROW_ALIGN_BYTES) * ROW_ALIGN_BYTES

    def page_bytes(self) -> int:
        """One unit: the token entries, the per-page parts, padded to whole
        token entries and every part's grid."""
        raw = self.page_size * self.token_bytes() + sum(
            p.nbytes(self.page_size) for p in self._of(PER_PAGE)
        )
        grid = math.lcm(self.page_size, *(p.grid() for p in self.parts))
        return -(-raw // grid) * grid

    def entry_bytes(self) -> int:
        # The arena counts a band in token entries; a unit is `page_size` of them.
        return self.page_bytes() // self.page_size

    def uniform(self) -> bool:
        """Whether a page is exactly its token entries, so every per-token
        part reads at one token stride across pages."""
        return self.page_bytes() == self.page_size * self.token_bytes()

    @property
    def zero_units_on_handout(self) -> bool:
        return any(p.zero_on_handout for p in self.parts)

    @property
    def copies_whole_units(self) -> bool:
        # Per-page bytes past the token entries move only with their page.
        return not self.uniform()

    def offsets(self) -> Dict[str, int]:
        """Each part's byte offset in page 0: per-token parts within the first
        token entry, per-page parts after the token entries."""
        found, token, page = {}, 0, self.page_size * self.token_bytes()
        for part in self.parts:
            if part.placement == PER_TOKEN:
                found[part.name], token = token, token + part.nbytes(self.page_size)
            else:
                found[part.name], page = page, page + part.nbytes(self.page_size)
        return found

    def part(self, name: str) -> UnitPart:
        return next(p for p in self.parts if p.name == name)

    def part_views(self, arena, name: str) -> List[torch.Tensor]:
        """Per layer, part ``name``'s data over this sub-pool's band of
        ``arena``, indexed by physical page."""
        pages = dense_rows(arena, self.name) // self.page_size
        check_physical_id_bound(sub_pool=self.name, n_rows=pages * self.page_size)
        return self.part(name).views(
            arena.raw, offset=self.offsets()[name], spec=self, pages=pages
        )

    def get_dtype(self) -> torch.dtype:
        return torch.uint8

    def _of(self, placement: str) -> List[UnitPart]:
        return [p for p in self.parts if p.placement == placement]
