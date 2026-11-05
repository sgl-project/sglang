# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""CPU contracts for Kimi K3 projection-storage coalescing."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.kernel_plugins.nvidia.kimi_k3 import projection_storage as storage
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _WeightParameter(torch.nn.Parameter):
    """Stand in for SGLang's weight-parameter subclasses."""


class _Weight(torch.nn.Module):
    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.weight = _WeightParameter(value, requires_grad=False)
        self.weight.loader_marker = object()


class _Attention(torch.nn.Module):
    pass


def _module(rows: int, columns: int, start: int) -> _Weight:
    value = torch.arange(start, start + rows * columns, dtype=torch.float32).reshape(
        rows, columns
    )
    return _Weight(value)


def _layout(*, alias: bool) -> storage._Layout:
    parts = (
        storage._Part("wide.weight", 4),
        storage._Part("small_a.weight", 2),
        storage._Part("small_b.weight", 1),
    )
    return storage._Layout(
        name="test KDA" if alias else "test MLA",
        parts=parts if alias else parts[:2],
        columns=4,
        dtype=torch.float32,
        alias_name="_tail" if alias else None,
        alias_part=1 if alias else None,
        padding_rows=1 if alias else 0,
    )


def _attention(*, alias: bool) -> _Attention:
    owner = _Attention()
    owner.wide = _module(4, 4, 0)
    owner.small_a = _module(2, 4, 100)
    owner.small_b = _module(1, 4, 200)
    if alias:
        tail = torch.cat(
            (
                owner.small_a.weight.detach(),
                owner.small_b.weight.detach(),
                torch.zeros((1, 4), dtype=torch.float32),
            )
        )
        owner.small_a.weight.data = tail[:2]
        owner.small_b.weight.data = tail[2:3]
        owner.__dict__["_tail"] = tail
    return owner


def _fallback(owner: _Attention, value: torch.Tensor, *, alias: bool) -> torch.Tensor:
    parts = [F.linear(value, owner.wide.weight), F.linear(value, owner.small_a.weight)]
    if alias:
        parts.append(F.linear(value, owner.small_b.weight))
    return torch.cat(parts, dim=-1)


class TestProjectionStorage(CustomTestCase):
    def _assert_parameter_and_state_contract(self, *, alias: bool) -> None:
        owner, layout = _attention(alias=alias), _layout(alias=alias)
        parameters = (owner.wide.weight, owner.small_a.weight) + (
            (owner.small_b.weight,) if alias else ()
        )
        identities = tuple(id(parameter) for parameter in parameters)
        loaders = tuple(parameter.loader_marker for parameter in parameters)
        state_before = {
            name: value.clone() for name, value in owner.state_dict().items()
        }
        keys_before = tuple(owner.state_dict())
        value = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        fallback_before = _fallback(owner, value, alias=alias)
        fused_tail_before = F.linear(value, owner.__dict__["_tail"]) if alias else None

        root = storage._coalesce_projection_storage(owner, layout, device_type="cpu")

        self.assertEqual(tuple(root.shape), (layout.rows, layout.columns))
        self.assertNotIsInstance(root, torch.nn.Parameter)
        self.assertEqual(identities, tuple(id(parameter) for parameter in parameters))
        self.assertTrue(
            all(type(parameter) is _WeightParameter for parameter in parameters)
        )
        self.assertEqual(
            loaders, tuple(parameter.loader_marker for parameter in parameters)
        )
        self.assertEqual(keys_before, tuple(owner.state_dict()))
        self.assertNotIn(storage._STORAGE_ATTR, owner.state_dict())
        self.assertNotIn(storage._STORAGE_ATTR, owner._parameters)
        self.assertNotIn(storage._STORAGE_ATTR, owner._buffers)
        self.assertNotIn(storage._STORAGE_ATTR, owner._modules)
        self.assertIs(owner.__dict__[storage._STORAGE_ATTR], root)
        for name, expected in state_before.items():
            torch.testing.assert_close(owner.state_dict()[name], expected)
        torch.testing.assert_close(
            _fallback(owner, value, alias=alias), fallback_before
        )
        if fused_tail_before is not None:
            torch.testing.assert_close(
                F.linear(value, owner.__dict__["_tail"]), fused_tail_before
            )

        row = 0
        for part, parameter in zip(layout.parts, parameters):
            self.assertTrue(
                storage._same_view(
                    parameter,
                    root,
                    row=row,
                    rows=part.rows,
                    columns=layout.columns,
                )
            )
            row += part.rows
        if alias:
            self.assertTrue(
                storage._same_view(
                    owner.__dict__["_tail"],
                    root,
                    row=4,
                    rows=4,
                    columns=4,
                )
            )
            self.assertEqual(torch.count_nonzero(root[-1]).item(), 0)

        pointers = tuple(parameter.data_ptr() for parameter in parameters)
        replacement = {
            name: value + index + 1
            for index, (name, value) in enumerate(owner.state_dict().items())
        }
        load_result = owner.load_state_dict(replacement, strict=True)
        self.assertEqual(load_result.missing_keys, [])
        self.assertEqual(load_result.unexpected_keys, [])
        for name, expected in replacement.items():
            torch.testing.assert_close(owner.state_dict()[name], expected)
        self.assertEqual(
            pointers, tuple(parameter.data_ptr() for parameter in parameters)
        )
        self.assertIs(
            storage._coalesce_projection_storage(owner, layout, device_type="cpu"),
            root,
        )
        self.assertEqual(
            pointers, tuple(parameter.data_ptr() for parameter in parameters)
        )

    def test_mla_preserves_parameters_state_dict_fallback_and_is_idempotent(self):
        self._assert_parameter_and_state_contract(alias=False)

    def test_kda_preserves_tail_views_state_dict_fallback_and_is_idempotent(self):
        self._assert_parameter_and_state_contract(alias=True)

    def test_kda_refuses_a_tail_that_is_not_shared_with_parameters(self):
        owner, layout = _attention(alias=False), _layout(alias=True)
        owner.__dict__["_tail"] = torch.cat(
            (
                owner.small_a.weight.detach(),
                owner.small_b.weight.detach(),
                torch.zeros((1, 4)),
            )
        )
        pointers = (
            owner.wide.weight.data_ptr(),
            owner.small_a.weight.data_ptr(),
            owner.small_b.weight.data_ptr(),
        )
        with self.assertRaisesRegex(ValueError, "not the expected _tail view"):
            storage._coalesce_projection_storage(owner, layout, device_type="cpu")
        self.assertNotIn(storage._STORAGE_ATTR, owner.__dict__)
        self.assertEqual(
            pointers,
            (
                owner.wide.weight.data_ptr(),
                owner.small_a.weight.data_ptr(),
                owner.small_b.weight.data_ptr(),
            ),
        )

    def test_invalid_component_fails_before_mutating_parameters(self):
        owner, layout = _attention(alias=False), _layout(alias=False)
        owner.small_a.weight.data = owner.small_a.weight.data.to(torch.float64)
        pointers = (owner.wide.weight.data_ptr(), owner.small_a.weight.data_ptr())
        with self.assertRaisesRegex(ValueError, "dtype"):
            storage._coalesce_projection_storage(owner, layout, device_type="cpu")
        self.assertNotIn(storage._STORAGE_ATTR, owner.__dict__)
        self.assertEqual(
            pointers, (owner.wide.weight.data_ptr(), owner.small_a.weight.data_ptr())
        )

    def test_mixed_devices_fail_before_mutating_parameters(self):
        owner, layout = _attention(alias=False), _layout(alias=False)
        owner.small_a.weight = _WeightParameter(
            torch.empty((2, 4), dtype=torch.float32, device="meta"),
            requires_grad=False,
        )
        wide_pointer = owner.wide.weight.data_ptr()
        with self.assertRaisesRegex(ValueError, "device"):
            storage._coalesce_projection_storage(owner, layout, device_type="cpu")
        self.assertNotIn(storage._STORAGE_ATTR, owner.__dict__)
        self.assertEqual(owner.wide.weight.data_ptr(), wide_pointer)

    def test_tampered_idempotent_state_is_rejected_not_repaired(self):
        owner, layout = _attention(alias=False), _layout(alias=False)
        root = storage._coalesce_projection_storage(owner, layout, device_type="cpu")
        owner.small_a.weight.data = owner.small_a.weight.detach().clone()
        with self.assertRaisesRegex(
            ValueError, "not the expected projection-storage view"
        ):
            storage._coalesce_projection_storage(owner, layout, device_type="cpu")
        self.assertIs(owner.__dict__[storage._STORAGE_ATTR], root)

    def test_post_install_failure_rolls_back_every_view(self):
        owner, layout = _attention(alias=True), _layout(alias=True)
        parameters = (owner.wide.weight, owner.small_a.weight, owner.small_b.weight)
        data = tuple(parameter.data for parameter in parameters)
        tail = owner.__dict__["_tail"]
        with (
            patch.object(
                storage, "_validate_installed", side_effect=RuntimeError("injected")
            ),
            self.assertRaisesRegex(RuntimeError, "injected"),
        ):
            storage._coalesce_projection_storage(owner, layout, device_type="cpu")
        self.assertNotIn(storage._STORAGE_ATTR, owner.__dict__)
        self.assertIs(owner.__dict__["_tail"], tail)
        for parameter, original in zip(parameters, data):
            self.assertEqual(parameter.data_ptr(), original.data_ptr())
            self.assertEqual(parameter.storage_offset(), original.storage_offset())

    def test_production_layouts_pin_the_supported_kimi_geometry(self):
        self.assertEqual(
            [(part.path, part.rows) for part in storage._MLA_LAYOUT.parts],
            [
                ("fused_qkv_a_proj_with_mqa.weight", 2112),
                ("g_proj.weight", 1536),
            ],
        )
        self.assertEqual(
            [(part.path, part.rows) for part in storage._KDA_LAYOUT.parts],
            [
                ("fused_qkvg_proj.weight", 6144),
                ("f_a_proj.weight", 128),
                ("b_proj.weight", 12),
            ],
        )
        self.assertEqual(storage._MLA_LAYOUT.rows, 3648)
        self.assertEqual(storage._KDA_LAYOUT.rows, 6288)
        self.assertEqual(storage._MLA_LAYOUT.columns, 7168)
        self.assertEqual(storage._KDA_LAYOUT.columns, 7168)
        self.assertEqual(storage._KDA_LAYOUT.padding_rows, 4)
        self.assertEqual(storage._KDA_LAYOUT.alias_part, 1)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
