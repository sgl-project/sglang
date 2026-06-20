#!/usr/bin/env python3
"""Generate msgspec.Struct codecs from a protobuf FileDescriptorSet.

The proto (`proto/sglang/runtime/v1/sglang.proto`) is the IDL / single source of
truth for the SGLang runtime API. This emits Python message types that serialize
to msgpack for the ZMQ transport path (RFC #22558 Phase 4+), interoperating with
the Rust codec (prost structs + rmp-serde) generated from the same proto.

msgpack is schemaless: `msgspec.Struct` (default, non-array_like) encodes as a
msgpack **map keyed by field name**, and `omit_defaults=True` drops fields equal
to their default. That matches the Rust side's `.with_struct_map()` +
`#[serde(skip_serializing_if = ...)]`, so the two interoperate.

Do not run this directly; use `proto/generate_msgspec.sh`, which also runs protoc.
Usage:  python gen_msgspec.py <descriptor_set.bin>  > messages.py
"""

import sys

from google.protobuf import descriptor_pb2

FD = descriptor_pb2.FieldDescriptorProto

# proto scalar type -> (python annotation, default literal)
SCALAR = {
    FD.TYPE_DOUBLE: ("float", "0.0"),
    FD.TYPE_FLOAT: ("float", "0.0"),
    FD.TYPE_INT64: ("int", "0"),
    FD.TYPE_UINT64: ("int", "0"),
    FD.TYPE_INT32: ("int", "0"),
    FD.TYPE_FIXED64: ("int", "0"),
    FD.TYPE_FIXED32: ("int", "0"),
    FD.TYPE_BOOL: ("bool", "False"),
    FD.TYPE_STRING: ("str", '""'),
    FD.TYPE_BYTES: ("bytes", 'b""'),
    FD.TYPE_UINT32: ("int", "0"),
    FD.TYPE_SFIXED32: ("int", "0"),
    FD.TYPE_SFIXED64: ("int", "0"),
    FD.TYPE_SINT32: ("int", "0"),
    FD.TYPE_SINT64: ("int", "0"),
}


def short(type_name):
    return type_name.rsplit(".", 1)[-1]


def collect_maps(msg, prefix, out):
    """Record synthetic map-entry types as {full_name: (key_field, value_field)}."""
    for nested in msg.nested_type:
        full = f"{prefix}.{nested.name}"
        if nested.options.map_entry:
            kf = next(f for f in nested.field if f.name == "key")
            vf = next(f for f in nested.field if f.name == "value")
            out[full] = (kf, vf)
        collect_maps(nested, full, out)


def main(desc_path):
    fds = descriptor_pb2.FileDescriptorSet()
    with open(desc_path, "rb") as fh:
        fds.ParseFromString(fh.read())

    map_entries = {}
    messages = {}  # name -> DescriptorProto
    for fp in fds.file:
        for msg in fp.message_type:
            collect_maps(msg, f".{fp.package}.{msg.name}", map_entries)
            if not msg.options.map_entry:
                messages[msg.name] = msg

    def value_anno(f):
        if f.type == FD.TYPE_MESSAGE:
            return short(f.type_name)
        if f.type == FD.TYPE_ENUM:
            return "int"
        return SCALAR[f.type][0]

    # Topologically order messages so a nested Struct type is defined before use
    # (msgspec resolves field types at class-definition time).
    deps = {name: set() for name in messages}
    for name, msg in messages.items():
        for f in msg.field:
            if f.type == FD.TYPE_MESSAGE and f.type_name in map_entries:
                _, vf = map_entries[f.type_name]
                if vf.type == FD.TYPE_MESSAGE and short(vf.type_name) in messages:
                    deps[name].add(short(vf.type_name))
            elif f.type == FD.TYPE_MESSAGE and short(f.type_name) in messages:
                deps[name].add(short(f.type_name))

    ordered, seen = [], set()

    def visit(n):
        if n in seen:
            return
        seen.add(n)
        for d in sorted(deps[n]):
            visit(d)
        ordered.append(n)

    for n in messages:
        visit(n)

    out = [
        '"""AUTO-GENERATED from proto via proto/generate_msgspec.sh. Do not edit by hand.',
        "",
        "msgpack codec for the SGLang runtime API, generated from",
        "proto/sglang/runtime/v1/sglang.proto. Wire format: msgpack map keyed by proto",
        "field name, default-valued fields omitted. Interoperates with the Rust codec",
        "(rust/sglang-grpc/src/msgpack.rs).",
        "",
        "    from sglang.srt.grpc import messages",
        "    blob = messages.encode(req)",
        "    req = messages.decode(blob, messages.TextGenerateRequest)",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import msgspec",
        "",
        "",
    ]

    for name in ordered:
        msg = messages[name]
        out.append(f"class {name}(msgspec.Struct, omit_defaults=True):")
        if not msg.field:
            out.append("    pass")
            out.append("")
            out.append("")
            continue
        for f in msg.field:
            is_map = f.type == FD.TYPE_MESSAGE and f.type_name in map_entries
            if is_map:
                kf, vf = map_entries[f.type_name]
                kt = SCALAR.get(kf.type, ("str", ""))[0]
                out.append(
                    f"    {f.name}: dict[{kt}, {value_anno(vf)}] = "
                    f"msgspec.field(default_factory=dict)"
                )
            elif f.label == FD.LABEL_REPEATED:
                out.append(
                    f"    {f.name}: list[{value_anno(f)}] = "
                    f"msgspec.field(default_factory=list)"
                )
            elif f.type == FD.TYPE_MESSAGE or f.proto3_optional:
                out.append(f"    {f.name}: {value_anno(f)} | None = None")
            else:  # implicit-presence proto3 scalar
                out.append(f"    {f.name}: {value_anno(f)} = {SCALAR[f.type][1]}")
        out.append("")
        out.append("")

    out += [
        "_ENCODER = msgspec.msgpack.Encoder()",
        "_DECODERS: dict = {}",
        "",
        "",
        "def encode(obj) -> bytes:",
        '    """Serialize a generated message Struct to msgpack bytes."""',
        "    return _ENCODER.encode(obj)",
        "",
        "",
        "def decode(data: bytes, typ):",
        '    """Deserialize msgpack bytes into the given generated message type."""',
        "    dec = _DECODERS.get(typ)",
        "    if dec is None:",
        "        dec = _DECODERS[typ] = msgspec.msgpack.Decoder(typ)",
        "    return dec.decode(data)",
        "",
    ]
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: gen_msgspec.py <descriptor_set.bin>")
    main(sys.argv[1])
