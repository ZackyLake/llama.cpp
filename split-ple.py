#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gguf
import mmap
import numpy as np
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from typing import Any, BinaryIO


MAX_MAP_BYTES = 4 * 1024**3
SPLIT_FILENAME = re.compile(r"^(?P<base>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})$")
PLE_TENSOR_NAME = "per_layer_token_embd.weight"


@dataclass(frozen=True)
class MetadataEntry:
    name: str
    value: Any
    value_type: gguf.GGUFValueType
    sub_type: gguf.GGUFValueType | None


@dataclass(frozen=True)
class TensorInfo:
    name: str
    path: Path
    offset: int
    n_bytes: int
    data_shape: tuple[int, ...]
    data_dtype: np.dtype
    tensor_type: gguf.GGMLQuantizationType


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    architecture: str | None
    endianess: gguf.GGUFEndian
    alignment: int
    has_alignment_key: bool
    split_count: int | None
    split_no: int | None
    declared_tensor_count: int | None
    metadata: tuple[MetadataEntry, ...]
    tensors: tuple[TensorInfo, ...]


def get_scalar_field(reader: gguf.GGUFReader, key: str, expected_type: gguf.GGUFValueType) -> int | None:
    field = reader.get_field(key)
    if field is None:
        return None
    if field.types != [expected_type]:
        raise ValueError(f"{reader.data.filename}: invalid type for {key}")
    return int(field.contents())


def inspect_shard(path: Path) -> ShardInfo:
    path = path.resolve()
    reader = gguf.GGUFReader(path)

    split_count = get_scalar_field(reader, gguf.Keys.Split.LLM_KV_SPLIT_COUNT, gguf.GGUFValueType.UINT16)
    split_no = get_scalar_field(reader, gguf.Keys.Split.LLM_KV_SPLIT_NO, gguf.GGUFValueType.UINT16)
    declared_tensor_count = get_scalar_field(
        reader, gguf.Keys.Split.LLM_KV_SPLIT_TENSORS_COUNT, gguf.GGUFValueType.INT32,
    )
    split_fields = (split_count, split_no, declared_tensor_count)
    if any(value is not None for value in split_fields) and any(value is None for value in split_fields):
        raise ValueError(f"{path}: incomplete split metadata")
    if split_count is not None and (split_count < 1 or split_no is None or not 0 <= split_no < split_count):
        raise ValueError(f"{path}: invalid split metadata")

    is_model_shard = split_no in (None, 0)
    architecture = None
    metadata = []
    if is_model_shard:
        architecture_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
        if architecture_field is None or architecture_field.types != [gguf.GGUFValueType.STRING]:
            raise ValueError(f"{path}: missing or invalid {gguf.Keys.General.ARCHITECTURE}")
        architecture = architecture_field.contents()

        for field in reader.fields.values():
            if field.name.startswith("GGUF.") or field.name.startswith("split."):
                continue
            if field.name in (gguf.Keys.General.ARCHITECTURE, gguf.Keys.General.ALIGNMENT):
                continue
            if not field.types:
                continue

            value_type = field.types[0]
            sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
            metadata.append(MetadataEntry(field.name, field.contents(), value_type, sub_type))

    tensors = tuple(
        TensorInfo(
            name=tensor.name,
            path=path,
            offset=tensor.data_offset,
            n_bytes=tensor.n_bytes,
            data_shape=tuple(int(dim) for dim in tensor.data.shape),
            data_dtype=tensor.data.dtype,
            tensor_type=tensor.tensor_type,
        )
        for tensor in reader.tensors
    )

    return ShardInfo(
        path=path,
        architecture=architecture,
        endianess=reader.endianess,
        alignment=reader.alignment,
        has_alignment_key=reader.get_field(gguf.Keys.General.ALIGNMENT) is not None,
        split_count=split_count,
        split_no=split_no,
        declared_tensor_count=declared_tensor_count,
        metadata=tuple(metadata),
        tensors=tensors,
    )


def discover_shards(input_info: ShardInfo) -> list[Path]:
    match = SPLIT_FILENAME.fullmatch(input_info.path.stem)
    if input_info.split_count is None:
        if match is not None and int(match.group("count")) > 1:
            raise ValueError(f"{input_info.path}: numbered filename has no split metadata")
        return [input_info.path]

    split_count = input_info.split_count
    if split_count == 1:
        if match is not None and (int(match.group("count")) != 1 or int(match.group("index")) != 1):
            raise ValueError(f"{input_info.path}: filename disagrees with split metadata")
        return [input_info.path]

    if match is None:
        raise ValueError(f"{input_info.path}: split model must use the standard -00001-of-000NN filename")
    if int(match.group("count")) != split_count:
        raise ValueError(f"{input_info.path}: filename split count does not match split.count")
    if int(match.group("index")) != input_info.split_no + 1:
        raise ValueError(f"{input_info.path}: filename index does not match split.no")

    base = match.group("base")
    paths = [
        input_info.path.with_name(f"{base}-{index:05d}-of-{split_count:05d}{input_info.path.suffix}")
        for index in range(1, split_count + 1)
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing model split: {missing[0]}")
    return [path.resolve() for path in paths]


def inspect_model(input_path: Path) -> tuple[ShardInfo, list[TensorInfo]]:
    input_shard = inspect_shard(input_path)
    paths = discover_shards(input_shard)
    shards = [input_shard if path == input_shard.path else inspect_shard(path) for path in paths]
    shards.sort(key=lambda shard: shard.split_no if shard.split_no is not None else 0)
    first_shard = shards[0]
    expected_count = first_shard.split_count
    if expected_count is not None and len(shards) != expected_count:
        raise ValueError(f"expected {expected_count} input shards, found {len(shards)}")

    tensors = []
    names = set()
    for index, shard in enumerate(shards):
        if expected_count is not None:
            if shard.split_no != index:
                raise ValueError(f"{shard.path}: shard ID does not match its position")
            if shard.declared_tensor_count != first_shard.declared_tensor_count:
                raise ValueError(f"{shard.path}: declared tensor count differs from the first split")

        for tensor in shard.tensors:
            if tensor.name in names:
                raise ValueError(f"duplicate tensor name across input splits: {tensor.name}")
            names.add(tensor.name)
            tensors.append(tensor)

    if expected_count is not None and len(tensors) != first_shard.declared_tensor_count:
        raise ValueError(
            f"input declares {first_shard.declared_tensor_count} tensors but contains {len(tensors)}",
        )

    ple_count = sum(tensor.name == PLE_TENSOR_NAME for tensor in tensors)
    if ple_count != 1:
        raise ValueError(f"expected exactly one {PLE_TENSOR_NAME}, found {ple_count}")

    return first_shard, tensors


def make_writer(path: Path, model: ShardInfo, tensors: list[TensorInfo], ple_only: bool) -> gguf.GGUFWriter:
    if model.architecture is None:
        raise ValueError(f"{model.path}: first shard has no model architecture")
    writer = gguf.GGUFWriter(path, model.architecture, endianess=model.endianess)
    if model.has_alignment_key:
        writer.add_custom_alignment(model.alignment)

    for entry in model.metadata:
        if ple_only and entry.name.startswith("tokenizer."):
            continue
        writer.add_key_value(entry.name, entry.value, entry.value_type, sub_type=entry.sub_type)

    for tensor in tensors:
        writer.add_tensor_info(
            tensor.name,
            tensor.data_shape,
            tensor.data_dtype,
            tensor.n_bytes,
            raw_dtype=tensor.tensor_type,
        )

    return writer


def copy_tensor_data(
    tensor: TensorInfo,
    output: BinaryIO,
    progress: Any | None,
    max_map_bytes: int = MAX_MAP_BYTES,
) -> None:
    granularity = mmap.ALLOCATIONGRANULARITY
    if max_map_bytes <= granularity:
        raise ValueError(f"mmap window must be larger than the allocation granularity ({granularity})")

    offset = tensor.offset
    remaining = tensor.n_bytes
    with tensor.path.open("rb") as source:
        while remaining:
            map_offset = offset - (offset % granularity)
            prefix = offset - map_offset
            chunk_size = min(remaining, max_map_bytes - prefix)
            map_length = prefix + chunk_size

            with mmap.mmap(
                source.fileno(), map_length, access=mmap.ACCESS_READ, offset=map_offset,
            ) as mapped:
                mapped_view = memoryview(mapped)
                chunk = mapped_view[prefix:prefix + chunk_size]
                try:
                    written = 0
                    while written < chunk_size:
                        part = chunk[written:]
                        try:
                            count = output.write(part)
                        finally:
                            part.release()
                        if count is None or count <= 0:
                            raise OSError(f"short write while copying tensor {tensor.name}")
                        written += count
                        progress.update(count)
                finally:
                    chunk.release()
                    mapped_view.release()

            offset += chunk_size
            remaining -= chunk_size


def write_output(
    path: Path,
    model: ShardInfo,
    tensors: list[TensorInfo],
    ple_only: bool,
    max_map_bytes: int,
) -> None:
    writer = make_writer(path, model, tensors, ple_only)
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()

        assert writer.fout is not None and len(writer.fout) == 1
        output = writer.fout[0]
        writer.write_padding(output, output.tell())

        total_bytes = sum(tensor.n_bytes for tensor in tensors)
        desc = "PLE" if ple_only else "No-PLE"
        with tqdm(total=total_bytes, desc=desc, unit="B", unit_scale=True) as progress:
            for tensor in tensors:
                copy_tensor_data(tensor, output, progress, max_map_bytes=max_map_bytes)
                writer.write_padding(output, tensor.n_bytes)
        writer.flush()
    finally:
        writer.close()


def output_paths(input_path: Path) -> tuple[Path, Path]:
    match = SPLIT_FILENAME.fullmatch(input_path.stem)
    base = match.group("base") if match is not None else input_path.stem
    return (
        input_path.with_name(f"{base}-ple.gguf"),
        input_path.with_name(f"{base}-nople.gguf"),
    )


def split_model(
    input_path: Path,
    max_map_bytes: int = MAX_MAP_BYTES,
    skip_ple: bool = False,
    skip_nople: bool = False,
) -> tuple[Path | None, Path | None]:
    if skip_ple and skip_nople:
        raise ValueError("cannot skip both outputs")

    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input GGUF does not exist: {input_path}")

    ple_path, nople_path = output_paths(input_path)
    targets = []
    if not skip_ple:
        targets.append(ple_path)
    if not skip_nople:
        targets.append(nople_path)
    for target in targets:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"output already exists: {target}")

    model, all_tensors = inspect_model(input_path)
    ple_tensors = [tensor for tensor in all_tensors if tensor.name == PLE_TENSOR_NAME]
    nople_tensors = [tensor for tensor in all_tensors if tensor.name != PLE_TENSOR_NAME]

    if not skip_ple:
        write_output(ple_path, model, ple_tensors, True, max_map_bytes)
    if not skip_nople:
        write_output(nople_path, model, nople_tensors, False, max_map_bytes)

    return (None if skip_ple else ple_path), (None if skip_nople else nople_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split per-layer embeddings into a separate GGUF file")
    parser.add_argument("input", type=Path, help="input GGUF file or one shard of a split GGUF model")
    parser.add_argument("--skip-ple", action="store_true", help="do not create the PLE-only GGUF")
    parser.add_argument("--skip-nople", action="store_true", help="do not create the GGUF without the PLE tensor")
    args = parser.parse_args(argv)

    try:
        ple_path, nople_path = split_model(args.input, skip_ple=args.skip_ple, skip_nople=args.skip_nople)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if ple_path is not None:
        print(f"PLE:     {ple_path}")
    if nople_path is not None:
        print(f"No-PLE:  {nople_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())