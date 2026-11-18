"""Evaluate the type-aware model on the Polyvore disjoint benchmark.

Reports the two paper tasks: compatibility ROC AUC and fill-in-the-blank
accuracy. Test images are streamed from the gated mvasil/polyvore-outfits Hub
dataset; only feature vectors needed by the benchmark are kept in a temp file.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
BENCHMARK_REPO = "Stylique/Polyvore"
BENCHMARK_REVISION = "1365152fecb3a8daee9d49c7d9d7f5809cc60b05"
IMAGE_REPO = "mvasil/polyvore-outfits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "type_aware_metrics.json",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def _download_annotations() -> tuple[dict, list[list[str]], list[dict], list[dict]]:
    data_dir = PROJECT_ROOT / "data" / "type_aware_benchmark" / "disjoint"
    data_dir.mkdir(parents=True, exist_ok=True)
    filenames = (
        "disjoint/test.json",
        "disjoint/compatibility_test.txt",
        "disjoint/fill_in_blank_test.json",
        "polyvore_item_metadata.json",
    )
    paths = {
        name: hf_hub_download(
            repo_id=BENCHMARK_REPO,
            filename=name,
            repo_type="dataset",
            revision=BENCHMARK_REVISION,
            local_dir=data_dir,
        )
        for name in filenames
    }
    outfits = json.loads(Path(paths[filenames[0]]).read_text(encoding="utf-8"))
    compatibility_rows = [
        line.split()
        for line in Path(paths[filenames[1]]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fitb_rows = json.loads(Path(paths[filenames[2]]).read_text(encoding="utf-8"))
    metadata = json.loads(Path(paths[filenames[3]]).read_text(encoding="utf-8"))
    return metadata, compatibility_rows, fitb_rows, outfits


def _image_tensor(image_data: dict, torch_module):
    if image_data.get("bytes") is not None:
        source = BytesIO(image_data["bytes"])
    else:
        source = image_data["path"]
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    width, height = image.size
    scale = 112 / min(width, height)
    resized = image.resize(
        (max(112, round(width * scale)), max(112, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    left = (resized.width - 112) // 2
    top = (resized.height - 112) // 2
    cropped = resized.crop((left, top, left + 112, top + 112))
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    tensor = torch_module.from_numpy(np.transpose(array, (2, 0, 1)).copy())
    mean = torch_module.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch_module.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _roc_auc(labels: list[int], scores: list[float]) -> float:
    """Compute ROC AUC with half credit for tied positive/negative scores."""
    order = np.argsort(np.asarray(scores), kind="mergesort")
    sorted_scores = np.asarray(scores)[order]
    sorted_labels = np.asarray(labels, dtype=np.int8)[order]
    positives = int(sorted_labels.sum())
    negatives = len(sorted_labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Compatibility AUC requires both positive and negative outfits")

    negative_below = 0
    concordant = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group = sorted_labels[start:end]
        group_positive = int(group.sum())
        group_negative = len(group) - group_positive
        concordant += group_positive * (negative_below + 0.5 * group_negative)
        negative_below += group_negative
        start = end
    return concordant / (positives * negatives)


def _evaluate(args: argparse.Namespace) -> dict:
    from home_made_rag.type_aware_compatibility import TypeAwareCompatibilityScorer

    metadata, compatibility_rows, fitb_rows, outfits = _download_annotations()
    scorer = TypeAwareCompatibilityScorer(
        enabled=True,
        checkpoint_path=PROJECT_ROOT / "models/type_aware/model_best.pth.tar",
        typespaces_path=PROJECT_ROOT / "models/type_aware/typespaces.p",
        device="auto",
    )
    if not scorer.available:
        raise RuntimeError(f"Type-aware checkpoint is unavailable: {scorer.reason}")
    if scorer.model is None or scorer.embedding_dim is None or scorer.n_conditions is None:
        raise RuntimeError("Type-aware model initialized without embedding dimensions")

    item_types: dict[str, str] = {}
    item_by_question_id: dict[str, str] = {}
    for outfit in outfits:
        set_id = str(outfit["set_id"])
        for item in outfit["items"]:
            item_id = str(item["item_id"])
            item_by_question_id[f"{set_id}_{item['index']}"] = item_id
            semantic_type = metadata.get(item_id, {}).get("semantic_category")
            normalized_type = scorer._normalize_type(semantic_type)
            if normalized_type is None:
                raise ValueError(f"Missing Polyvore semantic category for item {item_id}")
            item_types[item_id] = normalized_type

    general_index = scorer.n_conditions

    def condition_for(left_id: str, right_id: str) -> int:
        condition = scorer._condition_for(item_types[left_id], item_types[right_id])
        return general_index if condition is None else condition

    needed_conditions: dict[str, set[int]] = defaultdict(set)

    def register_pair(left_id: str, right_id: str) -> int:
        condition = condition_for(left_id, right_id)
        needed_conditions[left_id].add(condition)
        needed_conditions[right_id].add(condition)
        return condition

    compatibility_tasks: list[tuple[int, list[str]]] = []
    for row in compatibility_rows:
        label = int(row[0])
        garment_ids = [item_by_question_id[token] for token in row[1:]]
        for left_id, right_id in itertools.combinations(garment_ids, 2):
            register_pair(left_id, right_id)
        compatibility_tasks.append((label, garment_ids))

    fitb_tasks: list[tuple[list[str], list[tuple[str, bool]]]] = []
    for question in fitb_rows:
        question_ids = [item_by_question_id[token] for token in question["question"]]
        outfit_id = question["question"][0].rsplit("_", 1)[0]
        candidates = []
        for token in question["answers"]:
            item_id = item_by_question_id[token]
            for question_id in question_ids:
                register_pair(question_id, item_id)
            candidates.append((item_id, token.rsplit("_", 1)[0] == outfit_id))
        fitb_tasks.append((question_ids, candidates))

    used_ids = set(needed_conditions)
    offsets: dict[tuple[str, int], int] = {}
    vector_count = 0
    for item_id in sorted(used_ids):
        for condition in sorted(needed_conditions[item_id]):
            offsets[(item_id, condition)] = vector_count
            vector_count += 1

    print(
        f"Benchmark loaded: {len(compatibility_tasks):,} compatibility outfits, "
        f"{len(fitb_tasks):,} FITB questions, {len(used_ids):,} unique images; "
        f"saving {vector_count:,} type vectors.",
        flush=True,
    )

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    tensor_cache = np.memmap(
        Path(tempfile.gettempdir()) / f"polyvore-type-aware-{os.getpid()}.f32",
        mode="w+",
        dtype=np.float32,
        shape=(vector_count, scorer.embedding_dim),
    )
    pending = set(used_ids)
    batch_ids: list[str] = []
    batch_tensors = []
    decoded = 0
    stream_start = time.monotonic()

    def encode_batch() -> None:
        nonlocal decoded, batch_ids, batch_tensors
        if not batch_ids:
            return
        image_batch = torch.stack(batch_tensors).to(scorer.device)
        with torch.inference_mode():
            output = scorer.model(image_batch).detach().cpu().numpy()
        for batch_index, item_id in enumerate(batch_ids):
            for condition in needed_conditions[item_id]:
                tensor_cache[offsets[(item_id, condition)]] = output[batch_index, condition]
        decoded += len(batch_ids)
        if decoded % 2048 < len(batch_ids) or not pending:
            elapsed = max(time.monotonic() - stream_start, 1e-6)
            print(f"Encoded {decoded:,}/{len(used_ids):,} images ({decoded / elapsed:.1f}/s).", flush=True)
        batch_ids = []
        batch_tensors = []

    try:
        dataset = load_dataset(
            IMAGE_REPO,
            "disjoint",
            split="test",
            streaming=True,
        )
        for row in dataset:
            item_id = str(row["item_id"])
            if item_id not in pending:
                continue
            image_data = row.get("image")
            if not isinstance(image_data, dict):
                raise ValueError(f"Unexpected image payload for item {item_id}")
            batch_ids.append(item_id)
            batch_tensors.append(_image_tensor(image_data, torch))
            pending.remove(item_id)
            if len(batch_ids) >= args.batch_size:
                encode_batch()
            if not pending:
                break
        encode_batch()
    finally:
        tensor_cache.flush()

    if pending:
        sample = sorted(pending)[:10]
        raise RuntimeError(
            f"Polyvore image stream omitted {len(pending)} required images; examples: {sample}"
        )

    def pair_distance(left_id: str, right_id: str) -> float:
        condition = condition_for(left_id, right_id)
        left = tensor_cache[offsets[(left_id, condition)]]
        right = tensor_cache[offsets[(right_id, condition)]]
        return float(np.linalg.norm(left - right))

    compatibility_labels: list[int] = []
    compatibility_scores: list[float] = []
    for label, garment_ids in compatibility_tasks:
        distances = [
            pair_distance(left_id, right_id)
            for left_id, right_id in itertools.combinations(garment_ids, 2)
        ]
        compatibility_labels.append(label)
        compatibility_scores.append(1.0 - float(np.mean(distances)))

    fitb_correct = 0
    for question_ids, candidates in fitb_tasks:
        candidate_scores = []
        for answer_id, _is_correct in candidates:
            score = sum(pair_distance(question_id, answer_id) for question_id in question_ids)
            candidate_scores.append(score)
        predicted = int(np.argmin(candidate_scores))
        fitb_correct += int(candidates[predicted][1])

    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "Polyvore Outfits disjoint test",
        "checkpoint_source": "gyao15/fashion disjoint checkpoint",
        "checkpoint": "models/type_aware/model_best.pth.tar",
        "distance": "Euclidean in the matching type-specific embedding",
        "learned_metric_head": scorer.metric is not None,
        "embedding_dim": scorer.embedding_dim,
        "type_spaces": scorer.n_conditions,
        "compatibility_auc": _roc_auc(compatibility_labels, compatibility_scores),
        "compatibility_examples": len(compatibility_labels),
        "compatibility_positive_examples": int(sum(compatibility_labels)),
        "compatibility_negative_examples": len(compatibility_labels) - int(sum(compatibility_labels)),
        "fitb_accuracy": fitb_correct / len(fitb_tasks),
        "fitb_correct": fitb_correct,
        "fitb_questions": len(fitb_tasks),
        "images_encoded": decoded,
    }
    tensor_cache._mmap.close()
    try:
        (Path(tempfile.gettempdir()) / f"polyvore-type-aware-{os.getpid()}.f32").unlink()
    except FileNotFoundError:
        pass
    return report


def main() -> None:
    args = parse_args()
    report = _evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
