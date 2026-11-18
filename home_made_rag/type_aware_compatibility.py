"""Inference wrapper for the ECCV 2018 type-aware fashion embeddings.

The original implementation is intentionally old and expects the Polyvore
ResNet-18 variant plus the ``typespaces.p`` file that defines the order of the
type-specific masks.  This module keeps the inference path self-contained and
loads the original checkpoint format without importing the original training
script.

The scorer is optional.  If the checkpoint or type-space file is missing, it
returns an explicit ``unavailable`` result instead of making the recommendation
pipeline fail.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
from PIL import Image, ImageOps


logger = logging.getLogger(__name__)


TYPE_ALIASES = {
    "top": "tops",
    "tops": "tops",
    "shirt": "tops",
    "shirts": "tops",
    "t-shirt": "tops",
    "tshirt": "tops",
    "blouse": "tops",
    "sweater": "tops",
    "hoodie": "tops",
    "sweatshirt": "tops",
    "bottom": "bottoms",
    "bottoms": "bottoms",
    "pants": "bottoms",
    "pant": "bottoms",
    "trousers": "bottoms",
    "jeans": "bottoms",
    "shorts": "bottoms",
    "skirt": "bottoms",
    "skirts": "bottoms",
    "shoe": "shoes",
    "shoes": "shoes",
    "sneaker": "shoes",
    "sneakers": "shoes",
    "boot": "shoes",
    "boots": "shoes",
    "sandal": "shoes",
    "sandals": "shoes",
    "bag": "bags",
    "bags": "bags",
    "handbag": "bags",
    "handbags": "bags",
    "purse": "bags",
    "hat": "hats",
    "hats": "hats",
    "cap": "hats",
    "scarf": "scarves",
    "scarves": "scarves",
    "jewelry": "jewellery",
    "jewellery": "jewellery",
    "accessory": "accessories",
    "accessories": "accessories",
    "sunglass": "sunglasses",
    "sunglasses": "sunglasses",
    "jacket": "outerwear",
    "jackets": "outerwear",
    "coat": "outerwear",
    "coats": "outerwear",
    "outerwear": "outerwear",
    "dress": "all-body",
    "dresses": "all-body",
    "jumpsuit": "all-body",
    "jumpsuits": "all-body",
    "all-body": "all-body",
    "all body": "all-body",
}


def _load_torch():
    """Import torch only when the optional evaluator is enabled."""
    try:
        import torch
        import torch.nn as nn
    except Exception as error:  # pragma: no cover - depends on environment
        raise RuntimeError("PyTorch is required for type-aware compatibility scoring") from error
    return torch, nn


class _BasicBlock:
    """Placeholder base used to keep the torch import lazy."""


def _build_model_classes(torch, nn):
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample is not None:
                residual = self.downsample(x)
            return self.relu(out + residual)

    class ResNet18(nn.Module):
        """The three-stage ResNet-18 used by the paper, not torchvision's full model."""

        def __init__(self, embedding_size):
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(BasicBlock, 64, 2)
            self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
            self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
            self.avgpool = nn.AvgPool2d(7)
            self.fc_embed = nn.Linear(256, embedding_size)

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            layers = [block(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes * block.expansion
            layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            return self.fc_embed(x)

    class TypeAwareImageModel(nn.Module):
        def __init__(self, embedding_size, n_conditions):
            super().__init__()
            # These attribute names intentionally match the original checkpoint
            # after stripping the outer Tripletnet prefix.
            self.embeddingnet = ResNet18(embedding_size)
            self.masks = nn.Embedding(n_conditions, embedding_size)

        def forward(self, x):
            general = self.embeddingnet(x)
            masks = self.masks.weight.unsqueeze(0).expand(general.size(0), -1, -1)
            masked = general.unsqueeze(1).expand_as(masks) * masks
            norm = torch.norm(masked, p=2, dim=2, keepdim=True) + 1e-10
            masked = masked / norm
            # The original implementation concatenates the general embedding
            # after all type-specific spaces during test-time inference.
            return torch.cat((masked, general.unsqueeze(1)), dim=1)

    return TypeAwareImageModel


class TypeAwareCompatibilityScorer:
    """Score catalog-grounded outfits with the ECCV 2018 model."""

    model_label = "Learning Type-Aware Embeddings (ECCV 2018)"

    def __init__(
        self,
        *,
        enabled: bool,
        checkpoint_path: str | Path,
        typespaces_path: str | Path,
        device: str = "auto",
        category_aliases: dict[str, str] | None = None,
        max_batch_size: int = 16,
    ) -> None:
        self.enabled = bool(enabled)
        self.checkpoint_path = Path(checkpoint_path)
        self.typespaces_path = Path(typespaces_path)
        self.requested_device = device
        self.category_aliases = {**TYPE_ALIASES, **(category_aliases or {})}
        self.max_batch_size = max(1, int(max_batch_size))
        self.available = False
        self.reason: str | None = None
        self.device = "cpu"
        self.embedding_dim: int | None = None
        self.n_conditions: int | None = None
        self.typespaces: dict[tuple[str, str], int] = {}
        self.model = None
        self.metric = None
        self._torch = None
        self._image_cache: dict[str, Any] = {}
        self._cache_lock = RLock()

        if not self.enabled:
            self.reason = "disabled in configuration"
            return
        self._initialize()

    def _initialize(self) -> None:
        if not self.checkpoint_path.is_file():
            self.reason = f"checkpoint not found: {self.checkpoint_path}"
            logger.warning("Type-aware compatibility is unavailable: %s", self.reason)
            return
        if not self.typespaces_path.is_file():
            self.reason = f"typespaces file not found: {self.typespaces_path}"
            logger.warning("Type-aware compatibility is unavailable: %s", self.reason)
            return
        try:
            torch, nn = _load_torch()
            self._torch = torch
            self.device = self._resolve_device(torch)
            state = self._load_checkpoint(torch)
            self.typespaces = self._load_typespaces()
            if not self.typespaces:
                raise ValueError("typespaces file is empty")

            mask_tensor = self._find_state_tensor(state, "masks.weight")
            embedding_tensor = self._find_state_tensor(state, "fc_embed.weight")
            if mask_tensor is None or embedding_tensor is None:
                raise ValueError("checkpoint does not contain the expected type-aware image weights")
            self.n_conditions = int(mask_tensor.shape[0])
            self.embedding_dim = int(mask_tensor.shape[1])
            if int(embedding_tensor.shape[0]) != self.embedding_dim:
                raise ValueError("checkpoint embedding dimensions do not agree")
            if len(self.typespaces) != self.n_conditions:
                raise ValueError(
                    f"typespaces/checkpoint mismatch: {len(self.typespaces)} spaces for {self.n_conditions} masks"
                )

            model_class = _build_model_classes(torch, nn)
            model = model_class(self.embedding_dim, self.n_conditions)
            model_state = model.state_dict()
            selected = {}
            for key, value in state.items():
                normalized = key.removeprefix("module.")
                stripped = normalized.removeprefix("embeddingnet.")
                if stripped in model_state and tuple(model_state[stripped].shape) == tuple(value.shape):
                    selected[stripped] = value
                elif normalized in model_state and tuple(model_state[normalized].shape) == tuple(value.shape):
                    selected[normalized] = value
            missing, _unexpected = model.load_state_dict(selected, strict=False)
            required = {"embeddingnet.fc_embed.weight", "embeddingnet.conv1.weight", "masks.weight"}
            if required.intersection(missing):
                raise ValueError(f"checkpoint is missing required weights: {sorted(required.intersection(missing))}")

            metric_weight = self._find_state_tensor(state, "metric_branch.weight")
            if metric_weight is not None and tuple(metric_weight.shape) == (1, self.embedding_dim):
                metric = nn.Linear(self.embedding_dim, 1, bias=False)
                metric.weight.data.copy_(metric_weight)
                metric.eval()
                # Item embeddings are cached on CPU, so keep this tiny layer
                # there too.  This also avoids moving every cached item to GPU.
                self.metric = metric

            self.model = model.to(self.device)
            self.model.eval()
            self.available = True
            self.reason = None
            logger.info(
                "Loaded %s checkpoint=%s device=%s dimensions=%d type_spaces=%d learned_metric=%s",
                self.model_label,
                self.checkpoint_path,
                self.device,
                self.embedding_dim,
                self.n_conditions,
                self.metric is not None,
            )
        except Exception as error:
            self.available = False
            self.reason = f"load failed: {type(error).__name__}: {error}"
            logger.exception("Type-aware compatibility is unavailable")

    def _resolve_device(self, torch):
        requested = str(self.requested_device or "auto").lower()
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_checkpoint(self, torch) -> dict[str, Any]:
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:  # Older PyTorch versions do not know weights_only.
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
            return checkpoint["state_dict"]
        if isinstance(checkpoint, dict):
            return checkpoint
        raise ValueError("checkpoint is not a state dictionary")

    def _load_typespaces(self) -> dict[tuple[str, str], int]:
        with self.typespaces_path.open("rb") as file:
            raw = pickle.load(file)
        if isinstance(raw, dict):
            entries = raw.items()
        elif isinstance(raw, (list, tuple)):
            entries = ((pair, index) for index, pair in enumerate(raw))
        else:
            raise ValueError("typespaces must be a list or dictionary")
        result = {}
        for pair, index in entries:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            left = self._normalize_type(pair[0])
            right = self._normalize_type(pair[1])
            if left and right:
                result[(left, right)] = int(index)
        return result

    @staticmethod
    def _find_state_tensor(state: dict[str, Any], suffix: str):
        for key, value in state.items():
            if key.removeprefix("module.").endswith(suffix):
                return value
        return None

    def _normalize_type(self, value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower().replace("_", "-")
        return self.category_aliases.get(normalized, normalized) or None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "model": self.model_label,
            "checkpoint": str(self.checkpoint_path),
            "typespaces": str(self.typespaces_path),
            "device": self.device,
            "embedding_dim": self.embedding_dim,
            "type_spaces": self.n_conditions,
            "learned_metric": self.metric is not None,
            "reason": self.reason,
        }

    def evaluate_outfits(self, outfits: list[dict[str, Any]], catalog) -> list[dict[str, Any]]:
        """Return one serializable compatibility report per outfit."""
        if not self.available:
            return [self._unavailable_report() for _ in outfits]

        reports = []
        for outfit in outfits:
            try:
                reports.append(self._evaluate_outfit(outfit, catalog))
            except Exception as error:
                logger.exception("Could not score outfit %s", outfit.get("outfit_id"))
                reports.append({
                    **self._unavailable_report(),
                    "status": "error",
                    "reason": f"{type(error).__name__}: {error}",
                })
        return reports

    def _evaluate_outfit(self, outfit: dict[str, Any], catalog) -> dict[str, Any]:
        items = [item for item in outfit.get("items", []) if item.get("garment_id")]
        if len(items) < 2:
            return {
                "status": "unscorable",
                "model": self.model_label,
                "reason": "at least two garments are required",
                "pair_count": 0,
            }

        embeddings, missing = self._embeddings_for_items(items, catalog)
        pair_reports = []
        fallback_count = 0
        for left_index in range(len(items) - 1):
            for right_index in range(left_index + 1, len(items)):
                left = items[left_index]
                right = items[right_index]
                left_id = str(left["garment_id"])
                right_id = str(right["garment_id"])
                left_embedding = embeddings.get(left_id)
                right_embedding = embeddings.get(right_id)
                if left_embedding is None or right_embedding is None:
                    continue
                left_type = self._normalize_type(left.get("category"))
                right_type = self._normalize_type(right.get("category"))
                condition = self._condition_for(left_type, right_type)
                fallback = condition is None
                if fallback:
                    fallback_count += 1
                    condition_index = -1
                else:
                    condition_index = condition

                distance = self._distance(left_embedding[condition_index], right_embedding[condition_index])
                pair_score = 1.0 - distance
                pair_reports.append({
                    "garment_1": left_id,
                    "garment_2": right_id,
                    "category_1": left_type or left.get("category"),
                    "category_2": right_type or right.get("category"),
                    "type_pair": self._type_pair_label(left_type, right_type),
                    "distance": round(distance, 6),
                    "compatibility_score": round(pair_score, 6),
                    "used_general_embedding": fallback,
                })

        if not pair_reports:
            return {
                "status": "unscorable",
                "model": self.model_label,
                "reason": "no pair could be scored; check image paths and category mapping",
                "pair_count": 0,
                "missing_garment_ids": missing,
            }

        distances = [pair["distance"] for pair in pair_reports]
        weakest = max(pair_reports, key=lambda pair: pair["distance"])
        mean_distance = sum(distances) / len(distances)
        return {
            "status": "partial" if missing else "scored",
            "model": self.model_label,
            "compatibility_score": round(1.0 - mean_distance, 6),
            "mean_distance": round(mean_distance, 6),
            "pair_count": len(pair_reports),
            "fallback_pair_count": fallback_count,
            "missing_garment_ids": missing,
            "weakest_pair": weakest,
            "pairs": pair_reports,
        }

    def _unavailable_report(self) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "model": self.model_label,
            "reason": self.reason,
            "pair_count": 0,
        }

    def _condition_for(self, left: str | None, right: str | None) -> int | None:
        if not left or not right:
            return None
        return self.typespaces.get((left, right), self.typespaces.get((right, left)))

    @staticmethod
    def _type_pair_label(left: str | None, right: str | None) -> str:
        return " · ".join(value or "unknown" for value in (left, right))

    def _distance(self, left, right) -> float:
        torch = self._torch
        if self.metric is not None:
            return float(self.metric((left * right).unsqueeze(0)).squeeze().item())
        return float(torch.nn.functional.pairwise_distance(left.unsqueeze(0), right.unsqueeze(0), p=2).item())

    def _embeddings_for_items(self, items: list[dict[str, Any]], catalog):
        missing = []
        pending = []
        with self._cache_lock:
            for item in items:
                garment_id = str(item["garment_id"])
                if garment_id in self._image_cache:
                    continue
                try:
                    path = catalog.image_path(garment_id)
                except Exception:
                    path = None
                if path is None or not path.is_file():
                    missing.append(garment_id)
                    continue
                pending.append((garment_id, path))

        for start in range(0, len(pending), self.max_batch_size):
            batch = pending[start:start + self.max_batch_size]
            tensors = []
            valid_ids = []
            for garment_id, path in batch:
                try:
                    tensors.append(self._image_tensor(path))
                    valid_ids.append(garment_id)
                except Exception:
                    missing.append(garment_id)
            if not tensors:
                continue
            torch = self._torch
            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                encoded = self.model(batch_tensor).detach().cpu()
            with self._cache_lock:
                for garment_id, embedding in zip(valid_ids, encoded):
                    self._image_cache[garment_id] = embedding

        embeddings = {}
        with self._cache_lock:
            for item in items:
                garment_id = str(item["garment_id"])
                embedding = self._image_cache.get(garment_id)
                if embedding is None:
                    if garment_id not in missing:
                        missing.append(garment_id)
                else:
                    embeddings[garment_id] = embedding
        return embeddings, list(dict.fromkeys(missing))

    def _image_tensor(self, path: Path):
        torch = self._torch
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        scale = 112 / min(width, height)
        resized = image.resize((max(112, round(width * scale)), max(112, round(height * scale))), Image.Resampling.BILINEAR)
        left = (resized.width - 112) // 2
        top = (resized.height - 112) // 2
        cropped = resized.crop((left, top, left + 112, top + 112))
        array = np.asarray(cropped, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        tensor = torch.from_numpy(array)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std
