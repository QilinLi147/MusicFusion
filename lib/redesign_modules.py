"""Numerically safe implementations of the three MusicFusion redesign modules.

The classes in this file deliberately own only training-time state.  They do
not add encoders, heads, projectors, or inference-time branches.

Epoch indices are zero based:

* epochs ``0 .. warmup_epochs - 1``: ReliaPseudo and TriDistill are off;
* epoch ``warmup_epochs``: ReliaPseudo collects temporal statistics;
* epoch ``warmup_epochs + 1`` onward: the preceding epoch's pseudo mask is
  used by ReliaPseudo.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_VIEWS: Tuple[str, str, str] = ("mel", "coch", "fusion")
_EDGES: Tuple[Tuple[str, str], ...] = (
    ("fusion", "mel"),
    ("fusion", "coch"),
    ("mel", "coch"),
)


@contextmanager
def _autocast_disabled(device_type: str) -> Iterator[None]:
    """Disable an enclosing autocast region for FP32 state/loss arithmetic."""

    if hasattr(torch, "autocast"):
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    else:  # pragma: no cover - compatibility path for very old PyTorch.
        yield


@contextmanager
def freeze_bn_running_stats(module: nn.Module) -> Iterator[nn.Module]:
    """Freeze BatchNorm running statistics while leaving Dropout stochastic.

    Training BatchNorm layers keep using the current batch statistics, but
    temporarily stop tracking running buffers.  This makes a stochastic
    second forward comparable with the first training-mode forward.  Affine
    BatchNorm parameters keep ``requires_grad=True`` and all module training
    flags (including Dropout) retain their original values.
    """

    batch_norms = [
        child
        for child in module.modules()
        if isinstance(child, nn.modules.batchnorm._BatchNorm)
    ]
    original_tracking = [child.track_running_stats for child in batch_norms]
    try:
        for child in batch_norms:
            if child.training:
                child.track_running_stats = False
        yield module
    finally:
        for child, was_tracking in zip(batch_norms, original_tracking):
            child.track_running_stats = was_tracking


def _zero_from_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return a differentiable scalar zero on the tensors' device."""

    if not tensors:
        return torch.tensor(0.0, dtype=torch.float32)
    zero = tensors[0].float().sum() * 0.0
    for tensor in tensors[1:]:
        zero = zero + tensor.float().sum() * 0.0
    return zero


def _canonical_triplet(
    values: Mapping[str, torch.Tensor],
    suffix: str,
) -> Dict[str, torch.Tensor]:
    """Accept either canonical keys or the model's ``*_feat/logits`` keys."""

    result: Dict[str, torch.Tensor] = {}
    for view in _VIEWS:
        if view in values:
            result[view] = values[view]
        elif f"{view}_{suffix}" in values:
            result[view] = values[f"{view}_{suffix}"]
        else:
            raise KeyError(
                f"Missing {view!r}; expected {view!r} or {view + '_' + suffix!r}"
            )
    return result


def _validate_unique_indices(indices: torch.Tensor, num_samples: int) -> None:
    if indices.ndim != 1:
        raise ValueError("sample indices must be a one-dimensional tensor")
    if indices.numel() == 0:
        return
    if int(indices.min()) < 0 or int(indices.max()) >= num_samples:
        raise IndexError(
            f"sample index is outside [0, {num_samples - 1}]"
        )
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError("a batch must not contain duplicate immutable sample ids")


class ProtoAlign(nn.Module):
    """Shared supervised spherical class anchors for three existing views."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        momentum: float = 0.9,
        temperature: float = 0.5,
        eps: float = 1e-8,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("ProtoAlign requires at least two classes")
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.enabled = bool(enabled)

        # This is the only prototype bank.  It is intentionally not a
        # Parameter and therefore cannot be included in the optimizer.
        self.register_buffer(
            "prototypes",
            torch.zeros(num_classes, feature_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "prototype_initialized",
            torch.zeros(num_classes, dtype=torch.bool),
        )
        self.register_buffer(
            "update_count",
            torch.zeros(num_classes, dtype=torch.long),
            persistent=True,
        )

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        views = _canonical_triplet(features, "feat")
        tensors = [views[name] for name in _VIEWS]
        if not self.enabled:
            zero = _zero_from_tensors(tensors)
            return zero, {
                "pa_loss": zero.detach(),
                "pa_gdd": zero.detach(),
                "pa_lsd": zero.detach(),
                "pa_present_classes": zero.detach(),
                "pa_absent_classes": zero.detach(),
            }

        first = tensors[0]
        if first.ndim != 2 or first.shape[1] != self.feature_dim:
            raise ValueError(
                f"each feature must have shape [N, {self.feature_dim}]"
            )
        batch_size = first.shape[0]
        for tensor in tensors[1:]:
            if tensor.shape != first.shape:
                raise ValueError("mel, coch, and fusion features must have equal shapes")
            if tensor.device != first.device:
                raise ValueError("all feature views must be on the same device")

        labels = labels.to(device=first.device, dtype=torch.long).reshape(-1)
        if labels.numel() != batch_size:
            raise ValueError("labels and features must have the same batch size")
        if labels.numel() and (
            int(labels.min()) < 0 or int(labels.max()) >= self.num_classes
        ):
            raise ValueError("labels contain an invalid class index")
        if batch_size == 0:
            zero = _zero_from_tensors(tensors)
            return zero, {
                "pa_loss": zero.detach(),
                "pa_gdd": zero.detach(),
                "pa_lsd": zero.detach(),
                "pa_present_classes": zero.detach(),
                "pa_absent_classes": torch.tensor(
                    float(self.num_classes),
                    device=first.device,
                    dtype=torch.float32,
                ),
            }
        if self.prototypes.device != first.device:
            raise ValueError("move ProtoAlign to the feature device before use")
        if self.prototypes.dtype != torch.float32:
            raise TypeError("ProtoAlign prototypes must remain FP32")

        with _autocast_disabled(first.device.type):
            unit_views = [
                F.normalize(tensor.float(), p=2, dim=1, eps=self.eps)
                for tensor in tensors
            ]
            unit_prototypes = F.normalize(
                self.prototypes.float(), p=2, dim=1, eps=self.eps
            )
            view_nll = []
            for unit_features in unit_views:
                relation_logits = (
                    unit_features @ unit_prototypes.transpose(0, 1)
                ) / self.temperature
                view_nll.append(
                    F.cross_entropy(relation_logits, labels, reduction="none")
                )

            # Shape [N, 3], allowing GDD and LSD to aggregate exactly the same
            # atomic supervised prototype NLL values.
            atom_nll = torch.stack(view_nll, dim=1)
            gdd = atom_nll.mean()
            present_classes = torch.unique(labels, sorted=True)
            class_means = [
                atom_nll[labels == class_index].mean()
                for class_index in present_classes
            ]
            lsd = torch.stack(class_means).mean()
            loss = 0.5 * (gdd + lsd)

        return loss, {
            "pa_loss": loss.detach(),
            "pa_gdd": gdd.detach(),
            "pa_lsd": lsd.detach(),
            "pa_present_classes": torch.tensor(
                float(present_classes.numel()),
                device=first.device,
                dtype=torch.float32,
            ),
            "pa_absent_classes": torch.tensor(
                float(self.num_classes - present_classes.numel()),
                device=first.device,
                dtype=torch.float32,
            ),
        }

    @torch.no_grad()
    def update(
        self,
        fusion_features: torch.Tensor,
        labels: torch.Tensor,
        step_succeeded: bool = True,
    ) -> int:
        """Commit one GT-only EMA update per class after a successful step."""

        if not self.enabled or not step_succeeded:
            return 0
        if fusion_features.ndim != 2 or fusion_features.shape[1] != self.feature_dim:
            raise ValueError(
                f"fusion_features must have shape [N, {self.feature_dim}]"
            )
        if self.prototypes.device != fusion_features.device:
            raise ValueError("move ProtoAlign to the feature device before use")
        if self.prototypes.dtype != torch.float32:
            raise TypeError("ProtoAlign prototypes must remain FP32")

        labels = labels.to(
            device=fusion_features.device, dtype=torch.long
        ).reshape(-1)
        if labels.numel() != fusion_features.shape[0]:
            raise ValueError("labels and fusion_features must have equal batch size")
        if labels.numel() == 0:
            return 0
        if int(labels.min()) < 0 or int(labels.max()) >= self.num_classes:
            raise ValueError("labels contain an invalid class index")

        with _autocast_disabled(fusion_features.device.type):
            unit_features = F.normalize(
                fusion_features.detach().float(), p=2, dim=1, eps=self.eps
            )
            if not torch.isfinite(unit_features).all():
                raise FloatingPointError(
                    "non-finite fusion feature: prototype update was not committed"
                )

            proposals = []
            for class_index in torch.unique(labels, sorted=True).tolist():
                class_mask = labels == class_index
                class_mean = unit_features[class_mask].mean(dim=0)
                if not torch.isfinite(class_mean).all():
                    raise FloatingPointError(
                        "non-finite class mean: prototype update was not committed"
                    )
                if bool(self.prototype_initialized[class_index]):
                    candidate = (
                        self.momentum * self.prototypes[class_index]
                        + (1.0 - self.momentum) * class_mean
                    )
                    candidate = F.normalize(
                        candidate, p=2, dim=0, eps=self.eps
                    )
                else:
                    candidate = F.normalize(
                        class_mean, p=2, dim=0, eps=self.eps
                    )
                if not torch.isfinite(candidate).all():
                    raise FloatingPointError(
                        "non-finite prototype proposal: update was not committed"
                    )
                proposals.append((class_index, candidate.float()))

            # Commit prototypes and their persistent counters together only
            # after every proposal from this successful optimizer step is
            # known to be finite.
            for class_index, candidate in proposals:
                self.prototypes[class_index].copy_(candidate.float())
                self.prototype_initialized[class_index] = True
                self.update_count[class_index] += 1
        return len(proposals)

    @torch.no_grad()
    def diagnostics(
        self,
        epoch_start_update_count: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return detached prototype-state diagnostics without changing state.

        ``epoch_start_update_count`` allows the trainer to distinguish a class
        absent from the current epoch from a class that has never initialized.
        """

        if self.prototypes.dtype != torch.float32:
            raise TypeError("ProtoAlign prototypes must remain FP32")
        initialized = self.prototype_initialized
        norms = self.prototypes.float().norm(p=2, dim=1)
        initialized_norms = norms[initialized]
        zero = self.prototypes.new_zeros(())
        if initialized_norms.numel():
            norm_mean = initialized_norms.mean()
            norm_min = initialized_norms.min()
            norm_max = initialized_norms.max()
        else:
            norm_mean = norm_min = norm_max = zero

        initialized_indices = torch.nonzero(
            initialized, as_tuple=False
        ).flatten()
        if initialized_indices.numel() >= 2:
            unit = F.normalize(
                self.prototypes.index_select(0, initialized_indices).float(),
                p=2,
                dim=1,
                eps=self.eps,
            )
            cosine_matrix = unit @ unit.transpose(0, 1)
            pair_mask = torch.triu(
                torch.ones_like(cosine_matrix, dtype=torch.bool), diagonal=1
            )
            pairwise_cosine = cosine_matrix[pair_mask]
            cosine_mean = pairwise_cosine.mean()
            cosine_min = pairwise_cosine.min()
            cosine_max = pairwise_cosine.max()
            pair_count = pairwise_cosine.new_tensor(
                float(pairwise_cosine.numel())
            )
        else:
            cosine_mean = cosine_min = cosine_max = zero
            pair_count = zero

        if epoch_start_update_count is None:
            epoch_updates = self.update_count.clone()
        else:
            baseline = torch.as_tensor(
                epoch_start_update_count,
                dtype=torch.long,
                device=self.update_count.device,
            ).reshape(-1)
            if baseline.shape != self.update_count.shape:
                raise ValueError(
                    "epoch_start_update_count must match update_count shape"
                )
            if bool((baseline > self.update_count).any()):
                raise ValueError(
                    "epoch_start_update_count cannot exceed current counts"
                )
            epoch_updates = self.update_count - baseline

        stats: Dict[str, torch.Tensor] = {
            "pa_initialized_classes": initialized.sum().float(),
            "pa_uninitialized_classes": (~initialized).sum().float(),
            "pa_prototype_norm_mean": norm_mean,
            "pa_prototype_norm_min": norm_min,
            "pa_prototype_norm_max": norm_max,
            "pa_interclass_cosine_mean": cosine_mean,
            "pa_interclass_cosine_min": cosine_min,
            "pa_interclass_cosine_max": cosine_max,
            "pa_interclass_pairs": pair_count,
            "pa_update_count_total": self.update_count.sum().float(),
            "pa_epoch_update_count": epoch_updates.sum().float(),
            "pa_epoch_absent_classes": (epoch_updates == 0).sum().float(),
        }
        for class_index in range(self.num_classes):
            stats[f"pa_update_count_class_{class_index}"] = self.update_count[
                class_index
            ].float()
            stats[
                f"pa_epoch_update_count_class_{class_index}"
            ] = epoch_updates[class_index].float()
            stats[f"pa_prototype_norm_class_{class_index}"] = norms[
                class_index
            ]
            stats[f"pa_initialized_class_{class_index}"] = initialized[
                class_index
            ].float()
        return {key: value.detach() for key, value in stats.items()}


@dataclass
class _PendingPseudoUpdate:
    indices: torch.Tensor
    current_probs: torch.Tensor
    reliability: torch.Tensor
    confidence: torch.Tensor
    stability: torch.Tensor
    pseudo_labels: torch.Tensor
    versions: torch.Tensor


class ReliaPseudo(nn.Module):
    """Sample-id EMA pseudo curriculum with transactional state updates."""

    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        class_prior: Sequence[float],
        total_epochs: int,
        ema_momentum: float = 0.9,
        safety_floor: float = 0.60,
        min_coverage: float = 0.10,
        max_coverage: float = 0.60,
        warmup_epochs: int = 5,
        eps: float = 1e-8,
        enabled: bool = True,
        training_size: Optional[int] = None,
        coverage_schedule: str = "linear",
        target_mode: str = "soft",
        soft_ema_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if num_classes < 2:
            raise ValueError("ReliaPseudo requires at least two classes")
        if total_epochs < 1:
            raise ValueError("total_epochs must be positive")
        if not 0.0 <= ema_momentum < 1.0:
            raise ValueError("ema_momentum must be in [0, 1)")
        if not 0.0 <= safety_floor <= 1.0:
            raise ValueError("safety_floor must be in [0, 1]")
        if not 0.0 <= min_coverage <= max_coverage <= 1.0:
            raise ValueError("coverage bounds must satisfy 0 <= min <= max <= 1")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if training_size is None:
            training_size = num_samples
        if not 1 <= int(training_size) <= num_samples:
            raise ValueError("training_size must be in [1, num_samples]")
        if coverage_schedule not in {"linear", "cosine"}:
            raise ValueError("coverage_schedule must be 'linear' or 'cosine'")
        if target_mode not in {
            "soft",
            "hard",
            "oracle",
            "consistency",
            "agreement_consistency",
        }:
            raise ValueError(
                "target_mode must be 'soft', 'hard', 'oracle', "
                "'consistency', or 'agreement_consistency'"
            )
        if not 0.0 <= soft_ema_weight <= 0.5:
            raise ValueError("soft_ema_weight must be in [0, 0.5]")

        prior = torch.as_tensor(class_prior, dtype=torch.float32).reshape(-1)
        if prior.numel() != num_classes:
            raise ValueError("class_prior length must equal num_classes")
        if not torch.isfinite(prior).all() or bool((prior < 0).any()):
            raise ValueError("class_prior must be finite and non-negative")
        if float(prior.sum()) <= 0.0:
            raise ValueError("class_prior must have positive mass")
        prior = prior / prior.sum()

        self.num_samples = int(num_samples)
        self.training_size = int(training_size)
        self.num_classes = int(num_classes)
        self.total_epochs = int(total_epochs)
        self.ema_momentum = float(ema_momentum)
        self.safety_floor = float(safety_floor)
        self.min_coverage = float(min_coverage)
        self.max_coverage = float(max_coverage)
        self.warmup_epochs = int(warmup_epochs)
        self.eps = float(eps)
        self.enabled = bool(enabled)
        self.coverage_schedule = str(coverage_schedule)
        self.target_mode = str(target_mode)
        self.soft_ema_weight = float(soft_ema_weight)

        self.register_buffer(
            "class_prior", prior.clone(), persistent=True
        )
        self.register_buffer(
            "ema_probs",
            torch.zeros(num_samples, num_classes, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "initialized",
            torch.zeros(num_samples, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "epoch_mask",
            torch.zeros(num_samples, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "last_reliability",
            torch.zeros(num_samples, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "last_confidence",
            torch.zeros(num_samples, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "last_stability",
            torch.zeros(num_samples, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "last_pseudo_labels",
            torch.full((num_samples,), -1, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "seen_this_epoch",
            torch.zeros(num_samples, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "sample_versions",
            torch.zeros(num_samples, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "current_epoch",
            torch.tensor(-1, dtype=torch.long),
            persistent=True,
        )
        self._pending: Optional[_PendingPseudoUpdate] = None

    def _indices(self, indices: torch.Tensor) -> torch.Tensor:
        result = torch.as_tensor(
            indices, dtype=torch.long, device=self.ema_probs.device
        ).reshape(-1)
        _validate_unique_indices(result, self.num_samples)
        return result

    def _probabilities(self, values: torch.Tensor) -> torch.Tensor:
        probs = values.detach().to(
            device=self.ema_probs.device, dtype=torch.float32
        )
        if probs.ndim != 2 or probs.shape[1] != self.num_classes:
            raise ValueError(
                f"probabilities must have shape [N, {self.num_classes}]"
            )
        if not torch.isfinite(probs).all():
            raise FloatingPointError("non-finite pseudo probability")
        if bool((probs < 0).any()):
            raise ValueError("current_coch_probs must be non-negative probabilities")
        denominator = probs.sum(dim=1, keepdim=True)
        if bool((denominator <= 0).any()):
            raise ValueError("each probability row must have positive mass")
        return (probs / denominator).clamp(min=0.0, max=1.0)

    @torch.no_grad()
    def initialize(
        self,
        indices: torch.Tensor,
        coch_probs: torch.Tensor,
    ) -> None:
        """Initialize sample EMA from a no-grad complete-training-set pass."""

        if self._pending is not None:
            raise RuntimeError("commit or discard the pending batch before initialize")
        sample_ids = self._indices(indices)
        probabilities = self._probabilities(coch_probs)
        if probabilities.shape[0] != sample_ids.numel():
            raise ValueError("indices and coch_probs must have equal batch size")
        self.ema_probs.index_copy_(0, sample_ids, probabilities)
        self.initialized.index_fill_(0, sample_ids, True)
        old_versions = self.sample_versions.index_select(0, sample_ids)
        self.sample_versions.index_copy_(0, sample_ids, old_versions + 1)

    @torch.no_grad()
    def begin_epoch(self, epoch: int) -> None:
        if self._pending is not None:
            raise RuntimeError("cannot begin an epoch with an uncommitted batch")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.current_epoch.fill_(int(epoch))
        self.seen_this_epoch.zero_()

    def _resolve_epoch(self, epoch: Optional[int]) -> int:
        if epoch is None:
            epoch_value = int(self.current_epoch)
            if epoch_value < 0:
                raise RuntimeError("call begin_epoch before ReliaPseudo.forward")
            return epoch_value
        epoch_value = int(epoch)
        if int(self.current_epoch) != epoch_value:
            self.begin_epoch(epoch_value)
        return epoch_value

    def forward(
        self,
        second_fusion_logits: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
        current_coch_probs: torch.Tensor,
        epoch: Optional[int] = None,
        first_fusion_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self._pending is not None:
            raise RuntimeError(
                "ReliaPseudo.forward called before the previous batch was committed"
            )
        if not self.enabled:
            zero = second_fusion_logits.float().sum() * 0.0
            return zero, {
                "rp_loss": zero.detach(),
                "rp_selected": zero.detach(),
                "rp_coverage": zero.detach(),
                "rp_reliability_mean": zero.detach(),
                "rp_selected_reliability_mean": zero.detach(),
                "rp_confidence_mean": zero.detach(),
                "rp_stability_mean": zero.detach(),
                "rp_teacher_agreement_mean": zero.detach(),
            }
        epoch_value = self._resolve_epoch(epoch)
        sample_ids = self._indices(indices)
        if self.ema_probs.device != second_fusion_logits.device:
            raise ValueError("move ReliaPseudo to the logits device before use")
        if second_fusion_logits.ndim != 2 or (
            second_fusion_logits.shape[1] != self.num_classes
        ):
            raise ValueError(
                f"second_fusion_logits must have shape [N, {self.num_classes}]"
            )
        if second_fusion_logits.shape[0] != sample_ids.numel():
            raise ValueError(
                "the stochastic second forward must contain the complete batch"
            )

        labels = labels.to(
            device=second_fusion_logits.device, dtype=torch.long
        ).reshape(-1)
        if labels.numel() != sample_ids.numel():
            raise ValueError("labels and sample indices must have equal batch size")
        if labels.numel() and (
            int(labels.min()) < 0 or int(labels.max()) >= self.num_classes
        ):
            raise ValueError("labels contain an invalid class index")

        current_probs = self._probabilities(current_coch_probs)
        if current_probs.shape[0] != sample_ids.numel():
            raise ValueError("indices and current_coch_probs must have equal batch size")
        first_probs: Optional[torch.Tensor] = None
        if self.target_mode in {"consistency", "agreement_consistency"}:
            if first_fusion_probs is None:
                raise ValueError(
                    "first_fusion_probs is required for consistency modes"
                )
            first_probs = self._probabilities(first_fusion_probs)
            if first_probs.shape[0] != sample_ids.numel():
                raise ValueError(
                    "indices and first_fusion_probs must have equal batch size"
                )

        with _autocast_disabled(second_fusion_logits.device.type):
            initialized = self.initialized.index_select(0, sample_ids)
            stored = self.ema_probs.index_select(0, sample_ids).detach().float()
            old_probs = torch.where(
                initialized.unsqueeze(1), stored, current_probs
            )
            old_probs = old_probs / old_probs.sum(
                dim=1, keepdim=True
            ).clamp_min(self.eps)

            midpoint = 0.5 * (old_probs + current_probs)
            log_old = old_probs.clamp_min(self.eps).log()
            log_current = current_probs.clamp_min(self.eps).log()
            log_midpoint = midpoint.clamp_min(self.eps).log()
            js = 0.5 * (
                (old_probs * (log_old - log_midpoint)).sum(dim=1)
                + (
                    current_probs
                    * (log_current - log_midpoint)
                ).sum(dim=1)
            )
            stability = (1.0 - js / math.log(2.0)).clamp(0.0, 1.0)
            confidence, pseudo_labels = old_probs.max(dim=1)
            reliability = (confidence * stability).clamp(0.0, 1.0).detach()

            raw_mask = self.epoch_mask.index_select(0, sample_ids)
            active_epoch = epoch_value >= self.warmup_epochs + 1
            if self.enabled and active_epoch:
                selected = raw_mask & initialized
            else:
                selected = torch.zeros_like(raw_mask)
            if first_probs is not None:
                teacher_agreement = (
                    first_probs.argmax(dim=1) == pseudo_labels
                )
            else:
                teacher_agreement = torch.zeros_like(raw_mask)
            if self.target_mode == "agreement_consistency":
                selected = selected & teacher_agreement

            annotation = F.one_hot(
                labels, num_classes=self.num_classes
            ).float()
            if self.target_mode == "soft":
                target = (
                    (1.0 - self.soft_ema_weight) * annotation
                    + self.soft_ema_weight * old_probs
                )
            elif self.target_mode == "hard":
                target = F.one_hot(
                    pseudo_labels, num_classes=self.num_classes
                ).float()
            elif self.target_mode == "oracle":
                target = annotation
            else:
                if first_probs is None:  # Defensive; validated above.
                    raise RuntimeError("missing consistency teacher")
                target = first_probs
            target = target.detach()
            log_prediction = F.log_softmax(
                second_fusion_logits.float(), dim=1
            )
            if self.target_mode in {"consistency", "agreement_consistency"}:
                per_sample_loss = (
                    target
                    * (target.clamp_min(self.eps).log() - log_prediction)
                ).sum(dim=1)
            else:
                per_sample_loss = -(target * log_prediction).sum(dim=1)
            weights = selected.float() * reliability
            selected_count = selected.float().sum()
            loss = (weights * per_sample_loss).sum() / selected_count.clamp_min(
                1.0
            )

        versions = self.sample_versions.index_select(0, sample_ids).clone()
        self._pending = _PendingPseudoUpdate(
            indices=sample_ids.clone(),
            current_probs=current_probs.clone(),
            reliability=reliability.clone(),
            confidence=confidence.detach().clone(),
            stability=stability.detach().clone(),
            pseudo_labels=pseudo_labels.detach().clone(),
            versions=versions,
        )
        selected_reliability = reliability[selected]
        selected_reliability_mean = (
            selected_reliability.mean()
            if selected_reliability.numel()
            else reliability.new_zeros(())
        )
        return loss, {
            "rp_loss": loss.detach(),
            "rp_selected": selected_count.detach(),
            "rp_coverage": selected.float().mean().detach()
            if selected.numel()
            else loss.detach() * 0.0,
            "rp_reliability_mean": reliability.mean().detach()
            if reliability.numel()
            else loss.detach() * 0.0,
            "rp_selected_reliability_mean": selected_reliability_mean.detach(),
            "rp_confidence_mean": confidence.mean().detach()
            if confidence.numel()
            else loss.detach() * 0.0,
            "rp_stability_mean": stability.mean().detach()
            if stability.numel()
            else loss.detach() * 0.0,
            "rp_teacher_agreement_mean": teacher_agreement.float().mean().detach()
            if teacher_agreement.numel()
            else loss.detach() * 0.0,
        }

    @torch.no_grad()
    def commit(
        self,
        indices: torch.Tensor,
        current_coch_probs: Optional[torch.Tensor] = None,
        step_succeeded: bool = True,
    ) -> bool:
        """Commit prediction EMA only after a finite optimizer step succeeds."""

        if self._pending is None:
            raise RuntimeError("there is no pending ReliaPseudo batch to commit")
        pending = self._pending
        self._pending = None
        sample_ids = self._indices(indices)
        if not torch.equal(sample_ids, pending.indices):
            raise RuntimeError("commit sample ids do not match the pending batch")
        if not step_succeeded:
            return False

        if current_coch_probs is None:
            current_probs = pending.current_probs
        else:
            current_probs = self._probabilities(current_coch_probs)
            if current_probs.shape != pending.current_probs.shape:
                raise ValueError("commit probabilities have an unexpected shape")
            if not torch.allclose(
                current_probs,
                pending.current_probs,
                rtol=1e-5,
                atol=1e-7,
            ):
                raise RuntimeError(
                    "commit probabilities differ from those read during forward"
                )

        current_versions = self.sample_versions.index_select(0, sample_ids)
        if not torch.equal(current_versions, pending.versions):
            raise RuntimeError("stale ReliaPseudo update; sample EMA already changed")

        was_initialized = self.initialized.index_select(0, sample_ids)
        old_probs = self.ema_probs.index_select(0, sample_ids)
        blended = (
            self.ema_momentum * old_probs
            + (1.0 - self.ema_momentum) * current_probs
        )
        updated = torch.where(
            was_initialized.unsqueeze(1), blended, current_probs
        )
        updated = updated / updated.sum(
            dim=1, keepdim=True
        ).clamp_min(self.eps)
        if not torch.isfinite(updated).all():
            raise FloatingPointError(
                "non-finite EMA proposal: ReliaPseudo state was not committed"
            )

        self.ema_probs.index_copy_(0, sample_ids, updated.float())
        self.initialized.index_fill_(0, sample_ids, True)
        self.last_reliability.index_copy_(
            0, sample_ids, pending.reliability.float()
        )
        self.last_confidence.index_copy_(
            0, sample_ids, pending.confidence.float()
        )
        self.last_stability.index_copy_(
            0, sample_ids, pending.stability.float()
        )
        self.last_pseudo_labels.index_copy_(
            0, sample_ids, pending.pseudo_labels.long()
        )
        self.seen_this_epoch.index_fill_(0, sample_ids, True)
        self.sample_versions.index_copy_(
            0, sample_ids, current_versions + 1
        )
        return True

    def coverage_for_epoch(self, epoch: int) -> float:
        """Declared coverage cap for an epoch that is eligible to use a mask."""

        first_active = self.warmup_epochs + 1
        if epoch < first_active:
            return 0.0
        final_epoch = max(first_active, self.total_epochs - 1)
        progress = (epoch - first_active) / max(1, final_epoch - first_active)
        progress = min(1.0, max(0.0, progress))
        if self.coverage_schedule == "cosine":
            progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.min_coverage + progress * (
            self.max_coverage - self.min_coverage
        )

    @torch.no_grad()
    def finalize_epoch(self, epoch: int) -> Dict[str, torch.Tensor]:
        """Build the fixed next-epoch mask from committed current-epoch state."""

        if self._pending is not None:
            raise RuntimeError("commit or discard the pending batch before finalize")
        if int(self.current_epoch) != int(epoch):
            raise RuntimeError("finalize_epoch does not match current_epoch")

        next_epoch = int(epoch) + 1
        rho = self.coverage_for_epoch(next_epoch)
        next_mask = torch.zeros_like(self.epoch_mask)
        class_counts = torch.zeros(
            self.num_classes,
            dtype=torch.long,
            device=self.epoch_mask.device,
        )
        seen_class_counts = torch.zeros_like(class_counts)
        eligible_class_counts = torch.zeros_like(class_counts)
        # ``num_samples`` sizes the sample-id address space and can include
        # outer-validation ids. The curriculum budget uses the fixed number of
        # training annotations; successful commits only define eligibility.
        seen_count = int(self.seen_this_epoch.sum())

        eligible_base = (
            self.seen_this_epoch
            & self.initialized
            & (self.last_reliability >= self.safety_floor)
        )
        for class_index in range(self.num_classes):
            seen_class_counts[class_index] = (
                self.seen_this_epoch
                & self.initialized
                & (self.last_pseudo_labels == class_index)
            ).sum()
            eligible_class_counts[class_index] = (
                eligible_base
                & (self.last_pseudo_labels == class_index)
            ).sum()

        if self.enabled and rho > 0.0 and seen_count > 0:
            total_budget = int(math.floor(rho * self.training_size))
            for class_index in range(self.num_classes):
                class_budget = int(
                    math.floor(
                        float(self.class_prior[class_index]) * total_budget
                        + 1e-9
                    )
                )
                if class_budget <= 0:
                    continue
                candidates = torch.nonzero(
                    eligible_base
                    & (self.last_pseudo_labels == class_index),
                    as_tuple=False,
                ).flatten()
                if candidates.numel() == 0:
                    continue
                candidate_scores = self.last_reliability.index_select(
                    0, candidates
                )
                try:
                    order = torch.argsort(
                        candidate_scores, descending=True, stable=True
                    )
                except TypeError:  # pragma: no cover - older PyTorch fallback.
                    # Tiny index-dependent decrement gives deterministic
                    # sample-id priority without changing meaningful ranks.
                    adjusted = candidate_scores.double() - (
                        candidates.double()
                        * torch.finfo(torch.float64).eps
                    )
                    order = torch.argsort(adjusted, descending=True)
                chosen = candidates.index_select(
                    0, order[: min(class_budget, candidates.numel())]
                )
                next_mask.index_fill_(0, chosen, True)
                class_counts[class_index] = chosen.numel()

        self.epoch_mask.copy_(next_mask)
        selected = next_mask.sum().float()
        stats: Dict[str, torch.Tensor] = {
            "rp_next_rho": torch.tensor(
                rho, dtype=torch.float32, device=self.epoch_mask.device
            ),
            "rp_epoch_seen": torch.tensor(
                float(seen_count),
                dtype=torch.float32,
                device=self.epoch_mask.device,
            ),
            "rp_next_selected": selected,
            "rp_next_coverage": selected / float(self.training_size),
        }
        for class_index in range(self.num_classes):
            stats[f"rp_next_selected_class_{class_index}"] = class_counts[
                class_index
            ].float()
            stats[f"rp_seen_class_{class_index}"] = seen_class_counts[
                class_index
            ].float()
            stats[
                f"rp_eligible_class_{class_index}"
            ] = eligible_class_counts[class_index].float()
            stats[f"rp_next_coverage_class_{class_index}"] = (
                class_counts[class_index].float()
                / seen_class_counts[class_index].float().clamp_min(1.0)
            )
            stats[f"rp_next_eligible_coverage_class_{class_index}"] = (
                class_counts[class_index].float()
                / eligible_class_counts[class_index].float().clamp_min(1.0)
            )
        return stats


class TriDistill(nn.Module):
    """GT-consistent routed symmetric distillation over the full triangle."""

    def __init__(
        self,
        temperature: float = 2.0,
        warmup_epochs: int = 5,
        eps: float = 1e-8,
        enabled: bool = True,
        tie_priority: Sequence[str] = ("fusion", "mel", "coch"),
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if tuple(sorted(tie_priority)) != tuple(sorted(_VIEWS)):
            raise ValueError("tie_priority must contain fusion, mel, and coch once")
        self.temperature = float(temperature)
        self.warmup_epochs = int(warmup_epochs)
        self.eps = float(eps)
        self.enabled = bool(enabled)
        self.tie_priority = tuple(tie_priority)
        self._priority = {
            name: index for index, name in enumerate(self.tie_priority)
        }

    def _probabilities_and_margins(
        self,
        logits: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        canonical = _canonical_triplet(logits, "logits")
        first = canonical["fusion"]
        if first.ndim != 2 or first.shape[1] < 2:
            raise ValueError("each branch logits tensor must have shape [N, C>=2]")
        for view in _VIEWS:
            if canonical[view].shape != first.shape:
                raise ValueError("all three logits tensors must have equal shapes")
            if canonical[view].device != first.device:
                raise ValueError("all logits tensors must be on the same device")
        labels = labels.to(device=first.device, dtype=torch.long).reshape(-1)
        if labels.numel() != first.shape[0]:
            raise ValueError("labels and logits must have equal batch size")
        if labels.numel() and (
            int(labels.min()) < 0 or int(labels.max()) >= first.shape[1]
        ):
            raise ValueError("labels contain an invalid class index")

        probabilities: Dict[str, torch.Tensor] = {}
        margins: Dict[str, torch.Tensor] = {}
        with _autocast_disabled(first.device.type):
            for view in _VIEWS:
                probability = F.softmax(
                    canonical[view].float() / self.temperature, dim=1
                )
                probabilities[view] = probability
                gt = probability.gather(1, labels.unsqueeze(1)).squeeze(1)
                not_gt = probability.masked_fill(
                    F.one_hot(
                        labels, num_classes=probability.shape[1]
                    ).bool(),
                    float("-inf"),
                )
                other = not_gt.max(dim=1).values
                margins[view] = (gt - other).clamp_min(0.0)
        return probabilities, margins

    def route_teachers(
        self,
        logits: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return detached diagnostic route codes (0 inactive, 1 left, 2 right)."""

        _, margins = self._probabilities_and_margins(logits, labels)
        routes: Dict[str, torch.Tensor] = {}
        for left, right in _EDGES:
            left_margin = margins[left]
            right_margin = margins[right]
            active = torch.maximum(left_margin, right_margin) > 0.0
            if self._priority[left] < self._priority[right]:
                teacher_left = left_margin >= right_margin
            else:
                teacher_left = left_margin > right_margin
            routes[f"{left}-{right}"] = torch.where(
                active,
                torch.where(
                    teacher_left,
                    torch.ones_like(left_margin, dtype=torch.long),
                    torch.full_like(left_margin, 2, dtype=torch.long),
                ),
                torch.zeros_like(left_margin, dtype=torch.long),
            ).detach()
        return routes

    def forward(
        self,
        logits: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        canonical = _canonical_triplet(logits, "logits")
        tensors = [canonical[name] for name in _VIEWS]
        if not self.enabled or int(epoch) < self.warmup_epochs:
            zero = _zero_from_tensors(tensors)
            return zero, {
                "td_loss": zero.detach(),
                "td_active_edges": zero.detach(),
                "td_weight_sum": zero.detach(),
                "td_active_edge_fusion_mel": zero.detach(),
                "td_active_edge_fusion_coch": zero.detach(),
                "td_active_edge_mel_coch": zero.detach(),
                "td_teacher_fusion": zero.detach(),
                "td_teacher_mel": zero.detach(),
                "td_teacher_coch": zero.detach(),
            }

        probabilities, margins = self._probabilities_and_margins(
            canonical, labels
        )
        numerator = _zero_from_tensors(tensors)
        denominator = numerator.detach().clone()
        active_count = denominator.clone()
        teacher_counts = {
            view: denominator.clone() for view in _VIEWS
        }
        active_edge_counts = {
            edge: denominator.clone() for edge in _EDGES
        }

        with _autocast_disabled(tensors[0].device.type):
            for left, right in _EDGES:
                left_margin = margins[left]
                right_margin = margins[right]
                active = torch.maximum(left_margin, right_margin) > 0.0
                if self._priority[left] < self._priority[right]:
                    teacher_left = left_margin >= right_margin
                else:
                    teacher_left = left_margin > right_margin

                teacher = torch.where(
                    teacher_left.unsqueeze(1),
                    probabilities[left],
                    probabilities[right],
                ).detach()
                student = torch.where(
                    teacher_left.unsqueeze(1),
                    probabilities[right],
                    probabilities[left],
                )
                teacher_margin = torch.where(
                    teacher_left, left_margin, right_margin
                ).detach()
                weight = active.float() * teacher_margin

                log_teacher = teacher.clamp_min(self.eps).log()
                log_student = student.clamp_min(self.eps).log()
                kl_teacher_student = (
                    teacher * (log_teacher - log_student)
                ).sum(dim=1)
                kl_student_teacher = (
                    student * (log_student - log_teacher)
                ).sum(dim=1)
                symmetric_kl = 0.5 * (
                    kl_teacher_student + kl_student_teacher
                )
                numerator = numerator + (weight * symmetric_kl).sum()
                denominator = denominator + weight.sum()
                active_count = active_count + active.float().sum()
                active_edge_counts[(left, right)] = (
                    active_edge_counts[(left, right)]
                    + active.float().sum()
                )
                teacher_counts[left] = teacher_counts[left] + (
                    active & teacher_left
                ).float().sum()
                teacher_counts[right] = teacher_counts[right] + (
                    active & ~teacher_left
                ).float().sum()

            loss = (
                self.temperature * self.temperature
                * numerator
                / denominator.clamp_min(1.0)
            )

        return loss, {
            "td_loss": loss.detach(),
            "td_active_edges": active_count.detach(),
            "td_weight_sum": denominator.detach(),
            "td_active_edge_fusion_mel": active_edge_counts[
                ("fusion", "mel")
            ].detach(),
            "td_active_edge_fusion_coch": active_edge_counts[
                ("fusion", "coch")
            ].detach(),
            "td_active_edge_mel_coch": active_edge_counts[
                ("mel", "coch")
            ].detach(),
            "td_teacher_fusion": teacher_counts["fusion"].detach(),
            "td_teacher_mel": teacher_counts["mel"].detach(),
            "td_teacher_coch": teacher_counts["coch"].detach(),
        }


__all__ = [
    "ProtoAlign",
    "ReliaPseudo",
    "TriDistill",
    "freeze_bn_running_stats",
]
