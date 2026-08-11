#!/usr/bin/env python3
"""Public training entry point for the complete MusicFusion model."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from lib.data import MusicEmotionDataset, get_dataset_stats, stratified_split
from lib.models import MusicFusionModel
from lib.redesign_modules import (
    ProtoAlign,
    ReliaPseudo,
    TriDistill,
    freeze_bn_running_stats,
)
from lib.train_utils import set_seed


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _jsonl_append(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _hash_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _code_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _make_loader(
    subset: Subset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "dataset": subset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "generator": generator,
    }
    if workers > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )
    return DataLoader(**kwargs)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    labels_all = []
    probabilities_all = []
    loss_sum = 0.0
    sample_count = 0
    for batch in loader:
        mel = batch["mel"].to(device, non_blocking=True)
        coch = batch["coch"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        outputs = model(mel, coch)
        logits = outputs["fusion_logits"].float()
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
        probabilities = F.softmax(logits, dim=1)[:, 1]
        labels_all.append(labels.cpu().numpy())
        probabilities_all.append(probabilities.cpu().numpy())
        sample_count += labels.numel()
    if was_training:
        model.train()

    y_true = np.concatenate(labels_all).astype(np.int64)
    y_prob = np.concatenate(probabilities_all).astype(np.float64)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    return {
        "loss": loss_sum / max(1, sample_count),
        "acc": float((y_pred == y_true).mean()),
        "f1_positive": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_positive": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_positive": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "auc": float(roc_auc_score(y_true, y_prob))
        if np.unique(y_true).size == 2
        else 0.5,
    }


@torch.no_grad()
def initialize_relia_state(
    relia: ReliaPseudo,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    was_training = model.training
    model.eval()
    for batch in loader:
        mel = batch["mel"].to(device, non_blocking=True)
        coch = batch["coch"].to(device, non_blocking=True)
        indices = batch["index"].to(device, non_blocking=True)
        coch_probs = F.softmax(model(mel, coch)["coch_logits"].float(), dim=1)
        relia.initialize(indices, coch_probs)
    if was_training:
        model.train()


def _finite_gradients(model: nn.Module) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def _accumulate_diagnostics(
    sums: Dict[str, torch.Tensor],
    observations: Dict[str, int],
    stats: Mapping[str, torch.Tensor],
) -> None:
    """Accumulate detached scalar diagnostics without a per-batch CPU sync."""

    for key, value in stats.items():
        scalar = value.detach().float()
        if scalar.numel() != 1:
            raise ValueError(f"diagnostic {key!r} must be scalar")
        if key in sums:
            sums[key] = sums[key] + scalar
        else:
            sums[key] = scalar.clone()
        observations[key] = observations.get(key, 0) + 1


def _finalize_diagnostics(
    sums: Mapping[str, torch.Tensor],
    observations: Mapping[str, int],
) -> Dict[str, Dict[str, float]]:
    """Separate batch means from quantities that represent epoch totals."""

    epoch_sum_keys = {
        "rp_selected",
        "td_active_edges",
        "td_weight_sum",
        "td_active_edge_fusion_mel",
        "td_active_edge_fusion_coch",
        "td_active_edge_mel_coch",
        "td_teacher_fusion",
        "td_teacher_mel",
        "td_teacher_coch",
    }
    batch_mean: Dict[str, float] = {}
    epoch_sum: Dict[str, float] = {}
    for key, value in sums.items():
        total = float(value.detach().cpu())
        if key in epoch_sum_keys:
            epoch_sum[key] = total
        else:
            batch_mean[key] = total / max(1, observations[key])
    return {"batch_mean": batch_mean, "epoch_sum": epoch_sum}


def _detached_scalars(stats: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    return {
        key: float(value.detach().cpu())
        for key, value in stats.items()
    }


def _checkpoint_payload(
    epoch: int,
    model: nn.Module,
    proto: ProtoAlign,
    relia: ReliaPseudo,
    tri: TriDistill,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    generator: torch.Generator,
    best_epoch: int,
    best_metrics: Mapping[str, float],
    config: Mapping[str, Any],
    split_hash: str,
) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "protoalign": proto.state_dict(),
        "reliapseudo": relia.state_dict(),
        "tridistill": tri.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "train_generator_state": generator.get_state(),
        "rng_state": _rng_state(),
        "best_epoch": best_epoch,
        "best_metrics": dict(best_metrics),
        "config": dict(config),
        "split_hash": split_hash,
    }


def train(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    summary_path = run_dir / "summary.json"
    if summary_path.is_file() and not args.resume:
        raise RuntimeError(f"completed run already exists: {summary_path}")
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise RuntimeError(f"run directory is non-empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    memory_fraction: Optional[float] = None
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.gpu_memory_limit_gb > 0:
            total = torch.cuda.get_device_properties(device).total_memory
            memory_fraction = min(
                1.0, args.gpu_memory_limit_gb * (1024**3) / float(total)
            )
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)

    set_seed(args.seed)
    dataset = MusicEmotionDataset(
        args.dataset,
        label_mode=args.label_mode,
        data_root=args.data_root,
        cache_in_memory=True,
    )
    train_indices, val_indices = stratified_split(
        dataset.labels,
        train_ratio=1.0 - args.val_ratio,
        seed=args.split_seed,
    )
    train_array = np.asarray(train_indices, dtype=np.int64)
    val_array = np.asarray(val_indices, dtype=np.int64)
    split_hash = hashlib.sha256(
        train_array.tobytes() + b"|" + val_array.tobytes()
    ).hexdigest()
    np.savez_compressed(
        run_dir / "split_indices.npz", train=train_array, val=val_array
    )
    _json_write(
        run_dir / "split_manifest.json",
        {
            "dataset": args.dataset,
            "label_mode": args.label_mode,
            "split_seed": args.split_seed,
            "train_size": len(train_indices),
            "val_size": len(val_indices),
            "split_hash": split_hash,
        },
    )

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = _make_loader(
        train_subset, args.batch_size, args.workers, True, train_generator
    )
    train_eval_loader = _make_loader(
        train_subset, args.eval_batch_size, args.workers, False
    )
    val_loader = _make_loader(
        val_subset, args.eval_batch_size, args.workers, False
    )

    mel_dim = dataset.mel_nodes * dataset.feat_dim
    coch_dim = dataset.coch_nodes * dataset.feat_dim
    model = MusicFusionModel(
        mel_dim=mel_dim,
        coch_dim=coch_dim,
        hidden_mel=args.mel_hidden_dim,
        hidden_coch=args.coch_hidden_dim,
        fusion_hidden=args.fusion_hidden_dim,
        num_classes=2,
    ).to(device)

    use_p, use_r, use_t = (bit == "1" for bit in args.modules)
    train_labels = dataset.labels[torch.as_tensor(train_indices)]
    class_counts = torch.bincount(train_labels, minlength=2).double()
    class_prior = (class_counts / class_counts.sum()).tolist()
    proto = ProtoAlign(
        num_classes=2,
        feature_dim=args.fusion_hidden_dim,
        momentum=args.proto_momentum,
        temperature=args.proto_temperature,
        enabled=use_p,
    ).to(device)
    relia = ReliaPseudo(
        num_samples=len(dataset),
        num_classes=2,
        class_prior=class_prior,
        total_epochs=args.epochs,
        ema_momentum=args.relia_momentum,
        safety_floor=args.relia_floor,
        min_coverage=args.relia_min_coverage,
        max_coverage=args.relia_max_coverage,
        warmup_epochs=args.warmup_epochs,
        enabled=use_r,
        training_size=len(train_indices),
        coverage_schedule=args.coverage_schedule,
        target_mode=args.relia_target_mode,
        soft_ema_weight=args.relia_soft_ema_weight,
    ).to(device)
    tri = TriDistill(
        temperature=args.distill_temperature,
        warmup_epochs=args.warmup_epochs,
        enabled=use_t,
    ).to(device)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    effective_batches = (
        min(len(train_loader), args.max_train_batches)
        if args.max_train_batches > 0
        else len(train_loader)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, effective_batches * args.epochs)
    )
    amp_enabled = args.precision == "amp" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    config = vars(args).copy()
    config.update(
        {
            "class_prior": class_prior,
            "train_size": len(train_indices),
            "val_size": len(val_indices),
            "amp_enabled": amp_enabled,
        }
    )
    protocol_config = {
        key: value
        for key, value in config.items()
        if key
        not in {
            "seed",
            "run_dir",
            "resume",
            "device",
            "gpu_memory_limit_gb",
        }
    }
    config_hash = _hash_json(protocol_config)
    repo_dir = Path(__file__).resolve().parent
    code_hash = _code_hash(
        [
            Path(__file__).resolve(),
            repo_dir / "lib" / "models.py",
            repo_dir / "lib" / "data.py",
            repo_dir / "lib" / "redesign_modules.py",
        ]
    )
    _json_write(
        run_dir / "config.json",
        {
            "config": config,
            "protocol_config": protocol_config,
            "config_hash": config_hash,
            "code_hash": code_hash,
            "split_hash": split_hash,
            "gpu_memory_fraction": memory_fraction,
        },
    )

    start_epoch = 0
    best_epoch = -1
    best_metrics: Dict[str, float] = {}
    last_path = run_dir / "last.pt"
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device)
        if checkpoint["split_hash"] != split_hash:
            raise RuntimeError("resume checkpoint split hash does not match")
        model.load_state_dict(checkpoint["model"])
        proto.load_state_dict(checkpoint["protoalign"])
        relia.load_state_dict(checkpoint["reliapseudo"])
        tri.load_state_dict(checkpoint["tridistill"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        train_generator.set_state(checkpoint["train_generator_state"])
        _restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_metrics = dict(checkpoint["best_metrics"])

    history_path = run_dir / "history.jsonl"
    if start_epoch == 0 and history_path.exists():
        history_path.unlink()
    total_started = time.perf_counter()
    total_successful_steps = 0

    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.perf_counter()
        model.train()
        if use_r and epoch == args.warmup_epochs:
            initialize_relia_state(relia, model, train_eval_loader, device)
        if use_r and epoch >= args.warmup_epochs:
            relia.begin_epoch(epoch)

        sums: Dict[str, float] = {
            "total": 0.0,
            "sup": 0.0,
            "pa": 0.0,
            "rp": 0.0,
            "td": 0.0,
            "coverage": 0.0,
        }
        seen_batches = 0
        successful_steps = 0
        failed_steps = 0
        diagnostic_sums: Dict[str, torch.Tensor] = {}
        diagnostic_observations: Dict[str, int] = {}
        proto_update_count_start = (
            proto.update_count.detach().clone() if use_p else None
        )

        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches > 0 and batch_index >= args.max_train_batches:
                break
            mel = batch["mel"].to(device, non_blocking=True)
            coch = batch["coch"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            indices = batch["index"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(mel, coch)

            with torch.autocast(device_type=device.type, enabled=False):
                mel_logits = outputs["mel_logits"].float()
                coch_logits = outputs["coch_logits"].float()
                fusion_logits = outputs["fusion_logits"].float()
                supervised = F.cross_entropy(fusion_logits, labels) + 0.5 * (
                    F.cross_entropy(mel_logits, labels)
                    + F.cross_entropy(coch_logits, labels)
                )
                pa_stats: Dict[str, torch.Tensor] = {}
                if use_p:
                    pa_loss, pa_stats = proto(outputs, labels)
                else:
                    pa_loss = supervised.new_zeros(())

                td_stats: Dict[str, torch.Tensor] = {}
                if use_t:
                    td_core, td_stats = tri(outputs, labels, epoch)
                    if epoch >= args.warmup_epochs:
                        ramp = min(
                            1.0,
                            (epoch - args.warmup_epochs + 1)
                            / max(1, args.distill_ramp_epochs),
                        )
                    else:
                        ramp = 0.0
                    td_loss = td_core * ramp
                else:
                    td_loss = supervised.new_zeros(())

            rp_loss = supervised.new_zeros(())
            rp_stats: Dict[str, torch.Tensor] = {}
            current_coch_probs: Optional[torch.Tensor] = None
            if use_r and epoch >= args.warmup_epochs:
                current_coch_probs = F.softmax(coch_logits.detach(), dim=1)
                first_fusion_probs = F.softmax(fusion_logits.detach(), dim=1)
                with freeze_bn_running_stats(model):
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=amp_enabled,
                    ):
                        second_outputs = model(mel, coch)
                with torch.autocast(device_type=device.type, enabled=False):
                    rp_loss, rp_stats = relia(
                        second_outputs["fusion_logits"].float(),
                        labels,
                        indices,
                        current_coch_probs,
                        epoch=epoch,
                        first_fusion_probs=first_fusion_probs,
                    )

            total_loss = (
                supervised
                + args.lambda_p * pa_loss
                + args.lambda_r * rp_loss
                + args.lambda_t * td_loss
            )
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError(
                    f"non-finite loss at epoch={epoch + 1}, batch={batch_index + 1}"
                )

            if amp_enabled:
                previous_scale = float(scaler.get_scale())
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                gradients_finite = _finite_gradients(model)
                scaler.step(optimizer)
                scaler.update()
                step_succeeded = (
                    gradients_finite and float(scaler.get_scale()) >= previous_scale
                )
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                step_succeeded = _finite_gradients(model)
                if step_succeeded:
                    optimizer.step()

            if use_r and epoch >= args.warmup_epochs:
                relia.commit(
                    indices,
                    current_coch_probs,
                    step_succeeded=step_succeeded,
                )
            if step_succeeded:
                scheduler.step()
                if use_p:
                    proto.update(outputs["fusion_feat"], labels, True)
                successful_steps += 1
                total_successful_steps += 1
            else:
                failed_steps += 1
                optimizer.zero_grad(set_to_none=True)

            sums["total"] += float(total_loss.detach())
            sums["sup"] += float(supervised.detach())
            sums["pa"] += float(pa_loss.detach())
            sums["rp"] += float(rp_loss.detach())
            sums["td"] += float(td_loss.detach())
            sums["coverage"] += float(
                rp_stats.get("rp_coverage", supervised.new_zeros(()))
            )
            _accumulate_diagnostics(
                diagnostic_sums, diagnostic_observations, pa_stats
            )
            _accumulate_diagnostics(
                diagnostic_sums, diagnostic_observations, rp_stats
            )
            _accumulate_diagnostics(
                diagnostic_sums, diagnostic_observations, td_stats
            )
            seen_batches += 1

        mask_stats: Dict[str, float] = {}
        if use_r and epoch >= args.warmup_epochs:
            raw_mask_stats = relia.finalize_epoch(epoch)
            mask_stats = _detached_scalars(raw_mask_stats)
        diagnostics: Dict[str, Any] = _finalize_diagnostics(
            diagnostic_sums, diagnostic_observations
        )
        if use_p:
            diagnostics["protoalign_state"] = _detached_scalars(
                proto.diagnostics(proto_update_count_start)
            )
        if mask_stats:
            diagnostics["reliapseudo_mask"] = dict(mask_stats)
        should_evaluate = (
            args.checkpoint_rule == "best_acc"
            or epoch == args.epochs - 1
            or (
                args.eval_every > 0
                and (epoch + 1) % args.eval_every == 0
            )
        )
        validation = (
            evaluate(model, val_loader, device) if should_evaluate else None
        )
        epoch_seconds = time.perf_counter() - epoch_started
        peak_allocated = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved(device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        )
        record = {
            "epoch": epoch + 1,
            "train": {
                key: value / max(1, seen_batches) for key, value in sums.items()
            },
            "validation": validation,
            "successful_steps": successful_steps,
            "failed_steps": failed_steps,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_seconds,
            "samples_per_second": len(train_subset) / max(epoch_seconds, 1e-9),
            "peak_gpu_allocated_gib": peak_allocated,
            "peak_gpu_reserved_gib": peak_reserved,
            "diagnostics": diagnostics,
            **mask_stats,
        }
        _jsonl_append(history_path, record)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)

        should_save_best = validation is not None and (
            (
                args.checkpoint_rule == "best_acc"
                and (best_epoch < 0 or validation["acc"] > best_metrics["acc"])
            )
            or (
                args.checkpoint_rule == "final"
                and epoch == args.epochs - 1
            )
        )
        if should_save_best:
            best_epoch = epoch + 1
            best_metrics = dict(validation)
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "protoalign": proto.state_dict(),
                    "reliapseudo": relia.state_dict(),
                    "tridistill": tri.state_dict(),
                    "best_metrics": best_metrics,
                    "config": config,
                    "split_hash": split_hash,
                    "config_hash": config_hash,
                    "code_hash": code_hash,
                },
                run_dir / "best.pt",
            )

        torch.save(
            _checkpoint_payload(
                epoch,
                model,
                proto,
                relia,
                tri,
                optimizer,
                scheduler,
                scaler,
                train_generator,
                best_epoch,
                best_metrics,
                config,
                split_hash,
            ),
            last_path,
        )

    total_seconds = time.perf_counter() - total_started
    summary = {
        "schema_version": 1,
        "completed": True,
        "status": "completed",
        "dataset": args.dataset,
        "label_mode": args.label_mode,
        "modules": args.modules,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "split_hash": split_hash,
        "config_hash": config_hash,
        "code_hash": code_hash,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "config": config,
        "total_seconds": total_seconds,
        "successful_steps": total_successful_steps,
        "gpu_memory_limit_gb": args.gpu_memory_limit_gb,
        "gpu_memory_fraction": memory_fraction,
        "peak_gpu_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "peak_gpu_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "dataset_stats": get_dataset_stats(dataset),
    }
    _json_write(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the complete MusicFusion model.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--dataset",
        choices=("memo", "pmemo", "1000songs", "songs1000", "deam"),
        required=True,
    )
    parser.add_argument(
        "--label-mode",
        choices=("a", "v"),
        required=True,
        help="Emotion dimension: a for arousal or v for valence.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Directory containing the prepared dataset folders.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="New output directory for checkpoints, metrics, and logs.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from run-dir/last.pt.",
    )
    parser.set_defaults(
        modules="111",
        epochs=40,
        batch_size=512,
        eval_batch_size=1024,
        lr=1e-3,
        weight_decay=1e-4,
        optimizer="adamw",
        split_seed=42,
        val_ratio=0.3,
        precision="amp",
        gpu_memory_limit_gb=0.0,
        mel_hidden_dim=512,
        coch_hidden_dim=512,
        fusion_hidden_dim=256,
        lambda_p=0.05,
        lambda_r=0.05,
        lambda_t=0.05,
        proto_momentum=0.9,
        proto_temperature=0.5,
        relia_momentum=0.9,
        relia_floor=0.6,
        relia_min_coverage=0.1,
        relia_max_coverage=0.6,
        coverage_schedule="linear",
        relia_target_mode="soft",
        relia_soft_ema_weight=0.5,
        warmup_epochs=5,
        distill_temperature=2.0,
        distill_ramp_epochs=5,
        max_grad_norm=5.0,
        max_train_batches=0,
        checkpoint_rule="best_acc",
        eval_every=1,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 0:
        raise SystemExit("--workers must be non-negative")
    summary = train(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
