"""
Shared training infrastructure for Phases 8-12.

Phases 8, 9 and 10 are the same loop with different parts of the model frozen
and different learning rates (guide 11.1, 12.1, 13.1), so seeding, dataloader
construction, freezing, AMP, gradient clipping, early stopping and checkpoint
I/O all live here and are written once.

Nothing in this module is task-specific - it knows about the model and the
datasets, not about NER or severity.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import Collator, HealthNERDataset, SeverityDataset
from featurizer import FastTextFeaturizer

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
DEFAULT_CACHE = ROOT / "feature_cache.npz"


# --------------------------------------------------------------- environment


def set_seed(seed):
    """Seed every RNG the training path touches (guide 15.4 runs three seeds)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_amp(device, enabled=True):
    """(autocast_factory, scaler, actually_enabled) - mixed precision on GPU only.

    Wrapped because the torch.amp / torch.cuda.amp spelling moved between
    versions and Kaggle's preinstalled torch is not pinned by this project.
    """
    use = bool(enabled) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use)

        def autocast():
            return torch.amp.autocast("cuda", enabled=use)
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use)

        def autocast():
            return torch.cuda.amp.autocast(enabled=use)

    return autocast, scaler, use


# ---------------------------------------------------------------- data


def load_featurizer(cache_path=DEFAULT_CACHE):
    """FastTextFeaturizer backed by the precomputed 600D cache.

    Falls back to loading the two live FastText models only if the cache is
    missing, because that needs ~8GB of RAM and is not viable on a Kaggle GPU
    notebook - see scripts/build_feature_cache.py.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        from build_feature_cache import load_feature_cache

        cache = load_feature_cache(cache_path)
        return FastTextFeaturizer(precomputed_cache=cache)

    raise SystemExit(
        f"No feature cache at {cache_path}.\n"
        "Build it once with:  python scripts/build_feature_cache.py\n"
        "(or pass --cache pointing at the copy in your Kaggle dataset)."
    )


def _loader(dataset, featurizer, batch_size, shuffle, limit=None, num_workers=0):
    """DataLoader with the shared dynamic-padding collator.

    num_workers defaults to 0 on purpose: the collator holds the ~105MB feature
    cache, and every worker process would get its own pickled copy. The lookup
    itself is a dict access plus a numpy stack, so workers buy nothing here.
    """
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Collator(featurizer),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def ner_loader(featurizer, split, batch_size, shuffle=None, limit=None):
    """HealthNER split -> DataLoader. Shuffles the train split by default."""
    if shuffle is None:
        shuffle = split == "train"
    return _loader(HealthNERDataset(split), featurizer, batch_size, shuffle, limit)


def severity_loader(featurizer, split, batch_size, shuffle=None, limit=None):
    """Severity split -> DataLoader. Shuffles the train split by default."""
    if shuffle is None:
        shuffle = split == "train"
    return _loader(SeverityDataset(split), featurizer, batch_size, shuffle, limit)


# ---------------------------------------------------------------- parameters


def set_requires_grad(module, flag):
    """Freeze (False) or unfreeze (True) every parameter of a submodule."""
    for p in module.parameters():
        p.requires_grad_(flag)
    return module


def trainable_parameters(module):
    return [p for p in module.parameters() if p.requires_grad]


def count_parameters(module, only_trainable=False):
    params = trainable_parameters(module) if only_trainable else list(module.parameters())
    return sum(p.numel() for p in params)


def describe_trainable(model):
    """One dict per top-level block: how many parameters, how many trainable.

    Printed at the start of each phase so a mistaken freeze shows up in the log
    immediately instead of as a mysteriously flat loss curve.
    """
    rows = {}
    for name, child in model.named_children():
        total = count_parameters(child)
        if total:
            rows[name] = {
                "total": total,
                "trainable": count_parameters(child, only_trainable=True),
            }
    return rows


# ---------------------------------------------------------------- training


class EarlyStopping:
    """Patience on a maximised metric (entity F1, macro F1). Guide 11.3."""

    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.best_epoch = None
        self.bad_epochs = 0

    def step(self, value, epoch):
        """Returns True when `value` is a new best."""
        if self.best is None or value > self.best + self.min_delta:
            self.best = value
            self.best_epoch = epoch
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self):
        return self.bad_epochs >= self.patience


class AverageMeter:
    """Running mean weighted by batch size."""

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.total / self.count if self.count else 0.0


def clip_and_step(scaler, optimizer, params, max_norm=1.0):
    """Unscale, clip to a global norm, step, update (guide 11.3: clip at 1.0)."""
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(params, max_norm)
    scaler.step(optimizer)
    scaler.update()


# ---------------------------------------------------------------- checkpoints


def save_checkpoint(path, model, optimizer=None, epoch=None, metrics=None,
                    config=None, seed=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
            "seed": seed,
        },
        path,
    )
    return path


def load_checkpoint(path, model, optimizer=None, device=None, strict=True):
    """Restore weights (and optionally optimizer state) from a checkpoint.

    strict=False is what Phase 9 needs: best_ner.pt was saved before the
    severity head had ever been trained, so its severity-head weights are the
    random initialisation and may legitimately differ.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Missing checkpoint {path} - run the previous phase first.")

    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


# ---------------------------------------------------------------- reporting


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def human_time(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0

    @property
    def now(self):
        return time.time() - self.t0
