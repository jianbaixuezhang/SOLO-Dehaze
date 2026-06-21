from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class LossWeights:
    phase: float = 1
    perceptual: float = 1
    contrast: float = 5
    color: float = 0.01


@dataclass
class TrainConfig:
    mode: str = "train"
    data_root: Path = field(default_factory=lambda: Path("dataset"))
    results_dir: Path = field(default_factory=lambda: Path("results"))

    num_epochs: int = 6000
    batch_size: int = 4
    num_workers: int = 4
    device: str = "cuda"

    patch_size: int = 128
    rotate_degrees: float = 15.0

    lr: float = 1e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 200
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.0

    loss_weights: LossWeights = field(default_factory=LossWeights)

    contrast_global_margin: float = 0.2
    contrast_global_ema_momentum: float = 0.99

    val_interval: int = 20
    val_batch_size: int = 1
    log_interval: int = 10

    dehaze_num_levels: int = 3
    dehaze_hidden_dim: int = 64
    dehaze_level_dims: List[int] = field(default_factory=lambda: [32, 48, 64])
    dehaze_level_heads: List[int] = field(default_factory=lambda: [2, 3, 4])
    dehaze_base_dim: int = 96
    dehaze_base_heads: int = 6

    resume: bool = True
    resume_path: Path = field(default_factory=lambda: Path("results/checkpoints/last_full.pt"))

    test_weights: Path = field(default_factory=lambda: Path("results/checkpoints/best_weights.pt"))
    test_output_dir: Path = field(default_factory=lambda: Path("Out"))

    seed: int = 42


def _parse_args() -> TrainConfig:
    def _parse_int_list(text: str) -> List[int]:
        vals = [.strip() for  in text.split(",")]
        vals = [ for  in vals if ]
        if not vals:
            raise ValueError("List argument cannot be empty.")
        return [int(v) for v in vals]

    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="train", choices=("train", "test"))
    p.add_argument("--data_root", type=str, default="dataset")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--resume_path", type=str, default=None)
    p.add_argument("--test_weights", type=str, default=None)
    p.add_argument("--test_out", type=str, default=None)
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument("--dehaze_num_levels", type=int, default=None)
    p.add_argument("--dehaze_level_dims", type=str, default=None)
    p.add_argument("--dehaze_level_heads", type=str, default=None)
    p.add_argument("--dehaze_base_dim", type=int, default=None)
    p.add_argument("--dehaze_base_heads", type=int, default=None)
    args, _ = p.parse_known_args()

    cfg = TrainConfig()
    cfg.mode = args.mode
    cfg.data_root = Path(args.data_root)
    cfg.results_dir = Path(args.results_dir)
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device is not None:
        cfg.device = args.device
    if args.no_resume:
        cfg.resume = False
    if args.resume_path is not None:
        cfg.resume_path = Path(args.resume_path)
    if args.test_weights is not None:
        cfg.test_weights = Path(args.test_weights)
    if args.test_out is not None:
        cfg.test_output_dir = Path(args.test_out)
    if args.patch_size is not None:
        cfg.patch_size = args.patch_size
    if args.dehaze_num_levels is not None:
        cfg.dehaze_num_levels = args.dehaze_num_levels
    if args.dehaze_level_dims is not None:
        cfg.dehaze_level_dims = _parse_int_list(args.dehaze_level_dims)
    if args.dehaze_level_heads is not None:
        cfg.dehaze_level_heads = _parse_int_list(args.dehaze_level_heads)
    if args.dehaze_base_dim is not None:
        cfg.dehaze_base_dim = args.dehaze_base_dim
    if args.dehaze_base_heads is not None:
        cfg.dehaze_base_heads = args.dehaze_base_heads

    if len(cfg.dehaze_level_dims) != cfg.dehaze_num_levels:
        raise ValueError(
            f"dehaze_level_dims length ({len(cfg.dehaze_level_dims)}) must equal dehaze_num_levels ({cfg.dehaze_num_levels})."
        )
    if len(cfg.dehaze_level_heads) != cfg.dehaze_num_levels:
        raise ValueError(
            f"dehaze_level_heads length ({len(cfg.dehaze_level_heads)}) must equal dehaze_num_levels ({cfg.dehaze_num_levels})."
        )
    return cfg


def main() -> None:
    root = Path(__file__).resolve().parent
    r = str(root)
    if r not in sys.path:
        sys.path.insert(0, r)

    cfg = _parse_args()
    import train as train_mod

    if cfg.mode == "train":
        train_mod.run_training(cfg)
    else:
        train_mod.run_testing(cfg)


if __name__ == "__main__":
    main()
