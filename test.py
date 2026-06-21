from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    r = str(root)
    if r not in sys.path:
        sys.path.insert(0, r)

    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="dataset")
    p.add_argument("--weights", type=str, default="results/checkpoints/best_weights.pt")
    p.add_argument("--out", type=str, default="Out")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    from main import TrainConfig

    cfg = TrainConfig()
    cfg.mode = "test"
    cfg.data_root = Path(args.data_root)
    cfg.test_weights = Path(args.weights)
    cfg.test_output_dir = Path(args.out)
    if not cfg.test_output_dir.is_absolute():
        cfg.test_output_dir = root / cfg.test_output_dir
    cfg.device = args.device
    cfg.num_workers = args.num_workers

    import train as train_mod

    train_mod.run_testing(cfg)


if __name__ == "__main__":
    main()
