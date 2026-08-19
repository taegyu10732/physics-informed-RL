"""Train the SAC gas-source navigation agent."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from rlm.architectures import GasMapFeaturesExtractor
from rlm.mower_env import MowerEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--features", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--map-dir", type=Path, default=Path("maps"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or Path("experiments") / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2),
        encoding="utf-8",
    )

    env = Monitor(MowerEnv(map_dir=args.map_dir, seed=args.seed), str(output / "monitor.csv"))
    policy_kwargs = {
        "features_extractor_class": GasMapFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": args.features},
        "net_arch": {"pi": [args.features] * 2, "qf": [args.features] * 2},
    }
    if args.resume:
        model = SAC.load(args.resume, env=env)
    else:
        model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
            verbose=1,
        )
    callback = CheckpointCallback(
        save_freq=args.checkpoint_every,
        save_path=str(output / "checkpoints"),
        name_prefix="gas_agent",
    )
    try:
        model.learn(total_timesteps=args.steps, callback=callback)
        model.save(output / "agent")
    finally:
        env.close()


if __name__ == "__main__":
    main()
