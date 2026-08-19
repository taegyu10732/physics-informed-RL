"""Evaluate a trained gas-source navigation agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC

from rlm.mower_env import MowerEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to agent.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--map-dir", type=Path, default=Path("maps"))
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = MowerEnv(map_dir=args.map_dir, eval=True, render_mode="human" if args.render else None)
    model = SAC.load(args.model, env=env)
    successes = 0
    episode_rewards = []
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset()
            done = False
            total_reward = 0.0
            info = {}
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                if args.render:
                    env.render()
            successes += int(info.get("success", False))
            episode_rewards.append(total_reward)
            print(
                f"episode={episode + 1} success={info.get('success', False)} "
                f"steps={env.elapsed_steps} reward={total_reward:.2f}"
            )
    finally:
        env.close()

    mean_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"success_rate={successes / args.episodes:.1%} mean_reward={mean_reward:.2f}")


if __name__ == "__main__":
    main()
