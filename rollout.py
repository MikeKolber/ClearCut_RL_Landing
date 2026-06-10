import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC

import config
from landing_env import RocketLandingEnv
from rocket_sim import Rocket1DSim

ALGOS = {"ppo": PPO, "sac": SAC}  # pick which trained model to load


def freefall_trajectory(max_steps=5000):
    """Engine-off (throttle 0) trajectory from the same start, for a grey baseline."""
    sim = Rocket1DSim()
    sim.reset()
    ts, hs, vs = [0.0], [sim.h], [sim.v]
    t = 0.0
    for _ in range(max_steps):
        terminated = sim.step(0.0)
        t += sim.dt
        ts.append(t); hs.append(sim.h); vs.append(sim.v)
        if terminated:
            break
    return np.array(ts), np.array(hs), np.array(vs)


def run_episode(model, env, seed=123, deterministic=True):
    """Run one episode with a model; return its raw trajectory and an outcome summary."""
    obs, info = env.reset(seed=seed)
    rows = []  # t, h, v, throttle, m
    total_reward = 0.0
    terminated = truncated = False
    t = 0.0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        h_before, v_before, m_before = env.sim.h, env.sim.v, env.sim.m
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append((t, h_before, v_before, env.prev_throttle, m_before))
        total_reward += reward
        t += env.sim.dt

    impact = info.get("impact_speed", float("nan"))
    outcome = ("LANDED" if (terminated and impact <= config.SAFE_LANDING_SPEED)
               else ("crashed" if terminated else "timed out"))
    summary = dict(total_reward=total_reward, impact=impact, outcome=outcome, steps=len(rows))
    return np.array(rows), summary


def plot_trajectory(data, summary, outpath, title_prefix="Trained policy rollout"):
    t_a, h_a, v_a, thr_a, m_a = data.T
    ff_t, ff_h, ff_v = freefall_trajectory()  # grey dotted "engine off" baseline
    full_mass = config.DRY_MASS + config.FUEL_MASS

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 11))

    axes[0].plot(t_a, h_a, label="policy")
    axes[0].plot(ff_t, ff_h, ls=":", color="grey", label="free fall")
    axes[0].set_ylabel("altitude [m]"); axes[0].legend(loc="upper right")

    axes[1].plot(t_a, v_a, label="policy")
    axes[1].plot(ff_t, ff_v, ls=":", color="grey", label="free fall")
    axes[1].set_ylabel("velocity [m/s]"); axes[1].legend(loc="lower right")

    axes[2].plot(t_a, thr_a)
    axes[2].axhline(0.0, ls=":", color="grey")  # free fall = engine off
    axes[2].set_ylabel("throttle"); axes[2].set_ylim(-0.05, 1.05)

    axes[3].plot(t_a, m_a)
    axes[3].axhline(m_a[0], ls=":", color="grey")  # free fall = no fuel burned
    axes[3].set_ylabel("mass [kg]"); axes[3].set_xlabel("time [s]")
    axes[3].set_ylim(config.DRY_MASS - 5, full_mass + 5)  # fixed scale: tiny burns look tiny

    for ax in axes:
        ax.grid(True)
    axes[0].set_title(f"{title_prefix} - {summary['outcome']} "
                      f"(impact {summary['impact']:.2f} m/s, reward {summary['total_reward']:+.1f})")
    fig.tight_layout()
    fig.savefig(outpath, dpi=110)
    plt.close(fig)


def main():
    # Usage: python rollout.py [ppo|sac]   (defaults to ppo)
    prefix = sys.argv[1] if len(sys.argv) > 1 else "ppo"
    model_path = os.path.join("output", "models", f"{prefix}_lander")
    model = ALGOS[prefix].load(model_path)
    env = RocketLandingEnv()
    data, summary = run_episode(model, env)
    print(f"[{prefix}] outcome={summary['outcome']}  impact={summary['impact']:.2f} m/s  "
          f"total_reward={summary['total_reward']:+.2f}  steps={summary['steps']}")
    out = os.path.join("output", "plots", f"{prefix}_rollout.png")
    plot_trajectory(data, summary, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
