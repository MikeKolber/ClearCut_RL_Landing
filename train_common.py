"""Shared training machinery used by both train_ppo.py and train_sac.py.

Everything here is algorithm-agnostic. Each algorithm passes a `prefix`
("ppo" / "sac") so all output files stay separate. All generated files
go into the output/ folder:
    output/models/{prefix}_lander.zip          best model
    output/models/{prefix}_lander_last.zip     final model
    output/plots/{prefix}_best_landing.png     rollout of the best model
    output/plots/{prefix}_training_curve.png   learning curve
    output/plots/{prefix}_vs_astos.png         RL vs ASTOS overlay
    output/progress/{prefix}/                  periodic rollout snapshots
    output/{prefix}_monitor.monitor.csv        per-episode reward log
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import config
from landing_env import RocketLandingEnv
from rollout import run_episode, plot_trajectory, ALGOS

OUT = "output"
MODELS_DIR = os.path.join(OUT, "models")
PLOTS_DIR = os.path.join(OUT, "plots")
PROGRESS_DIR = os.path.join(OUT, "progress")
ASTOS_FILE = "Full_data_4Michael.xlsx"


def linear_schedule(initial_value):
    """Learning-rate schedule: linearly decay from initial_value to 0 over training.
    SB3 calls this with progress_remaining, which runs 1.0 -> 0.0 across the run."""
    def schedule(progress_remaining):
        return progress_remaining * initial_value
    return schedule


class EvalSaveBestCallback(BaseCallback):
    """Every EVAL_FREQ steps: evaluate the (deterministic) policy, keep the best
    model so far, refresh its 'best landing' plot, and periodically save a progress
    plot. All outputs are prefixed so PPO and SAC never overwrite each other."""

    def __init__(self, prefix, verbose=0):
        super().__init__(verbose)
        self.prefix = prefix
        self.model_path = os.path.join(MODELS_DIR, f"{prefix}_lander")
        self.best_plot = os.path.join(PLOTS_DIR, f"{prefix}_best_landing.png")
        self.progress_dir = os.path.join(PROGRESS_DIR, prefix)
        self.eval_env = RocketLandingEnv()
        self.best_reward = float("-inf")

    def _on_training_start(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)
        os.makedirs(self.progress_dir, exist_ok=True)
        self._evaluate_and_save()

    def _on_step(self):
        if self.n_calls % config.EVAL_FREQ == 0:
            self._evaluate_and_save()
        return True

    def _evaluate_and_save(self):
        rewards = []
        first = None
        for i in range(config.EVAL_EPISODES):
            data, summary = run_episode(self.model, self.eval_env, seed=123 + i)
            rewards.append(summary["total_reward"])
            if first is None:
                first = (data, summary)
        mean_reward = sum(rewards) / len(rewards)
        data, summary = first

        marker = ""
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            self.model.save(self.model_path)
            plot_trajectory(data, summary, self.best_plot,
                            title_prefix=f"{self.prefix} best landing")
            marker = "  <- new best, saved"

        if self.n_calls % config.PLOT_FREQ == 0:
            os.makedirs(self.progress_dir, exist_ok=True)
            path = os.path.join(self.progress_dir, f"rollout_{self.num_timesteps:07d}.png")
            plot_trajectory(data, summary, path,
                            title_prefix=f"{self.prefix} step {self.num_timesteps}")

        if self.verbose:
            print(f"[{self.prefix} eval] step {self.num_timesteps}: {summary['outcome']} "
                  f"impact={summary['impact']:.2f} m/s reward={mean_reward:+.1f}{marker}")


def plot_learning_curve(prefix):
    df = pd.read_csv(os.path.join(OUT, f"{prefix}_monitor.monitor.csv"), skiprows=1)
    timesteps = df["l"].cumsum().to_numpy()
    reward = df["r"].to_numpy()
    rolling = pd.Series(reward).rolling(50, min_periods=1).mean().to_numpy()

    plt.figure(figsize=(9, 5))
    plt.plot(timesteps, reward, alpha=0.3, label="episode reward")
    plt.plot(timesteps, rolling, color="black", label="rolling mean (50 episodes)")
    plt.xlabel("training timesteps")
    plt.ylabel("episode reward")
    plt.title(f"{prefix.upper()} training curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{prefix}_training_curve.png")
    plt.savefig(out, dpi=120)
    print(f"saved {out}")


def generate_comparison(prefix, model=None):
    """Overlay the best RL trajectory on the full ASTOS reference trajectory."""
    if not os.path.exists(ASTOS_FILE):
        print(f"[compare] {ASTOS_FILE} not found, skipping overlay plot")
        return

    df = pd.read_excel(ASTOS_FILE, header=None)
    df.columns = ["time", "altitude", "velocity", "thrust", "mass"]
    df["altitude"] *= 1000
    df["velocity"] *= 1000
    df["thrust"]   *= 1000
    df["mass"]     *= 1000

    apogee_idx = int(df["altitude"].idxmax())

    if model is None:
        algo_cls = ALGOS[prefix]
        model = algo_cls.load(os.path.join(MODELS_DIR, f"{prefix}_lander"))

    env = RocketLandingEnv()
    rl_data, rl_summary = run_episode(model, env)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    t_apogee = df.loc[apogee_idx, "time"]
    a_t, a_h, a_v, a_thr, a_m = [df[c].values for c in df.columns]
    rl_t, rl_h, rl_v, rl_thr, rl_m = rl_data.T
    rl_t_shifted = rl_t + t_apogee
    rl_thrust_N = rl_thr * config.MAX_THRUST

    label = prefix.upper()
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 13))
    astos_kw = dict(color="C3", linewidth=1.5, linestyle=":", label="ASTOS (full flight)")
    rl_kw    = dict(color="C0", linewidth=2, label=f"{label} RL (landing)")

    axes[0].plot(a_t, a_h, **astos_kw)
    axes[0].plot(rl_t_shifted, rl_h, **rl_kw)
    axes[0].axvline(t_apogee, ls=":", color="grey", alpha=0.6, label="apogee")
    axes[0].set_ylabel("altitude [m]")
    axes[0].legend(loc="upper right")

    axes[1].plot(a_t, a_v, **astos_kw)
    axes[1].plot(rl_t_shifted, rl_v, **rl_kw)
    axes[1].axvline(t_apogee, ls=":", color="grey", alpha=0.6)
    axes[1].set_ylabel("velocity [m/s]")

    axes[2].plot(a_t, a_thr, **astos_kw)
    axes[2].plot(rl_t_shifted, rl_thrust_N, **rl_kw)
    axes[2].axvline(t_apogee, ls=":", color="grey", alpha=0.6)
    axes[2].set_ylabel("thrust [N]")

    axes[3].plot(a_t, a_m, **astos_kw)
    axes[3].plot(rl_t_shifted, rl_m, **rl_kw)
    axes[3].axvline(t_apogee, ls=":", color="grey", alpha=0.6)
    axes[3].set_ylabel("mass [kg]")
    axes[3].set_xlabel("time [s]")

    apogee_row = df.loc[apogee_idx]
    ic_text = (f"ASTOS at apogee (v=0):\n"
               f"  altitude = {apogee_row['altitude']:.1f} m\n"
               f"  velocity = {apogee_row['velocity']:.4f} m/s\n"
               f"  thrust   = {apogee_row['thrust']:.0f} N\n"
               f"  fuel     = {apogee_row['mass'] - config.DRY_MASS:.1f} kg")
    axes[3].text(0.02, 0.05, ic_text, transform=axes[3].transAxes,
                 fontsize=9, verticalalignment="bottom", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    for ax in axes:
        ax.grid(True, alpha=0.3)

    impact_str = f"{rl_summary['impact']:.2f}" if not np.isnan(rl_summary['impact']) else "n/a"
    axes[0].set_title(f"ASTOS vs {label} RL  "
                      f"({rl_summary['outcome']}, impact {impact_str} m/s)")

    fig.tight_layout()
    outpath = os.path.join(PLOTS_DIR, f"{prefix}_vs_astos.png")
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"saved {outpath}")


def run_training(prefix, build_model, total_timesteps):
    """Build env + model, train, save best (via callback) and final, plot the curve."""
    os.makedirs(OUT, exist_ok=True)
    env = Monitor(RocketLandingEnv(), filename=os.path.join(OUT, f"{prefix}_monitor"))
    model = build_model(env)
    callback = EvalSaveBestCallback(prefix, verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    last_path = os.path.join(MODELS_DIR, f"{prefix}_lander_last")
    model.save(last_path)
    print(f"saved final model to {last_path}.zip; "
          f"best model is {MODELS_DIR}/{prefix}_lander.zip (mean reward {callback.best_reward:+.1f})")
    plot_learning_curve(prefix)
    generate_comparison(prefix)
