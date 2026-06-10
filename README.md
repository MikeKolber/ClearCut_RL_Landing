# Landing_RL — 1D Rocket Landing with PPO

A reinforcement-learning agent learns to land a rocket softly in a small,
hand-written 1D simulator. The agent (PPO, from Stable-Baselines3) controls one
thing — the engine throttle — and tries to bring the rocket from apogee down to
the ground at a gentle speed without running out of fuel.

## Setup

One time per machine, in the `clearcut` conda environment:

```bash
conda activate clearcut
pip install -r requirements.txt
```

Every terminal session:

```bash
conda activate clearcut
cd /Users/michaelkolber/Documents/ClearCut-Space/Landing_RL
```

## The files

| File | What it is |
|------|------------|
| `rocket_sim.py`  | The physics. A point-mass rocket under gravity, drag, and thrust; fuel depletes with throttle. Pure NumPy, no RL. |
| `landing_env.py` | The Gymnasium environment: wraps the sim, defines the observation, the throttle action, and the **reward**. Edit reward constants here. |
| `train_ppo.py`   | Trains a PPO agent and saves everything (model, curves, progress plots). |
| `rollout.py`     | Loads the trained agent and shows what it actually does. |

## How to run

**Train an agent** (from scratch, ~2 minutes):

```bash
python train_ppo.py
```

It prints a status line every 20,000 steps, e.g.
`[eval] step 60000: timed out impact=nan m/s reward=-23.2`, and saves outputs
when done.

**See what the trained agent does:**

```bash
python rollout.py
```

This flies one episode with the saved agent, prints the outcome
(`LANDED` / `crashed` / `timed out`, with impact speed), and saves a plot.
You can run it any number of times without retraining.

Mental model: **`train_ppo.py` writes a brain file (`ppo_lander.zip`);
`rollout.py` reads it.**

## What gets produced, and where to look

| Output | Tells you |
|--------|-----------|
| `ppo_training_curve.png` | Episode reward vs. training steps. Rising = learning, flat = stuck. **Caution: high reward does not prove a good landing** — it can be a reward hack. |
| `progress/rollout_*.png` | A flipbook: one rollout plot every 20k steps, named by step number. Open in order to watch behavior evolve. |
| `ppo_rollout.png` | What the **final** agent does: altitude / velocity / throttle / mass vs. time. **This is the ground truth — always check it.** |
| `ppo_lander.zip` | The saved trained policy (used by `rollout.py`). |
| `ppo_monitor.monitor.csv` | Raw per-episode reward log (source of the curve). |

To answer "what did it learn?", look at `ppo_rollout.png` and the printed outcome.

## Experimenting with the reward

The reward is built from three terms, all in `landing_env.py`:

- **Shaping** (`W_H`, `W_V`): a small per-step reward for getting closer and
  slower (potential-based, so it can't be farmed by stalling).
- **Fuel penalty** (`W_FUEL`): a small per-step cost for burning fuel.
- **Terminal**: `+LANDING_BONUS` for a soft touchdown
  (`|v| <= SAFE_LANDING_SPEED`); otherwise a crash penalty that scales with
  impact speed (`CRASH_SPEED_PENALTY`, capped at `CRASH_PENALTY_CAP`).

To try a change: edit a constant in `landing_env.py`, then
`python train_ppo.py` and `python rollout.py` to see the effect.
