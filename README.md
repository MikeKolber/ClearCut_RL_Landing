# Landing_RL — 1D Rocket Landing with Reinforcement Learning

A reinforcement-learning agent learns to land a rocket softly in a hand-written
1D physics simulator. The agent controls one thing — the engine throttle — and
must bring the rocket from apogee down to the ground at a gentle speed without
running out of fuel. Trained with PPO (and optionally SAC) via Stable-Baselines3.

The project serves as RL infrastructure that will later be applied to a full
3D physics engine. The 1D sim is simple enough to iterate on reward design and
understand agent behavior before scaling up.

## Setup

```bash
conda activate clearcut
pip install -r requirements.txt
```

## Project structure

```
Landing_RL/
  config.py              all tunable constants (physics, rewards, training)
  rocket_sim.py          1D physics engine (point mass, RK4 integration)
  landing_env.py         Gymnasium wrapper: observation, action, reward
  train_common.py        shared training logic + ASTOS comparison plot
  train_ppo.py           train with PPO:  python train_ppo.py
  train_sac.py           train with SAC:  python train_sac.py
  rollout.py             evaluate a trained model:  python rollout.py [ppo|sac]
  Full_data_4Michael.xlsx  ASTOS reference trajectory for comparison
  output/
    models/              saved .zip model files (best + last, per algorithm)
    plots/               best landing, training curve, RL-vs-ASTOS PNGs
    progress/            periodic rollout snapshots during training
```

## How to run

**Train** (PPO, ~2-3 min for 1M steps):
```bash
python train_ppo.py
```

**Evaluate** the best saved model:
```bash
python rollout.py ppo
```

**Compare RL vs ASTOS** — this happens automatically after training, but the
overlay plot is saved to `output/plots/ppo_vs_astos.png`.

All tunable values live in `config.py`. Change a value there, retrain, and
check the results.

---

## Simulator

`rocket_sim.py` implements a 1D point-mass rocket under:
- **Gravity** (constant, 9.81 m/s^2)
- **Aerodynamic drag** (constant air density, lumped Cd*A)
- **Thrust** (proportional to throttle, with fuel depletion via Isp)

Integration uses **RK4** (Runge-Kutta 4th order). The state vector is
`(altitude, velocity, mass)`. The rocket starts at apogee (v=0) and the
episode ends when it reaches the ground (h <= 0).

Vehicle parameters are based on a real sounding rocket spec sheet: dry mass
798.1 kg, 168.1 kg landing fuel reserve, 13 kN max thrust, Isp 208.1 s.

## Environment (MDP)

`landing_env.py` wraps the simulator as a Gymnasium environment.

**Observation** (3 values, normalized):
- `altitude / init_altitude` — how high, scaled to ~1 at start
- `velocity / V_SCALE` — how fast, scaled by ~terminal velocity
- `fuel_fraction` — fuel remaining as a fraction of initial fuel

**Action** (1 value, continuous):
- Throttle in `[MIN_THROTTLE, 1.0]` — the engine cannot go below 60% when on

**Constraints:**
- The first timestep is forced to `INIT_THROTTLE` (matching the ASTOS reference
  trajectory at apogee), then the agent takes over
- A **throttle rate limit** (`MAX_THROTTLE_RATE`) caps how fast the throttle can
  change per step, producing physically realistic smooth thrust profiles

---

## Reward design

The reward function was developed iteratively. Each component exists to solve
a specific problem that appeared during training. The total reward each step is:

```
reward = reward_shaping + reward_fuel + reward_time + reward_terminal
```

### Terminal reward (paid once when the episode ends)

**Landing bonus** — `LANDING_BONUS * exp(-impact_speed / SOFTNESS_TAU)`

Any touchdown is rewarded. The bonus decays exponentially with impact speed
but is **never negative** — there is no "crash cliff." This was a deliberate
choice: early designs had a hard crash penalty (e.g. -200 for impact > 2 m/s),
but the agent became so afraid of crashing that it preferred to hover
indefinitely and time out. By making every landing positive (just less
positive if hard), reaching the ground always beats floating, and the agent
commits to landing rather than avoiding it.

**Timeout penalty** — `-100` if still airborne when `MAX_STEPS` is reached.

**Out-of-fuel penalty** — `-100` if fuel runs out before reaching the ground.

These two are the only true failure modes. They're set equal so the agent
treats both as equally bad — it shouldn't waste all its fuel, but it also
shouldn't hoard fuel and time out.

### Per-step shaping (paid every step)

**Potential-based shaping** — `phi(s') - phi(s)` where
`phi(s) = -(W_alt * |altitude| + W_vel * |velocity|)`

This rewards the agent for getting closer to the ground and slower, without
changing which policy is theoretically optimal (potential-based shaping is
provably neutral on the optimal policy). Without it, the agent has no
guidance until the very end of the episode and learns extremely slowly — the
terminal bonus at step 300+ is invisible through random exploration.

The velocity weight (`W_vel = 1.0`) is much larger than the altitude weight
(`W_alt = 0.05`) because controlling speed matters more than altitude for a
safe landing.

**Fuel penalty** — `-FUEL_PENALTY * kg_burned_this_step`

A per-step cost for burning propellant. This encourages fuel efficiency. Set
to 0 when fuel efficiency is not a priority (e.g. when tuning other rewards).
When active, it pushes the agent toward fuel-optimal trajectories like the
suicide burn (coast, then brake hard at the last moment).

**Time penalty (living cost)** — `-TIME_PENALTY` every step the rocket is
still airborne

This was the key fix for the **hovering problem**. The shaping potential peaks
at `h=0, v=0`, but the potential surface is broad — hovering a few meters above
the ground scores almost as well as actually landing. Without a living cost,
the agent parks just above the pad, collecting near-maximum shaping reward
forever while never touching down. The time penalty makes every airborne step
bleed reward, so "land now" strictly beats "hover then time out."

### Design principles and lessons learned

1. **No crash cliff.** A hard boundary between "landed" and "crashed" creates
   a discontinuity the agent exploits by avoiding the ground entirely. The
   exponential decay is smooth and always positive.

2. **Failure = not landing.** The only negative terminal rewards are timeout
   and out-of-fuel. This frames the problem as "land somehow, then land
   better" rather than "avoid crashing."

3. **Potential-based shaping is safe.** Unlike arbitrary per-step rewards,
   potential-based shaping provably doesn't change the optimal policy — it
   only speeds up learning. Other shaping attempts (e.g. a proximity reward
   for being near the ground) created new exploits where the agent camped
   at low altitude without landing.

4. **Living cost breaks hover equilibria.** The time penalty is small (0.1
   per step) but accumulates. Over 400 hovering steps that's -40, which is
   enough to make landing strictly better.

5. **Gamma matters.** The discount factor must be high enough that the agent
   can "see" the terminal bonus through hundreds of steps. At gamma=0.99
   (horizon ~100 steps), the landing bonus is invisible for long trajectories.
   At gamma=0.999 (horizon ~1000 steps), it works. The episode is 600 steps
   max, so gamma=0.999 or higher covers the full episode.

---

## Training details

**PPO** (Proximal Policy Optimization) — the primary algorithm. On-policy,
stable, works well with the cheap simulator. Uses linear learning rate decay
(3e-4 to 0) and an entropy bonus to prevent premature convergence. Trained
for 1M steps.

**SAC** (Soft Actor-Critic) — tested as an alternative. Off-policy and more
sample-efficient, but showed convergence instability on this problem: the
policy found the landing solution around 88k steps, then drifted out of it
by 120k. PPO's trust region (clipped objective) keeps it stable once it
converges. SAC is better suited for expensive simulators where sample
efficiency matters.

**Save-best checkpointing** — during training, the policy is periodically
evaluated (deterministic, no exploration noise). The best-performing
checkpoint is saved separately from the final model, protecting against
late-training degradation.

## ASTOS comparison

The project includes an ASTOS reference trajectory (`Full_data_4Michael.xlsx`)
for the same vehicle. After each training run, an overlay plot is automatically
generated comparing the RL trajectory against the ASTOS solution across all
four channels (altitude, velocity, thrust, mass). The ASTOS trajectory covers
the full flight (liftoff to landing); the RL trajectory is time-shifted to
start at the ASTOS apogee for direct comparison.
