import numpy as np
import gymnasium as gym
from gymnasium import spaces

import config
from rocket_sim import Rocket1DSim


class RocketLandingEnv(gym.Env):
    """Gymnasium environment wrapping the 1D rocket physics (the MDP layer).

    Observation: normalized [altitude, velocity, fuel_fraction].
    Action: throttle in [0, 1].
    Reward: per-step potential shaping + fuel penalty + a living cost (so
    hovering never pays), plus a terminal bonus/penalty at episode end. All
    reward constants live in config.py.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps=config.MAX_STEPS):
        super().__init__()
        self.sim = Rocket1DSim()
        self.max_steps = max_steps  # truncation horizon (time limit), in steps

        # Observations are normalized to ~unit scale; generous finite bounds.
        high = np.full(3, 10.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=config.MIN_THROTTLE, high=1.0, shape=(1,), dtype=np.float32)

        self.steps = 0
        self.prev_throttle = config.INIT_THROTTLE

    def _send_agent_state(self):
        h = self.sim.h / self.sim.init_altitude
        v = self.sim.v / config.V_SCALE
        fuel_fraction = (self.sim.m - self.sim.dry_mass) / self.sim.fuel_mass
        return np.array([h, v, fuel_fraction], dtype=np.float32)

    def _shaping_potential(self):
        # phi(s) = -(W_alt*|h| + W_vel*|v|). The per-step reward phi(s') - phi(s)
        # pays for getting closer and slower. Potential-based: it speeds learning
        # without changing which policy is optimal.
        return -(config.SHAPING_W_ALTITUDE * abs(self.sim.h)
                 + config.SHAPING_W_VELOCITY * abs(self.sim.v))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.steps = 0
        self.prev_throttle = config.INIT_THROTTLE
        return self._send_agent_state(), {}

    def step(self, action):
        if self.steps == 0:
            throttle = config.INIT_THROTTLE
        else:
            desired = float(action[0])
            delta = np.clip(desired - self.prev_throttle,
                            -config.MAX_THROTTLE_RATE, config.MAX_THROTTLE_RATE)
            throttle = self.prev_throttle + delta
        self.prev_throttle = throttle

        phi_before = self._shaping_potential()
        mass_before = self.sim.m
        terminated = self.sim.step(throttle)  # True when the rocket reaches ground
        phi_after = self._shaping_potential()
        self.steps += 1

        # Running out of fuel while still airborne is its own terminal failure.
        out_of_fuel = (not terminated) and (self.sim.m <= self.sim.dry_mass + 1e-9)
        if out_of_fuel:
            terminated = True

        truncated = (not terminated) and (self.steps >= self.max_steps)

        fuel_used = mass_before - self.sim.m  # kg burned this step
        reward_shaping = phi_after - phi_before
        reward_fuel = -config.FUEL_PENALTY * fuel_used
        # Living cost for every step not-yet-landed: makes hovering bleed reward, so
        # the policy can't sit just above the pad collecting (near-max) shaping forever.
        reward_time = 0.0 if terminated else -config.TIME_PENALTY
        reward_terminal = 0.0

        info = {}
        if out_of_fuel:
            reward_terminal = config.OUT_OF_FUEL_PENALTY
            info["out_of_fuel"] = True
        elif terminated:
            impact_speed = abs(self.sim.v)
            # Any touchdown is rewarded; softer = much more, never negative (no crash cliff).
            reward_terminal = config.LANDING_BONUS * np.exp(-impact_speed / config.SOFTNESS_TAU)
            info["impact_speed"] = impact_speed
        elif truncated:
            reward_terminal = config.TIMEOUT_PENALTY
            info["timeout"] = True

        reward = reward_shaping + reward_fuel + reward_time + reward_terminal
        info["reward_shaping"] = reward_shaping
        info["reward_fuel"] = reward_fuel
        info["reward_time"] = reward_time
        info["reward_terminal"] = reward_terminal

        return self._send_agent_state(), reward, terminated, truncated, info
