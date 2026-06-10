"""Train the lander with PPO. Run: python train_ppo.py
Outputs are prefixed 'ppo_' (ppo_lander.zip, ppo_training_curve.png, ...)."""

import config
from stable_baselines3 import PPO
from train_common import run_training, linear_schedule


def build_ppo(env):
    return PPO(
        "MlpPolicy", env,
        gamma=config.GAMMA,
        learning_rate=linear_schedule(config.LEARNING_RATE),  # decayed to 0
        ent_coef=config.ENT_COEF,                             # entropy bonus
        seed=config.SEED,
        verbose=1,
    )


if __name__ == "__main__":
    run_training("ppo", build_ppo, config.PPO_TIMESTEPS)
