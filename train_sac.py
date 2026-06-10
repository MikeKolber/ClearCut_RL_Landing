"""Train the lander with SAC. Run: python train_sac.py
Outputs are prefixed 'sac_' (sac_lander.zip, sac_training_curve.png, ...).

SAC is off-policy (replay buffer -> more sample-efficient) and auto-tunes its own
exploration (ent_coef='auto' by default), so the setup is deliberately minimal.
"""

import config
from stable_baselines3 import SAC
from train_common import run_training


def build_sac(env):
    return SAC(
        "MlpPolicy", env,
        gamma=config.GAMMA,
        learning_rate=config.LEARNING_RATE,  # SAC uses a constant LR; no decay needed
        seed=config.SEED,
        verbose=1,
        # ent_coef left at its default "auto": SAC tunes exploration automatically.
    )


if __name__ == "__main__":
    run_training("sac", build_sac, config.SAC_TIMESTEPS)
