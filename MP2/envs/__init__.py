import numpy as np
from gymnasium.envs.registration import register
from .cartpole import CartPoleEnv

register(
    id='CartPole-v2',
    entry_point='envs:CartPoleEnv',
)
