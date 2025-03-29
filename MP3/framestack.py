import gymnasium as gym
from collections import deque
from gymnasium import spaces 
import numpy as np
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self, seed=None, options=None):
        ob = self.env.reset(seed=seed, options=options)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, _, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, False, info

    def _get_ob(self):
        states = np.array(self.frames)
        return np.concatenate(states,axis=-1)

    def __getattr__(self, name):
        return getattr(self.env, name)
