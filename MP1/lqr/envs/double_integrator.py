import logging
import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from os import path


class DoubleIntegratorEnv(gymnasium.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }

    def __init__(self, max_acc=1., init_y=3, init_v=4, render_mode=None):
        self.dt = .05
        self.max_acc = max_acc 
        self.action_space = spaces.Box(
            low=-max_acc,
            high=max_acc, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, shape=(2,),
            dtype=np.float32
        )
        self.init_y = init_y 
        self.init_v = init_v

        self.screen = None
        self.clock = None
        self.render_mode = render_mode

    def step(self, u):
        u = np.clip(u, -self.max_acc, self.max_acc)[0]
        y, v = self.state
        newv = v + u*self.dt
        newy = y + v*self.dt
        self.state = np.array([newy, newv])
        costs = newv**2 + newy**2 + u**2
        self.last_u = u
        success = np.abs(y) < 0.1 and np.abs(v) < 0.1
        truncated = False
        return self._get_obs(), -costs, False, truncated, {'metric': {'success': success}}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        high = np.array([self.init_y, self.init_v])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        info = {'seed': seed}
        return self._get_obs(), info

    def _get_obs(self):
        y, v = self.state
        return np.array([y, v]).astype(np.float32)
    
    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gymnasium.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasiumnasium[classic_control]"`'
            ) from e
        width, height = 500, 200
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((width, height))
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((width, height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((width, height))
        self.surf.fill((255, 255, 255))
        
        w = 0.6; h = 0.4
        l, r, t, b = -w/2, w/2, h/2, -h/2
        coords = np.array([[l,b], [l,t], [r,t], [r,b]])
        transformed_coords = coords + 0
        transformed_coords[:,0] += self.state[0]
        offset = np.array([[250, 100]])
        scale = 500/10
        transformed_coords = transformed_coords * scale + offset
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.hline(self.surf, 0, 500, 110, (0, 0, 0))
        
        if self.last_u:
            gfxdraw.hline(self.surf, int(self.state[0]*scale + offset[0,0]), 
                          int((self.state[0] + self.last_u)*scale + offset[0,0]), 
                          100, (0, 0, 0))
        
        self.screen.blit(self.surf, (0,0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
