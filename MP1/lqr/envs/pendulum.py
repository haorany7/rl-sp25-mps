import logging
import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from os import path


class PendulumEnv(gymnasium.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }

    def __init__(self, g=10.0, max_speed=np.inf, max_torque=np.inf,
                 init_theta=np.pi, init_thetadot=1, noise=0., render_mode=None):
        logging.info('PendulumEnv.max_torque: %f', max_torque)
        logging.info('PendulumEnv.max_speed: %f', max_speed)
        logging.info('PendulumEnv.init_theta: %f', init_theta)
        logging.info('PendulumEnv.init_thetadot: %f', init_thetadot)
        logging.info('PendulumEnv.noise: %f', noise)

        self.init_theta = init_theta
        self.init_thetadot = init_thetadot
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.noise = noise
        
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.screen = None
        self.screen_dim = 500
        self.clock = None

        self.states = []
        self.controls = []

        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        high = np.array([np.inf, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )
        self.total_time = 0
        self.total_time_upright = 0
        self.render_mode = render_mode

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + thdot ** 2 + (u ** 2)

        thnoise = self.np_random.standard_normal()*self.noise
        thdotnoise = self.np_random.standard_normal()*self.noise
        
        newth = th + thdot * dt 
        newth = angle_normalize(newth + thnoise)
        
        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. * u / (m * l ** 2)) * dt
        newthdot = newthdot + thdotnoise
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        self.states.append(self.state)
        self.controls.append(u)
        self.total_time += 1
        self.total_time_upright += np.abs(th) < 0.1
        metric = {'fraction_upright': self.total_time_upright / self.total_time}
        return self._get_obs(), -costs, False, False, {'metric': metric}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        high = np.array([self.init_theta, self.init_thetadot])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.states.append(self.state)
        self.last_u = None
        self.total_time = 0
        self.total_time_upright = 0
        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([theta, thetadot]).astype(np.float32)
    
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

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if self.last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (
                    float(scale * np.abs(self.last_u) / 2),
                    float(scale * np.abs(self.last_u) / 2),
                ),
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
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

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
