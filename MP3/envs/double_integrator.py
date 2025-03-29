import logging
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from os import path
from PIL import Image
import pygame

class DoubleIntegratorEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, max_acc=1., init_y=3, init_v=4, visual=False):
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
        self.visual = visual

        self.seed()
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        u = np.clip(u, -self.max_acc, self.max_acc)[0]
        y, v = self.state
        newv = v + u*self.dt
        newy = y + v*self.dt
        self.state = np.array([newy, newv])
        costs = newv**2 + newy**2 + u**2
        self.last_u = u
        success = np.abs(y) < 0.1 and np.abs(v) < 0.1
        return self._get_obs(), -costs, False, False, {'metric': {'success': success}}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        if options is not None:
            pass
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(2,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)
    
    def reset_to_state(self, state):
        self.state = np.array(state)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        if self.visual:
            _ = self.last_u
            self.last_u = 0
            out = Image.fromarray(self.render(mode='rgb_array'))
            self.last_u = _
        else:
            y, v = self.state
            out = np.array([y, v])
        return out

    def render(self, mode='human'):
        import pygame
        import numpy as np

        if self.viewer is None:
            pygame.init()
            self.screen_width = 600
            self.screen_height = 400
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height),
                pygame.DOUBLEBUF | pygame.HWSURFACE
            )
            self.backbuffer = pygame.Surface((self.screen_width, self.screen_height)).convert()
            pygame.display.set_caption("Double Integrator")
            self.clock = pygame.time.Clock()

            # Define world bounds: x ∈ [-5, 5], y ∈ [-2, 2]
            self.x_min, self.x_max = -5.0, 5.0
            self.y_min, self.y_max = -2.0, 2.0
            # Scale factors for mapping world coordinates to pixels.
            self.scale_x = self.screen_width / (self.x_max - self.x_min)  # 600/10 = 60
            self.scale_y = self.screen_height / (self.y_max - self.y_min)   # 400/4 = 100

            # Agent (box) dimensions in world units.
            self.box_width = 0.6
            self.box_height = 0.4

            self.viewer = self.screen

        # Handle events without blocking.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Clear the backbuffer.
        self.backbuffer.fill((255, 255, 255))  # White background

        # Helper function to map world coordinates to screen pixels.
        def world_to_screen(x, y):
            pixel_x = int((x - self.x_min) * self.scale_x)
            # Flip y: world y_min maps to screen bottom.
            pixel_y = int(self.screen_height - (y - self.y_min) * self.scale_y)
            return pixel_x, pixel_y

        # Draw the track line at y = -box_height/2.
        track_y = -self.box_height / 2.0
        start_line = world_to_screen(self.x_min, track_y)
        end_line = world_to_screen(self.x_max, track_y)
        pygame.draw.line(self.backbuffer, (0, 0, 0), start_line, end_line, 2)

        # Draw the agent as a rectangle (box) centered at (state[0], 0).
        agent_x = self.state[0]
        agent_y = 0.0
        # Compute the bottom-left world coordinate of the box.
        box_left = agent_x - self.box_width / 2.0
        box_bottom = agent_y - self.box_height / 2.0

        # Convert the top-left (needed for pygame.Rect) to screen coordinates.
        top_left = world_to_screen(box_left, agent_y + self.box_height / 2.0)
        box_pixel_width = int(self.box_width * self.scale_x)
        box_pixel_height = int(self.box_height * self.scale_y)
        rect = pygame.Rect(top_left[0], top_left[1], box_pixel_width, box_pixel_height)
        pygame.draw.rect(self.backbuffer, (204, 102, 102), rect)

        # Optionally, draw an arrow indicating the last control input.
        if self.last_u is not None:
            start_arrow = world_to_screen(self.state[0], 0)
            end_arrow = world_to_screen(self.state[0] + self.last_u, 0)
            pygame.draw.line(self.backbuffer, (0, 0, 255), start_arrow, end_arrow, 2)

        # Blit the backbuffer to the screen and update the display.
        self.screen.blit(self.backbuffer, (0, 0))
        pygame.display.flip()
        self.clock.tick(60)  # Maintain a consistent frame rate

        # Convert the current screen to a numpy array.
        array3d = pygame.surfarray.array3d(self.screen)
        array3d = np.transpose(array3d, (1, 0, 2))
        return array3d

    def close(self):
        if self.viewer:
            pygame.quit()
            self.viewer = None
