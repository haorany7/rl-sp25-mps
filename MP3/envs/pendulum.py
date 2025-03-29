import logging
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from os import path
from PIL import Image
import pygame

class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0, max_speed=np.inf, max_torque=np.inf,
                 init_theta=np.pi, init_thetadot=1, noise=0., visual=False):
        # logging.info('PendulumEnv.max_torque: %f', max_torque)
        # logging.info('PendulumEnv.max_speed: %f', max_speed)
        # logging.info('PendulumEnv.init_theta: %f', init_theta)
        # logging.info('PendulumEnv.init_thetadot: %f', init_thetadot)
        # logging.info('PendulumEnv.noise: %f', noise)

        self.init_theta = init_theta
        self.init_thetadot = init_thetadot
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.noise = noise
        
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None
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
        self.visual = visual 
        self.is_terminal = False

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + thdot ** 2 + (u ** 2)

        thnoise = self.np_random.normal(0, 1) * self.noise
        thdotnoise = self.np_random.normal(0, 1) * self.noise
        
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
        if seed is not None:
            self.seed(seed)
        if options is not None:
            pass
        high = np.array([self.init_theta, self.init_thetadot])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.states.append(self.state)
        self.last_u = None
        self.total_time = 0
        self.total_time_upright = 0
        return np.array(self._get_obs(), dtype=np.float32)
    
    def reset_to_state(self, state):
        self.state = np.array(state)
        self.states.append(self.state)
        self.last_u = None
        self.total_time = 0
        self.total_time_upright = 0
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        if self.visual:
            _ = self.last_u
            self.last_u = 0
            out = Image.fromarray(self.render(mode='rgb_array'))
            self.last_u = _
        else:
            theta, thetadot = self.state
            out = np.array([theta, thetadot])
        return out

    def render(self, mode='human'):
        import pygame
        import numpy as np
        from os import path

        if self.viewer is None:
            pygame.init()
            self.screen_width = 500
            self.screen_height = 500
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height),
                pygame.DOUBLEBUF | pygame.HWSURFACE
            )
            self.backbuffer = pygame.Surface((self.screen_width, self.screen_height)).convert()
            pygame.display.set_caption("Pendulum")
            self.clock = pygame.time.Clock()
            # Define world bounds similar to classic_control viewer
            self.x_min, self.x_max = -1.3, 1.3
            self.y_min, self.y_max = -1.3, 1.3
            self.scale_x = self.screen_width / (self.x_max - self.x_min)
            self.scale_y = self.screen_height / (self.y_max - self.y_min)
            # Initialize image holder for control input indicator
            self.img = None
            self.viewer = self.screen

        # Handle events without blocking
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Clear the backbuffer
        self.backbuffer.fill((255, 255, 255))

        # Helper function to convert world coordinates to screen pixels
        def world_to_screen(x, y):
            pixel_x = int((x - self.x_min) * self.scale_x)
            pixel_y = int(self.screen_height - (y - self.y_min) * self.scale_y)
            return (pixel_x, pixel_y)

        # Pendulum parameters
        pivot_world = (0, 0)
        L = 1.0  # length of the rod
        theta = self.state[0]
        # Calculate the end of the rod so that when theta=0, the pendulum hangs downward
        end_world = (L * np.sin(theta), -L * np.cos(theta))
        pivot_screen = world_to_screen(*pivot_world)
        end_screen = world_to_screen(*end_world)

        # Draw the rod as a line
        rod_width = max(int(0.2 * self.scale_x), 1)  # thickness in pixels
        pygame.draw.line(self.backbuffer, (204, 102, 102), pivot_screen, end_screen, rod_width)

        # Draw the axle as a small circle at the pivot
        axle_radius = max(int(0.05 * self.scale_x), 1)
        pygame.draw.circle(self.backbuffer, (0, 0, 0), pivot_screen, axle_radius)

        # Optionally, draw the control input indicator image if last_u is set
        if self.last_u is not None:
            if self.img is None:
                fname = path.join(path.dirname(__file__), "assets/clockwise.png")
                try:
                    self.img = pygame.image.load(fname).convert_alpha()
                except Exception as e:
                    self.img = None
            if self.img is not None:
                # Determine a scaling factor based on last_u (control input)
                scale_factor = abs(self.last_u) / 2.0
                pixel_scale = scale_factor * self.scale_x  # convert world units to pixels
                img_width, img_height = self.img.get_size()
                new_width = max(int(img_width * pixel_scale), 1)
                new_height = max(int(img_height * pixel_scale), 1)
                scaled_img = pygame.transform.scale(self.img, (new_width, new_height))
                # Flip image horizontally if last_u is negative
                if self.last_u < 0:
                    scaled_img = pygame.transform.flip(scaled_img, True, False)
                # Blit the image at the pivot (centered)
                img_rect = scaled_img.get_rect(center=pivot_screen)
                self.backbuffer.blit(scaled_img, img_rect)

        # Blit the backbuffer to the screen and update the display
        self.screen.blit(self.backbuffer, (0, 0))
        pygame.display.flip()
        self.clock.tick(60)

        # Convert the screen to a numpy array and return it
        array3d = pygame.surfarray.array3d(self.screen)
        array3d = np.transpose(array3d, (1, 0, 2))
        return array3d

    def close(self):
        if self.viewer:
            pygame.quit()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
