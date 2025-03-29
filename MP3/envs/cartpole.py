"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gymnasium as gym
from gymnasium import spaces, logger
from gymnasium.utils import seeding
from PIL import Image
import numpy as np
import pygame

class CartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, visual=False,png=False):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.visual = visual
        self.png = png
        self.total_fames = self.upright_frames = 0
        self.is_terminal = False
        self.screen = None
        self.window = None
        self.clock = None
        self.track_background = None
        
        # Make sure to add this to your metadata
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render.modes": ["human", "rgb_array"],  # For backwards compatibility
            "render_fps": 50
        }

        self.steps_beyond_done = None
        if visual:
            self.observation_space = spaces.Box(0,255,(210,160,3),dtype=np.uint8)
            # self.observation_space = spaces.Box(0,255,(200,600,3),dtype=np.uint8)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if not isinstance(action,np.int64) and not isinstance(action,int):
            action = action[0]
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        self.total_fames += 1
        if not done:
            reward = 1.0
            self.upright_frames += 1
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
            self.upright_frames += 1
        else:
            self.steps_beyond_done += 1
            reward = 0.0
            self.upright_frames += 0

        info = {'metric': {'upright_frames': self.upright_frames / self.total_fames}}
        return self.get_observation(), reward, done, False, info

    def get_observation(self):
        if self.visual:
            if self.png:
                out = Image.fromarray(self.render(mode='rgb_array')).resize((160,210))
            else:
                self.render_mode = "rgb_array"
                frame = self.render()
                if frame is not None:
                    out = np.array(Image.fromarray(self.render()).resize((160,210)))
                else:
                    out = np.zeros((210, 160, 3), dtype=np.uint8)
        else:
            out = np.array(self.state)
        return out

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        if options is not None:
            pass
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.upright_frames = 0
        self.total_fames = 0
        return self.get_observation()
        
    def reset_to_state(self, state):
        self.state = state+0
        self.steps_beyond_done = None
        self.upright_frames = 0
        self.total_fames = 0
        return self.get_observation()

    def render(self, render_mode=None):
        screen_width = 600
        screen_height = 200
        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 50
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0
        
        import pygame
        if not hasattr(self, '_initialized') or not self._initialized:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((screen_width, screen_height))
            self.clock = pygame.time.Clock()
            self._initialized = True
            
        if self.screen is None:
            import pygame
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("CartPole")
            self.window = pygame.display.set_mode((screen_width, screen_height))
            self.screen = pygame.Surface((screen_width, screen_height))
            self.clock = pygame.time.Clock()
            
            # Create background track
            self.track_background = pygame.Surface((screen_width, screen_height))
            self.track_background.fill((255, 255, 255))
            pygame.draw.line(
                self.track_background,
                (0, 0, 0),
                (0, carty),
                (screen_width, carty),
                1
            )
        
        if self.state is None:
            return None
        
        # Clear screen
        self.screen.fill((255, 255, 255))
        
        # Add track
        self.screen.blit(self.track_background, (0, 0))
        
        # Draw cart
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        
        x = self.state
        cartx = x[0] * scale + screen_width / 2.0
        cart_coords = [(cartx + l, carty + b), (cartx + l, carty + t), (cartx + r, carty + t), (cartx + r, carty + b)]
        
        import pygame
        import pygame.gfxdraw
        pygame.gfxdraw.filled_polygon(self.screen, cart_coords, (0, 0, 0))
        
        # Draw pole
        axleoffset = cartheight / 4.0
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        
        # Rotate pole according to angle
        pole_coords = []
        pole_angle = -x[2]
        
        import math
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord_x = coord[0] * math.cos(pole_angle) - coord[1] * math.sin(pole_angle)
            coord_y = coord[0] * math.sin(pole_angle) + coord[1] * math.cos(pole_angle)
            pole_coords.append((cartx + coord_x, carty + axleoffset + coord_y))
        
        pygame.gfxdraw.filled_polygon(self.screen, pole_coords, (204, 153, 102))
        
        # Draw axle
        pygame.gfxdraw.filled_circle(
            self.screen,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth/2),
            (127, 127, 204)
        )

        import numpy as np
        # Get the raw image
        img = np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )
        # Flip the image vertically to fix orientation
        img = np.flipud(img)
        img_copy = img.copy()
        
        # Convert the flipped numpy array back to a pygame surface
        import pygame
        flipped_surface = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
            
        # Clear the window and display the flipped image
        self.window.fill((255, 255, 255))
        self.window.blit(flipped_surface, (0, 0))
            
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

        return img.copy()

    def close(self):
        if self.viewer:
            pygame.quit()
            self.viewer = None

