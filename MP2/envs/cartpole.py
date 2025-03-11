"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gymnasium as gym
from gymnasium import spaces, logger
from gymnasium.utils import seeding
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

    def __init__(self):
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

        self.steps_beyond_done = None
        self.is_terminal = False
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # action = action[0]
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

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, False, {'metric': {'none': 0}}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        if options is not None:
            # e.g., max_episode_steps = options.get("max_episode_steps", 200)
            pass
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def render(self):
        if self.viewer is None:
            pygame.init()
            self.screen_width = 600
            self.screen_height = 400
            # Initialize main screen with double buffering
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height),
                pygame.DOUBLEBUF | pygame.HWSURFACE
            )
            # Create backbuffer surface for drawing operations (converted for speed)
            self.backbuffer = pygame.Surface((self.screen_width, self.screen_height)).convert()
            pygame.display.set_caption("CartPole")
            self.clock = pygame.time.Clock()
            
            # Geometry parameters
            self.cartwidth = 50
            self.cartheight = 30
            self.polewidth = 10
            self.polelen = 100
            self.axleoffset = self.cartheight / 4.0
            self.carty = self.screen_height * 0.75
            self.x_threshold = 2.4  # Must match your environment's limits
            self.scale = self.screen_width / (self.x_threshold * 2)
            
            # Set viewer so that initialization runs only once
            self.viewer = self.screen

        # Handle events without blocking
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Clear backbuffer first (not the visible screen)
        self.backbuffer.fill((255, 255, 255))  # White background

        # Calculate clamped cart position
        cart_pos = np.clip(self.state[0], -self.x_threshold, self.x_threshold)
        cartx = cart_pos * self.scale + self.screen_width / 2.0

        # Draw track (static element)
        pygame.draw.line(self.backbuffer, (0, 0, 0),
                         (0, self.carty),
                         (self.screen_width, self.carty), 2)

        # Only draw cart/pole if visible
        if (-self.cartwidth/2 < cartx < self.screen_width + self.cartwidth/2
                and not self.is_terminal):  # Stop drawing when terminated
            # Draw cart
            cart_rect = pygame.Rect(
                cartx - self.cartwidth/2,
                self.carty - self.cartheight/2,
                self.cartwidth,
                self.cartheight
            )
            pygame.draw.rect(self.backbuffer, (100, 100, 200), cart_rect)

            # Draw pole
            angle = self.state[2]
            pole_top_x = cartx + (self.polelen * np.sin(angle))
            pole_top_y = self.carty - (self.polelen * np.cos(angle)) - self.axleoffset
            pygame.draw.line(self.backbuffer, (204, 153, 102),
                             (cartx, self.carty - self.axleoffset),
                             (pole_top_x, pole_top_y),
                             self.polewidth)

            # Draw axle
            pygame.draw.circle(self.backbuffer, (127, 127, 204),
                               (int(cartx), int(self.carty - self.axleoffset)),
                               int(self.polewidth/2))

        # Blit complete frame to visible screen
        self.screen.blit(self.backbuffer, (0, 0))
        pygame.display.flip()
        self.clock.tick(60)  # Maintain consistent frame rate

        return self.screen

    def close(self):
        if self.viewer:
            pygame.quit()
            self.viewer = None
