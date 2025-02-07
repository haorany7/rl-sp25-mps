import gymnasium as gym
import numpy as np
import logging
import time
from PIL import Image
from absl import app
from absl import flags
from lqr_solve import LQRControl

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', 1, 'Number of episodes to evaluate.')
flags.DEFINE_string('env_name', None, 'Name of environment.')
flags.DEFINE_boolean('vis', False, 'To visualize or not.')
flags.DEFINE_boolean('vis_save', False, 'To save visualization or not')
flags.DEFINE_float('pendulum_noise', 0.0, 'Standard deviation for additive gaussian noise for balancing a pendulum.')

def setup_logging():
    log_formatter = logging.Formatter(
        '%(asctime)s: %(levelname)s %(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logging.getLogger().handlers = []
    if not len(logging.getLogger().handlers):
        logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO)

def run(env_name, num_episodes, vis, vis_save):
    setup_logging()
    # This import is needed here to make sure that pendulum_noise can be passed
    # in through a command line argument.
    import envs
    env = gym.make(env_name, render_mode="rgb_array")
    if vis:
      from gymnasium.wrappers import HumanRendering
      env = HumanRendering(env)
    total_rewards, total_metrics = [], []
    state, reset_info = env.reset(seed=1)
    for i in range(num_episodes):
        reward_i = 0
        state, reset_info = env.reset()
        states, controls = [state], []
        controller = LQRControl(env, state)
        done = False
        gif = []
        for j in range(200):
            action = controller.act(state)
            state, reward, done, truncated, info = env.step(action)
            if vis_save:
                img = env.render()
                gif.append(Image.fromarray(img))
            states.append(state)
            controls.append(action)
            reward_i += reward
        if vis_save:
            gif[0].save(fp=f'vis-{env.unwrapped.spec.id}-{i}.gif',
                        format='GIF', append_images=gif,
                        save_all=True, duration=50, loop=0)
        metric_name = list(info['metric'].keys())[0]
        metric_value = info['metric'][metric_name]
        total_metrics += [metric_value]
        logging.error('Final State: %7.3f, %7.3f. Episode Cost: %9.3f, %s: %7.3f.',
                      state[0], state[1], -reward_i, metric_name, metric_value)
        total_rewards += [reward_i]
    logging.error('Average Cost: %7.3f', -np.mean(total_rewards))
    logging.error('%s: %7.3f', metric_name, np.mean(total_metrics))
    env.close()
    return -np.mean(total_rewards) 

def main(_):
  setup_logging()
  run(FLAGS.env_name, FLAGS.num_episodes, FLAGS.vis, FLAGS.vis_save)

if __name__ == '__main__':
    app.run(main)
