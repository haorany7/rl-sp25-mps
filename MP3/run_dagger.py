import gymnasium as gym
import os
import numpy as np
from pathlib import Path
import pdb
import envs
import torch
from absl import app
from absl import flags
from policies import NNPolicy, CNNPolicy
from evaluation import val, test_model_in_env
from tqdm import tqdm
from framestack import FrameStack
from stable_baselines3 import PPO
import random

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes_val', 100, 'Number of episodes to evaluate.')
flags.DEFINE_string('env_name', 'CartPole-v2', 'Name of environment.')
flags.DEFINE_boolean('vis', False, 'To visualize or not.')
flags.DEFINE_boolean('vis_save', False, 'Save visualization')
flags.DEFINE_string('logdir', './dagger_out', 'Directory to store loss plots, etc.')
flags.DEFINE_string('version', 'both', 'Version to run: both/none/persistence/penalty')

def get_dims(env_name):
    if env_name == 'CartPole-v2':
        discrete = True
        return 4, 2, discrete 
    else:
        raise Exception(f'Not Implemented')

def main(_):
    # Modify logdir to include version
    logdir = Path(FLAGS.logdir) / FLAGS.env_name / FLAGS.version  # Add version to path
    logdir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    state_dim,action_dim,discrete = get_dims(FLAGS.env_name)

    # Load environment with framestack 4
    stack_states =4 
    env = FrameStack(gym.make('VisualCartPole-v2'),stack_states)

    # setup CNN based policy and learning boilerplate
    c,h,w = (3, 210, 160)
    learner = CNNPolicy(stack_states, (c, h, w), [16, 32, 64], action_dim, discrete)
    learner.to(device)

    # load the PPO based expert policy
    senv = gym.make('CartPole-v2')
    expert = PPO('MlpPolicy', senv, verbose=1)
    expert = PPO.load('./ppo-cartpole-expert.zip')
    # query expert actions at given states
    def query_expert(state):
        return expert.predict(state.cpu().numpy(),deterministic=True)[0]

    # Modify this part to load different versions with correct file names
    if FLAGS.version == 'both':
        from dagger_trainer import dagger_trainer  # original version with both modifications
    elif FLAGS.version == 'persistence':
        from dagger_trainer_with_persistence import dagger_trainer  # only persistence
    elif FLAGS.version == 'penalty':
        from dagger_trainer_with_penalty import dagger_trainer  # only penalty
    elif FLAGS.version == 'none':
        from dagger_trainer_with_none import dagger_trainer  # no modifications
    
    dagger_trainer(env, learner, query_expert, device)
        
    # Evaluation
    val_envs = [FrameStack(gym.make('VisualCartPole-v2'),stack_states) for _ in range(FLAGS.num_episodes_val)]
    [env.reset(seed=i+1000) for i, env in enumerate(val_envs)]
    val(learner, device, val_envs, 200, visual=False)
    [env.close() for env in val_envs]

    if FLAGS.vis or FLAGS.vis_save:
        env_vis = FrameStack(gym.make('VisualCartPole-v2'),stack_states)
        state, g, gif, info = test_model_in_env(
            learner, env_vis, 200, device, vis=FLAGS.vis, 
            vis_save=FLAGS.vis_save, visual=True)
        if FLAGS.vis_save:  
            import pathlib
            pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)
            gif[0].save(fp=f'{logdir}/vis-{FLAGS.version}-{env_vis.unwrapped.spec.id}.gif',  # Add version to filename
                        format='GIF', append_images=gif,
                        save_all=True, duration=50, loop=0)
        env_vis.close()

if __name__ == '__main__':
    app.run(main)
