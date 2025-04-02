import gymnasium as gym
import os
import numpy as np
from pathlib import Path
import pdb
from tqdm import tqdm
import envs
import logging
import time
import torch
import utils
from absl import app
from absl import flags
from policies import NNPolicy, CNNPolicy
from evaluation import val, test_model_in_env
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter
from framestack import FrameStack
from train_bc import train_model

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes_val', 100, 'Number of episodes to evaluate.')
flags.DEFINE_integer('num_episodes_train', 250, 'Number of episodes to train.')
flags.DEFINE_integer('episode_len', 200, 'Length of each episode at test time.')
flags.DEFINE_string('env_name', 'CartPole-v2', 'Name of environment.')
flags.DEFINE_boolean('vis', False, 'To visualize or not.')
flags.DEFINE_boolean('vis_save', False, 'To save visualization or not')
flags.DEFINE_string('logdir', None, 'Directory to store loss plots, etc.')
flags.DEFINE_string('datadir', 'data/', 'Directory with expert data.')
flags.mark_flag_as_required('logdir')

def get_dims(env_name):
    if env_name == 'CartPole-v2':
        discrete = True
        return 4, 2, discrete 
    elif env_name == 'DoubleIntegrator-v1':
        discrete = False
        return 2, 1, discrete
    elif env_name == 'PendulumBalance-v1':
        discrete = False
        return 2, 1, discrete

def load_data(num_episodes):
    datadir = Path(FLAGS.datadir)
    
    # Load training data for training the policy.
    dt = utils.load_variables(datadir / f'{FLAGS.env_name}.pkl')
    dt['states'] = dt['states'][:num_episodes,:]
    dt['actions'] = dt['actions'][:num_episodes,:]
    return dt
    
def main(_):
    logdir = Path(FLAGS.logdir) / FLAGS.env_name
    logdir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    torch.set_num_threads(4)
    
    # Generate episode counts on log scale
    episode_counts = []
    current = FLAGS.num_episodes_train
    while current >= 1:
        episode_counts.append(current)
        current = current // 2
    
    results = []
    for num_eps in episode_counts:
        print(f"\n=== Training with {num_eps} episodes ===")
        
        # Load data with current number of episodes
        dt = load_data(num_eps)
        
        # Setup and train model
        state_dim, action_dim, discrete = get_dims(FLAGS.env_name)
        model = NNPolicy(state_dim, [16, 32, 64], action_dim, discrete)
        model = model.to(device)
        train_model(model, logdir, dt['states'], dt['actions'], device, discrete)
        model = model.eval()

        # Evaluate
        val_envs = [gym.make(FLAGS.env_name) for _ in range(FLAGS.num_episodes_val)]
        [env.reset(seed=i+1000) for i, env in enumerate(val_envs)]
        reward = val(model, device, val_envs, FLAGS.episode_len, False)
        [env.close() for env in val_envs]
        
        results.append({
            'num_episodes': num_eps,
            'reward': reward
        })
        print(f"Average reward with {num_eps} episodes: {reward}")

    # Plot results
    plt.figure(figsize=(10,6))
    episodes = [r['num_episodes'] for r in results]
    rewards = [r['reward'] for r in results]
    plt.semilogx(episodes, rewards, 'o-')
    plt.xlabel('Number of Expert Episodes (log scale)')
    plt.ylabel('Average Reward')
    plt.title(f'Data Efficiency - {FLAGS.env_name}')
    plt.grid(True)
    plt.savefig(f'{logdir}/data_efficiency.png')
    
    # Save numerical results
    with open(f'{logdir}/data_efficiency_results.txt', 'w') as f:
        for r in results:
            f.write(f"Episodes: {r['num_episodes']}, Reward: {r['reward']:.2f}\n")

if __name__ == '__main__':
    app.run(main)
