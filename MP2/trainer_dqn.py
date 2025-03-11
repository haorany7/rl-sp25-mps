import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Parameters
num_steps_per_rollout = 5 
num_updates = 10000
reset_every = 200
val_every = 10000

replay_buffer_size = 1000000
q_target_update_every = 50
q_batch_size = 256
q_num_steps = 1

gamma = 0.99  # Discount factor
learning_rate = 1e-3  # Learning rate
epsilon_start = 1.0  # Initial exploration rate
epsilon_end = 0.05  # Final exploration rate
epsilon_decay = 0.995  # Decay rate per update

#define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(writer, iteration, name, value, print_every=10, log_every=10):
    # A simple function to let you log progress to the console and tensorboard.
    if np.mod(iteration, print_every) == 0:
        if name is not None:
            print('{:8d}{:>30s}: {:0.3f}'.format(iteration, name, value))
        else:
            print('')
    if name is not None and np.mod(iteration, log_every) == 0:
        writer.add_scalar(name, value, iteration)

# Implement a replay buffer class that a) stores rollouts as they
# come along, overwriting older rollouts as needed, and b) allows random
# sampling of transition quadruples for training of the Q-networks.
class ReplayBuffer(object):
    def __init__(self, size, state_dim, action_dim):
        self.buffer = deque(maxlen=size)

    def insert(self, rollouts):
        self.buffer.extend(rollouts)

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

class PrioritizedReplayBuffer:
    def __init__(self, size, alpha=0.6):
        self.buffer = deque(maxlen=size)
        self.priorities = deque(maxlen=size)
        self.alpha = alpha

    def insert(self, transition, td_error):
        self.buffer.append(transition)
        priority = (abs(td_error) + 1e-5) ** self.alpha
        self.priorities.append(priority)

    def sample_batch(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], []

        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # Importance-sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-5) ** self.alpha

# Starting off from states in envs, rolls out num_steps_per_rollout for each
# environment using the policy in `model`. Returns rollouts in the form of
# states, actions, rewards and new states. Also returns the state the
# environments end up in after num_steps_per_rollout time steps.
def collect_rollouts(models, envs, states, num_steps_per_rollout, epsilon, device):
    rollouts = []
    for _ in range(num_steps_per_rollout):
        actions = []
        for model, state, env in zip(models, states, envs):
            if np.random.rand() < epsilon:
                # Use the environment's action_space to sample a random action
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor.unsqueeze(0))
                action = torch.argmax(q_values).item()
            actions.append(action)

        next_states, rewards, dones, _ = zip(*[env.step(action) for env, action in zip(envs, actions)])
        rollouts.extend(zip(states, actions, rewards, next_states, dones))
        states = [next_state if not done else env.reset() for next_state, done, env in zip(next_states, dones, envs)]
    return rollouts, states
   
# Function to train the Q function. Samples q_num_steps batches of size
# q_batch_size from the replay buffer, runs them through the target network to
# obtain target values for the model to regress to. Takes optimization steps to
# do so. Returns the bellman_error for plotting.
def update_model(replay_buffer, models, targets, optimizers, gamma, action_dim, q_batch_size, q_num_steps, device):
    total_bellman_error = 0.
    for _ in range(q_num_steps):
        if len(replay_buffer.buffer) < q_batch_size:
            continue  # Skip update if not enough samples

        batch = replay_buffer.sample_batch(q_batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        for model, target, optimizer in zip(models, targets, optimizers):
            q_values = model(states).gather(1, actions).squeeze(1)

            with torch.no_grad():
                next_q_values = target(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * gamma * next_q_values

            loss = nn.MSELoss()(q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_bellman_error += loss.item()
    return total_bellman_error / q_num_steps

def train_model_dqn(models, targets, state_dim, action_dim, envs, gamma, device, logdir, val_fn):
    train_writer = SummaryWriter(logdir / 'train')
    val_writer = SummaryWriter(logdir / 'val')
    
    # Initialize a list of optimizers, one for each model
    optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in models]

    replay_buffer = ReplayBuffer(replay_buffer_size, state_dim, 1)

    num_steps, total_samples = 0, 0
    states = [e.reset() for e in envs]
    
    for updates_i in range(num_updates):
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** updates_i))

        [m.train() for m in models]

        if np.mod(updates_i, q_target_update_every) == 0:
            for model, target in zip(models, targets):
                target.load_state_dict(model.state_dict())
        
        rollouts, states = collect_rollouts(models, envs, states, num_steps_per_rollout, epsilon, device)
        num_steps += num_steps_per_rollout
        total_samples += num_steps_per_rollout * len(envs)
        
        replay_buffer.insert(rollouts)

        bellman_error = update_model(replay_buffer, models, targets, optimizers, gamma, action_dim, q_batch_size, q_num_steps, device)
        log(train_writer, updates_i, 'train-samples', total_samples, 100, 10)
        log(train_writer, updates_i, 'train-bellman-error', bellman_error, 100, 10)
        log(train_writer, updates_i, 'train-epsilon', epsilon, 100, 10)
        log(train_writer, updates_i, None, None, 100, 10)

        if num_steps >= reset_every:
            states = [e.reset() for e in envs]
            num_steps = 0
        
        cross_boundary = total_samples // val_every > (total_samples - len(envs) * num_steps_per_rollout) // val_every
        if cross_boundary:
            models[0].eval()
            mean_reward = val_fn(models[0], device)
            log(val_writer, total_samples, 'val-mean_reward', mean_reward, 1, 1)
            log(val_writer, total_samples, None, None, 1, 1)
            models[0].train()
