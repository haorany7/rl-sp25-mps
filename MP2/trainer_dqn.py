import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.optim as optim

num_steps_per_rollout = 5 
num_updates = 10000
reset_every = 200
val_every = 10000

replay_buffer_size = 1000000
q_target_update_every = 50
q_batch_size = 256
q_num_steps = 1

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
        self.size = size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer = {
            'states': np.zeros((size, state_dim)),
            'actions': np.zeros((size, action_dim)),
            'rewards': np.zeros(size),
            'next_states': np.zeros((size, state_dim)),
            'dones': np.zeros(size)
        }
        self.index = 0
        self.full = False

    def insert(self, rollouts):
        for state, action, reward, next_state, done in rollouts:
            self.buffer['states'][self.index] = state
            self.buffer['actions'][self.index] = action
            self.buffer['rewards'][self.index] = reward
            self.buffer['next_states'][self.index] = next_state
            self.buffer['dones'][self.index] = done
            self.index = (self.index + 1) % self.size
            if self.index == 0:
                self.full = True

    def sample_batch(self, batch_size):
        max_index = self.size  # Use the current size of the buffer
        actual_batch_size = min(batch_size, max_index)  # Use the smaller of the two
        indices = np.random.choice(max_index, actual_batch_size, replace=False)
        return {key: self.buffer[key][indices] for key in self.buffer}

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
                    q_values = model(state_tensor)
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
def update_model(replay_buffer, models, targets, optim, gamma, action_dim,
                 q_batch_size, q_num_steps, device):
    total_bellman_error = 0.
    for _ in range(q_num_steps):
        batch = replay_buffer.sample_batch(q_batch_size)
        states = torch.tensor(batch['states'], dtype=torch.float32).to(device)
        # Ensure actions is a 1D tensor and then unsqueeze for gather
        actions = torch.tensor(batch['actions'], dtype=torch.int64).to(device).flatten()
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32).to(device)
        next_states = torch.tensor(batch['next_states'], dtype=torch.float32).to(device)
        dones = torch.tensor(batch['dones'], dtype=torch.float32).to(device)

        # Get Q-values and gather using actions
        q_values = models[0](states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = targets[0](next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * gamma * next_q_values

        loss = torch.nn.functional.mse_loss(q_values, target_q_values)
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_bellman_error += loss.item()
    return total_bellman_error / q_num_steps

def train_model_dqn(models, targets, state_dim, action_dim, envs, gamma, device, logdir, val_fn):
    train_writer = SummaryWriter(logdir / 'train')
    val_writer = SummaryWriter(logdir / 'val')
    
    # Set up the optimizer for the models
    optimizers = [optim.Adam(model.parameters(), lr=1e-3) for model in models]

    # Set up the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, state_dim, 1)

    # Resetting all environments to initialize the state.
    num_steps, total_samples = 0, 0
    states = [e.reset() for e in envs]
    
    # Define epsilon schedule parameters
    epsilon_start = 1.0  # Start with full exploration
    epsilon_end = 0.1    # End with limited exploration
    epsilon_decay = 0.995  # Decay rate per update

    for updates_i in range(num_updates):
        # Update epsilon based on the schedule
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** updates_i))

        # Put model in training mode.
        [m.train() for m in models]

        if np.mod(updates_i, q_target_update_every) == 0:
            # Update target networks to match the current models
            for model, target in zip(models, targets):
                target.load_state_dict(model.state_dict())
        
        # Collect rollouts using the policy.
        rollouts, states = collect_rollouts(models, envs, states, num_steps_per_rollout, epsilon, device)
        num_steps += num_steps_per_rollout
        total_samples += num_steps_per_rollout * len(envs)
        
        # Push rollouts into the replay buffer.
        replay_buffer.insert(rollouts)

        # Use replay buffer to update the policy and take gradient steps.
        for model, target, optimizer in zip(models, targets, optimizers):
            bellman_error = update_model(replay_buffer, [model], [target], optimizer,
                                         gamma, action_dim, q_batch_size,
                                         q_num_steps, device)
            log(train_writer, updates_i, 'train-samples', total_samples, 100, 10)
            log(train_writer, updates_i, 'train-bellman-error', bellman_error, 100, 10)
            log(train_writer, updates_i, 'train-epsilon', epsilon, 100, 10)
            log(train_writer, updates_i, None, None, 100, 10)

        # We are solving a continuing MDP which never returns a done signal. We
        # are going to manually reset the environment every few time steps. To
        # track progress on the training environments you can maintain the
        # returns on the training environments, and log or print it out when
        # you reset the environments.
        if num_steps >= reset_every:
            states = [e.reset() for e in envs]
            num_steps = 0
        
        # Every once in a while run the policy on the environment in the
        # validation set. We will use this to plot the learning curve as a
        # function of the number of samples.
        cross_boundary = total_samples // val_every > \
            (total_samples - len(envs) * num_steps_per_rollout) // val_every
        if cross_boundary:
            models[0].eval()
            mean_reward = val_fn(models[0], device)
            log(val_writer, total_samples, 'val-mean_reward', mean_reward, 1, 1)
            log(val_writer, total_samples, None, None, 1, 1)
            models[0].train()
