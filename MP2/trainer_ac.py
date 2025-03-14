from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn.functional as F

# Key hyperparameters
num_steps_per_rollout = 5  # Shorter rollouts
gamma = 0.99  # Standard discount factor
learning_rate = 3e-4  # Standard learning rate for Adam
value_loss_coef = 0.5  # Weight for value loss
max_grad_norm = 0.5  # Gradient clipping threshold
entropy_coef = 0.01  # Entropy bonus for exploration

num_updates = 10000
reset_every = 200
val_every = 10000

def log(writer, iteration, name, value, print_every=10, log_every=10):
    # A simple function to let you log progress to the console and tensorboard.
    if np.mod(iteration, print_every) == 0:
        if name is not None:
            print('{:8d}{:>30s}: {:0.3f}'.format(iteration, name, value))
        else:
            print('')
    if name is not None and np.mod(iteration, log_every) == 0:
        writer.add_scalar(name, value, iteration)

# Starting off from states in envs, rolls out num_steps_per_rollout for each
# environment using the policy in `model`. Returns rollouts in the form of
# states, actions, rewards and new states. Also returns the state the
# environments end up in after num_steps_per_rollout time steps.
def collect_rollouts(model, envs, states, num_steps, device):
    rollouts = []
    
    for _ in range(num_steps):
        # Convert states to tensor
        state_tensor = torch.FloatTensor(np.array(states, dtype=np.float32)).to(device)
        
        # Get actions from the policy
        with torch.no_grad():
            actions = model.act(state_tensor, sample=True)
            actor_logits, values = model(state_tensor)
        
        # Get log probabilities for the actions
        action_dist = model.actor_to_distribution(actor_logits)
        log_probs = action_dist.log_prob(actions)
        
        # Convert actions to numpy for environment stepping
        actions_np = actions.cpu().numpy()
        
        # Step environments
        next_states = []
        rewards = []
        dones = []
        for env, action in zip(envs, actions_np):
            next_state, reward, done, _ = env.step(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
        
        # Store transition
        rollouts.append({
            'states': state_tensor,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': torch.FloatTensor(rewards).to(device),
            'values': values,
            'next_states': torch.FloatTensor(np.array(next_states, dtype=np.float32)).to(device),
            'dones': torch.FloatTensor(dones).to(device)
        })
        
        # Reset environments that are done
        states = []
        for i, (next_state, done) in enumerate(zip(next_states, dones)):
            if done:
                state = envs[i].reset()
            else:
                state = next_state
            states.append(state)
    
    return rollouts, states
        
# Using the rollouts returned by collect_rollouts function, updates the actor
# and critic models. You will need to:
# 1a. Compute targets for the critic using the current critic in model.
# 1b. Compute loss for the critic, and optimize it.
# 2a. Compute returns, or estimate for returns, or advantages for updating the actor.
# 2b. Set up the appropriate loss function for actor, and optimize it.
# Function can return actor and critic loss, for plotting.
def update_model(model, gamma, optim, rollouts, device, iteration, writer):
    gae_lambda = 0.95
    returns = []
    advantages = []
    gae = 0
    
    with torch.no_grad():
        _, next_value = model(rollouts[-1]['next_states'])
    next_value = next_value.squeeze(-1)
    
    # Calculate returns and advantages
    for t in reversed(range(len(rollouts))):
        if t == len(rollouts) - 1:
            next_non_terminal = 1.0 - rollouts[t]['dones']
            next_values = next_value
        else:
            next_non_terminal = 1.0 - rollouts[t+1]['dones']
            next_values = rollouts[t+1]['values'].squeeze(-1)
        
        delta = rollouts[t]['rewards'] + gamma * next_values * next_non_terminal - rollouts[t]['values'].squeeze(-1)
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        
        returns.insert(0, gae + rollouts[t]['values'].squeeze(-1))
        advantages.insert(0, gae)
    
    returns = torch.stack(returns)
    advantages = torch.stack(advantages)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Prepare batched data
    states = torch.cat([r['states'] for r in rollouts])
    actions = torch.cat([r['actions'] for r in rollouts])
    old_log_probs = torch.cat([r['log_probs'] for r in rollouts])
    old_log_probs = old_log_probs.sum(dim=-1)
    
    batch_size = states.size(0)
    actions = actions.reshape(batch_size)
    returns = returns.reshape(batch_size)
    advantages = advantages.reshape(batch_size)
    
    # Get current policy distribution and values
    actor_logits, values = model(states)
    dist = model.actor_to_distribution(actor_logits)
    new_log_probs = dist.log_prob(actions)
    
    # Policy loss (simple policy gradient)
    policy_loss = -(new_log_probs * advantages).mean()
    
    # Value loss
    value_pred = values.squeeze(-1)
    value_loss = F.mse_loss(value_pred, returns)
    
    # Entropy bonus
    entropy_loss = -entropy_coef * dist.entropy().mean()
    
    # Total loss
    total_loss = policy_loss + value_loss_coef * value_loss + entropy_loss
    
    # Optimize
    optim.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optim.step()
    
    return policy_loss.item(), value_loss.item()


def train_model_ac(model, envs, gamma, device, logdir, val_fn):
    model.to(device)
    train_writer = SummaryWriter(logdir / 'train')
    val_writer = SummaryWriter(logdir / 'val')
    
    # Setup optimizer
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Resetting all environments to initialize the state.
    num_steps, total_samples = 0, 0
    states = [e.reset() for e in envs]
    
    for updates_i in range(num_updates):
        
        # Put model in training mode.
        model.train()
        
        # Collect rollouts using the policy.
        rollouts, states = collect_rollouts(model, envs, states, num_steps_per_rollout, device)
        num_steps += num_steps_per_rollout
        total_samples += num_steps_per_rollout * len(envs)

        # Use rollouts to update the policy and take gradient steps.
        actor_loss, critic_loss = update_model(model, gamma, optim, rollouts, device, updates_i, train_writer)
        log(train_writer, updates_i, 'train-samples', total_samples, 100, 10)
        log(train_writer, updates_i, 'train-actor_loss', actor_loss, 100, 10)
        log(train_writer, updates_i, 'train-critic_loss', critic_loss, 100, 10)
        log(train_writer, updates_i, None, None, 100, 10)

        # Reset environments periodically
        if num_steps >= reset_every:
            states = [e.reset() for e in envs]
            num_steps = 0
        
        # Validation
        cross_boundary = total_samples // val_every > \
            (total_samples - len(envs) * num_steps_per_rollout) // val_every
        if cross_boundary:
            model.eval()
            mean_reward = val_fn(model, device)
            log(val_writer, total_samples, 'val-mean_reward', mean_reward, 1, 1)
            log(val_writer, total_samples, None, None, 1, 1)
            model.train()
