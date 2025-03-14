from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn.functional as F

# Key hyperparameters
num_steps_per_rollout = 10
gamma = 0.99
learning_rate = 3e-4
q_loss_coef = 0.5
max_grad_norm = 0.5
entropy_coef = 0.1  # Increased for more exploration
temperature = 1.0   # Temperature for softmax

num_updates = 10000
reset_every = 200
val_every = 10000

# Modify reward scaling
reward_scale = 0.1  # Adjust this scale factor as needed

def log(writer, iteration, name, value, print_every=10, log_every=10):
    if np.mod(iteration, print_every) == 0:
        if name is not None:
            print('{:8d}{:>30s}: {:0.3f}'.format(iteration, name, value))
        else:
            print('')
    if name is not None and np.mod(iteration, log_every) == 0:
        writer.add_scalar(name, value, iteration)

def collect_rollouts(model, envs, states, num_steps, device):
    rollouts = []
    
    for _ in range(num_steps):
        state_tensor = torch.FloatTensor(np.array(states, dtype=np.float32)).to(device)
        
        # Get actions from the policy
        with torch.no_grad():
            actions = model.act(state_tensor, sample=True)
            actor_logits, q_values = model(state_tensor)
        
        # Get log probabilities for the actions
        action_dist = model.actor_to_distribution(actor_logits)
        log_probs = action_dist.log_prob(actions)
        
        # Get Q-values for taken actions
        q_values_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Ensure actions are properly squeezed before converting to numpy
        actions_np = actions.squeeze().cpu().numpy()
        
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
            'actions': actions.squeeze(),  # Squeeze actions
            'log_probs': log_probs,
            'rewards': torch.FloatTensor(rewards).to(device) * reward_scale,
            'q_values': q_values_taken,
            'next_states': torch.FloatTensor(np.array(next_states, dtype=np.float32)).to(device),
            'dones': torch.FloatTensor(dones).to(device)
        })
        
        states = []
        for i, (next_state, done) in enumerate(zip(next_states, dones)):
            if done:
                state = envs[i].reset()
            else:
                state = next_state
            states.append(state)
    
    return rollouts, states

def update_model(model, gamma, optim, rollouts, device, iteration, writer):
    returns = []
    
    # Calculate returns
    with torch.no_grad():
        for t in reversed(range(len(rollouts))):
            if t == len(rollouts) - 1:
                next_non_terminal = 1.0 - rollouts[t]['dones']
                _, next_q_values = model(rollouts[t]['next_states'])
                next_value = next_q_values.max(dim=1)[0]
            else:
                next_non_terminal = 1.0 - rollouts[t+1]['dones']
                _, next_q_values = model(rollouts[t+1]['states'])
                next_value = next_q_values.max(dim=1)[0]
            
            rewards = rollouts[t]['rewards']
            returns.insert(0, rewards + gamma * next_value * next_non_terminal)
    
    returns = torch.cat(returns)
    
    # Prepare batched data
    states = torch.cat([r['states'] for r in rollouts])
    actions = torch.cat([r['actions'] for r in rollouts]).squeeze()
    q_values = torch.cat([r['q_values'] for r in rollouts])
    
    # Get current policy distribution and Q-values
    actor_logits, new_q_values = model(states)
    
    # Get action distribution
    dist = model.actor_to_distribution(actor_logits)
    log_probs = dist.log_prob(actions)
    
    # Policy loss using Q-values directly
    policy_loss = -(log_probs * q_values.detach()).mean() * 5.0
    
    # Q-value loss
    q_loss = F.smooth_l1_loss(q_values, returns.detach())
    
    # Entropy bonus with fixed coefficient
    entropy = dist.entropy().mean()
    entropy_loss = -0.1 * entropy
    
    # Total loss
    total_loss = policy_loss + q_loss_coef * q_loss + entropy_loss
    
    # Print debugging info
    action_probs = torch.softmax(actor_logits, dim=-1)
    # print(f"\nIteration {iteration}")
    # print(f"Policy Loss: {policy_loss.item()/5.0:.6f}")  # Show unscaled loss
    # print(f"Q Loss: {q_loss.item():.6f}")
    # print(f"Entropy: {entropy.item():.6f}")
    # print(f"Action Probs: {action_probs.mean(0)}")
    # print(f"Q-values - Mean: {q_values.mean():.6f}, Std: {q_values.std():.6f}")
    
    # Optimize
    optim.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optim.step()
    
    return policy_loss.item() / 5.0, q_loss.item()

def train_model_qac(model, envs, gamma, device, logdir, val_fn):
    model.to(device)
    train_writer = SummaryWriter(logdir / 'train')
    val_writer = SummaryWriter(logdir / 'val')
    
    # Setup optimizer
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize state
    num_steps, total_samples = 0, 0
    states = [e.reset() for e in envs]
    
    for updates_i in range(num_updates):
        model.train()
        
        # Collect rollouts
        rollouts, states = collect_rollouts(model, envs, states, num_steps_per_rollout, device)
        num_steps += num_steps_per_rollout
        total_samples += num_steps_per_rollout * len(envs)

        # Update model
        actor_loss, q_loss = update_model(model, gamma, optim, rollouts, device, updates_i, train_writer)
        log(train_writer, updates_i, 'train-samples', total_samples, 100, 10)
        log(train_writer, updates_i, 'train-actor_loss', actor_loss, 100, 10)
        log(train_writer, updates_i, 'train-q_loss', q_loss, 100, 10)
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