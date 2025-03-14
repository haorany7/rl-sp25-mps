from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from absl import flags, app
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

FLAGS = flags.FLAGS
flags.DEFINE_string('logdir', 'LOGDIR_QAC', 'Directory containing the QAC logs')
flags.DEFINE_integer('seeds', 5, 'Number of seeds')
flags.DEFINE_string('output_prefix', 'qac', 'Prefix for output files')

def process_event_file(event_dir):
    """Process a TensorBoard event file."""
    try:
        event_files = glob.glob(os.path.join(event_dir, "events.out.tfevents.*"))
        if not event_files:
            print(f"No event files found in {event_dir}")
            return None
            
        event_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        event_file = event_files[0]
        
        if not os.path.exists(event_file):
            print(f"File does not exist: {event_file}")
            return None
            
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()
        
        data = {}
        tags = event_acc.Tags()['scalars']
        
        for tag in tags:
            values = event_acc.Scalars(tag)
            steps = [s.step for s in values]
            vals = [v.value for v in values]
            data[tag] = (np.array(steps), np.array(vals))
            
        return data
        
    except Exception as e:
        print(f"Error processing {event_dir}: {str(e)}")
        return None

def main(_):
    # Create figures
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig2, ax3 = plt.subplots(figsize=(10, 6))
    
    train_policy_losses = []
    train_q_losses = []
    eval_rewards = []
    
    for seed in range(FLAGS.seeds):
        # Process training data
        train_path = os.path.join(FLAGS.logdir, f'seed{seed}', 'train')
        train_data = process_event_file(train_path)
        
        # Process validation data
        val_path = os.path.join(FLAGS.logdir, f'seed{seed}', 'val')
        val_data = process_event_file(val_path)
        
        if train_data:
            if 'train-actor_loss' in train_data and 'train-q_loss' in train_data:
                steps, policy_loss = train_data['train-actor_loss']
                _, q_loss = train_data['train-q_loss']
                train_policy_losses.append((steps, policy_loss))
                train_q_losses.append((steps, q_loss))
        
        if val_data:
            if 'val-mean_reward' in val_data:
                steps, rewards = val_data['val-mean_reward']
                eval_rewards.append((steps, rewards))
    
    # Plot training losses
    if train_policy_losses:
        for i, (steps, loss) in enumerate(train_policy_losses):
            ax1.plot(steps, loss, alpha=0.3, label=f'Seed {i} Policy Loss')
        mean_steps = train_policy_losses[0][0]
        mean_loss = np.mean([loss for _, loss in train_policy_losses], axis=0)
        std_loss = np.std([loss for _, loss in train_policy_losses], axis=0)
        ax1.plot(mean_steps, mean_loss, 'b-', linewidth=2, label='Mean Policy Loss')
        ax1.fill_between(mean_steps, mean_loss-std_loss, mean_loss+std_loss, alpha=0.2)
        ax1.set_title('Policy Loss During Training')
        ax1.set_xlabel('Environment Steps')
        ax1.set_ylabel('Policy Loss')
        ax1.grid(True)
        ax1.legend()

    if train_q_losses:
        for i, (steps, loss) in enumerate(train_q_losses):
            ax2.plot(steps, loss, alpha=0.3, label=f'Seed {i} Q Loss')
        mean_steps = train_q_losses[0][0]
        mean_loss = np.mean([loss for _, loss in train_q_losses], axis=0)
        std_loss = np.std([loss for _, loss in train_q_losses], axis=0)
        ax2.plot(mean_steps, mean_loss, 'r-', linewidth=2, label='Mean Q Loss')
        ax2.fill_between(mean_steps, mean_loss-std_loss, mean_loss+std_loss, alpha=0.2)
        ax2.set_title('Q Loss During Training')
        ax2.set_xlabel('Environment Steps')
        ax2.set_ylabel('Q Loss')
        ax2.grid(True)
        ax2.legend()
    
    # Plot evaluation rewards
    if eval_rewards:
        for i, (steps, reward) in enumerate(eval_rewards):
            ax3.plot(steps, reward, alpha=0.3, label=f'Seed {i} Reward')
        mean_steps = eval_rewards[0][0]
        mean_reward = np.mean([reward for _, reward in eval_rewards], axis=0)
        std_reward = np.std([reward for _, reward in eval_rewards], axis=0)
        ax3.plot(mean_steps, mean_reward, 'g-', linewidth=2, label='Mean Evaluation Reward')
        ax3.fill_between(mean_steps, mean_reward-std_reward, mean_reward+std_reward, alpha=0.2)
        ax3.set_title('Evaluation Rewards')
        ax3.set_xlabel('Environment Steps')
        ax3.set_ylabel('Average Return')
        ax3.grid(True)
        ax3.legend()
        ax3.set_ylim([0, 210])
        ax3.axhline(y=195, color='r', linestyle='--', label='Target')
    
    # Save plots
    plt.tight_layout()
    fig1.savefig(f'{FLAGS.output_prefix}_losses.png', bbox_inches='tight')
    fig2.savefig(f'{FLAGS.output_prefix}_rewards.png', bbox_inches='tight')
    print(f"\nPlots saved as {FLAGS.output_prefix}_losses.png and {FLAGS.output_prefix}_rewards.png")

if __name__ == '__main__':
    app.run(main) 