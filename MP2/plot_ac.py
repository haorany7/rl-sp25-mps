from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from absl import flags, app
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

FLAGS = flags.FLAGS
flags.DEFINE_string('logdir', 'LOGDIR_AC', 'Directory containing the AC logs')
flags.DEFINE_integer('seeds', 5, 'Number of seeds')
flags.DEFINE_string('output_prefix', 'ac', 'Prefix for output files')

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
    
    train_actor_losses = []
    train_critic_losses = []
    val_rewards = []
    
    for seed in range(FLAGS.seeds):
        # Process training data
        train_path = os.path.join(FLAGS.logdir, f'seed{seed}', 'train')
        train_data = process_event_file(train_path)
        
        if train_data and 'train-actor_loss' in train_data and 'train-critic_loss' in train_data:
            steps, actor_loss = train_data['train-actor_loss']
            _, critic_loss = train_data['train-critic_loss']
            train_actor_losses.append((steps, actor_loss))
            train_critic_losses.append((steps, critic_loss))
            
        # Process validation data
        val_path = os.path.join(FLAGS.logdir, f'seed{seed}', 'val')
        val_data = process_event_file(val_path)
        
        if val_data and 'val-mean_reward' in val_data:
            steps, rewards = val_data['val-mean_reward']
            val_rewards.append((steps, rewards))
    
    # Plot training losses
    if train_actor_losses:
        for steps, loss in train_actor_losses:
            ax1.plot(steps, loss, alpha=0.3)
        mean_steps = train_actor_losses[0][0]
        mean_loss = np.mean([loss for _, loss in train_actor_losses], axis=0)
        std_loss = np.std([loss for _, loss in train_actor_losses], axis=0)
        ax1.plot(mean_steps, mean_loss, 'b-', label='Mean Actor Loss')
        ax1.fill_between(mean_steps, mean_loss-std_loss, mean_loss+std_loss, alpha=0.2)
        ax1.set_title('Actor Loss During Training')
        ax1.set_xlabel('Environment Steps')
        ax1.set_ylabel('Actor Loss')
        ax1.grid(True)
        ax1.legend()

    if train_critic_losses:
        for steps, loss in train_critic_losses:
            ax2.plot(steps, loss, alpha=0.3)
        mean_steps = train_critic_losses[0][0]
        mean_loss = np.mean([loss for _, loss in train_critic_losses], axis=0)
        std_loss = np.std([loss for _, loss in train_critic_losses], axis=0)
        ax2.plot(mean_steps, mean_loss, 'r-', label='Mean Critic Loss')
        ax2.fill_between(mean_steps, mean_loss-std_loss, mean_loss+std_loss, alpha=0.2)
        ax2.set_title('Critic Loss During Training')
        ax2.set_xlabel('Environment Steps')
        ax2.set_ylabel('Critic Loss')
        ax2.grid(True)
        ax2.legend()
    
    # Plot validation rewards
    if val_rewards:
        for steps, reward in val_rewards:
            ax3.plot(steps, reward, alpha=0.3)
        mean_steps = val_rewards[0][0]
        mean_reward = np.mean([reward for _, reward in val_rewards], axis=0)
        std_reward = np.std([reward for _, reward in val_rewards], axis=0)
        ax3.plot(mean_steps, mean_reward, 'g-', label='Mean Validation Reward')
        ax3.fill_between(mean_steps, mean_reward-std_reward, mean_reward+std_reward, alpha=0.2)
        ax3.set_title('Validation Rewards')
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