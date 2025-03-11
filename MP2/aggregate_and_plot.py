from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from absl import flags, app
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
FLAGS = flags.FLAGS

flags.DEFINE_spaceseplist('logdirs', [], 
    'Space separated list of directories to plot results from.')
flags.DEFINE_string('output_file_name', 'out.pdf', 
    'Output file to generate plot.')
flags.DEFINE_integer('seeds', 5,
    'Number of seeds per run')

def get_latest_event_file(val_dir):
    """Get the most recent event file in the directory."""
    event_files = glob.glob(os.path.join(val_dir, 'events.out.tfevents.*'))
    if not event_files:
        return None
    # Sort by modification time and get the latest
    return max(event_files, key=os.path.getmtime)

def process_event_file(logdir, seed):
    """Process a single event file and return steps and values."""
    val_dir = os.path.join(logdir, f'seed{seed}', 'val')
    print(f"\nProcessing directory: {val_dir}")
    
    # Get the latest event file
    event_file = get_latest_event_file(val_dir)
    if event_file is None:
        print("No event files found")
        return None, None
    
    print(f"Using event file: {os.path.basename(event_file)}")
    
    try:
        event_acc = EventAccumulator(val_dir)
        event_acc.Reload()
        
        # Print available tags for debugging
        tags = event_acc.Tags()
        print(f"Available tags: {tags}")
        
        if 'scalars' not in tags or 'val-mean_reward' not in tags['scalars']:
            print("val-mean_reward not found in tags")
            return None, None
        
        scalar_events = event_acc.Scalars('val-mean_reward')
        if not scalar_events:
            print("No scalar events found")
            return None, None
        
        steps = [event.step for event in scalar_events]
        vals = [event.value for event in scalar_events]
        
        print(f"Found {len(steps)} events")
        print(f"Steps range: [{min(steps)}, {max(steps)}]")
        print(f"Values range: [{min(vals):.2f}, {max(vals):.2f}]")
        
        return steps, vals
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None, None

def main(_):
    print(f"Current working directory: {os.getcwd()}")
    print(f"Processing directories: {FLAGS.logdirs}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, logdir in enumerate(FLAGS.logdirs):
        print(f"\nProcessing logdir {i+1}: {logdir}")
        if not os.path.exists(logdir):
            print(f"Directory does not exist: {logdir}")
            continue
            
        samples = []
        rewards = []
        valid_seeds = 0
        
        for seed in range(FLAGS.seeds):
            steps, vals = process_event_file(logdir, seed)
            if steps is not None and vals is not None:
                samples.append(steps)
                rewards.append(vals)
                valid_seeds += 1
        
        if valid_seeds == 0:
            print(f"No valid data found in {logdir}")
            continue
        
        samples = np.array(samples)
        rewards = np.array(rewards)
        
        mean_rewards = np.mean(rewards, 0)
        std_rewards = np.std(rewards, 0)
        
        label = f"Run {i+1} ({valid_seeds} seeds)"
        ax.plot(samples[0,:], mean_rewards, linewidth=2, label=label)
        ax.fill_between(samples[0,:], 
                       mean_rewards-std_rewards, 
                       mean_rewards+std_rewards, 
                       alpha=0.3)
    
    ax.set_title('DQN Training Progress')
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Average Return')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 210])
    ax.grid(True)
    ax.axhline(y=195, color='r', linestyle='--', label='Target')
    
    plt.tight_layout()
    fig.savefig(FLAGS.output_file_name, bbox_inches='tight')
    print(f"\nPlot saved to: {FLAGS.output_file_name}")

if __name__ == '__main__':
    app.run(main)

# Test one directory
test_dir = "LOGDIR/seed0/val"
event_acc = EventAccumulator(test_dir)
event_acc.Reload()

# Print available tags
print("Available tags:", event_acc.Tags())
