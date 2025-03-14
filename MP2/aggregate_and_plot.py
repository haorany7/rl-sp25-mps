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
flags.DEFINE_string('output_file_name', 'out.png', 
    'Output file to generate plot.')
flags.DEFINE_integer('seeds', 5,
    'Number of seeds per run')
flags.DEFINE_spaceseplist('curve_names', None, 'Space separated list of names for the curves in the plot')

def get_latest_event_file(val_dir):
    """Get the most recent event file in the directory."""
    event_files = glob.glob(os.path.join(val_dir, 'events.out.tfevents.*'))
    if not event_files:
        return None
    # Sort by modification time and get the latest
    return max(event_files, key=os.path.getmtime)

def process_event_file(event_dir, data_type):
    """Process a TensorBoard event file."""
    try:
        # Get all event files and sort them by timestamp (newest first)
        event_files = glob.glob(os.path.join(event_dir, "events.out.tfevents.*"))
        if not event_files:
            print(f"No event files found in {event_dir}")
            return None, None
            
        # Sort event files by modification time (newest first)
        event_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        event_file = event_files[0]  # Take the newest file
        
        # Check file existence and permissions
        if not os.path.exists(event_file):
            print(f"File does not exist: {event_file}")
            return None, None
            
        print(f"\nProcessing {data_type} file: {event_file}")
        print(f"File size: {os.path.getsize(event_file)} bytes")
        print(f"File permissions: {oct(os.stat(event_file).st_mode)[-3:]}")
        
        # Create EventAccumulator with size guidance
        event_acc = EventAccumulator(
            event_file,
            size_guidance={
                'scalars': 0,  # 0 means load all
                'tensors': 0,
            }
        )
        
        # Load the data
        print("Loading event file...")
        event_acc.Reload()
        print("Event file loaded.")
        
        tags = event_acc.Tags()
        print(f"Available tags: {tags}")
        
        # Print all available scalar tags for debugging
        print(f"\nAvailable scalar tags in {event_dir}:")
        for tag in tags['scalars']:
            print(f"  - {tag}")
            # Print first few values for this tag
            values = event_acc.Scalars(tag)
            print(f"    First few values: {values[:5]}")
        
        # Process based on data type
        if data_type == 'train':
            if 'train-samples' not in tags['scalars'] or 'train-reward' not in tags['scalars']:
                print("Required training tags not found")
                print(f"Available tags: {tags['scalars']}")
                return None, None
                
            samples = [s.step for s in event_acc.Scalars('train-samples')]
            rewards = [r.value for r in event_acc.Scalars('train-reward')]
            print(f"Loaded {len(samples)} training samples")
            print(f"Reward range: [{min(rewards)}, {max(rewards)}]")
            return np.array(samples), np.array(rewards)
            
        elif data_type == 'val':
            if 'val-mean_reward' not in tags['scalars']:
                print("Required validation tags not found")
                return None, None
            val_data = event_acc.Scalars('val-mean_reward')
            samples = [s.step for s in val_data]
            rewards = [r.value for r in val_data]
            print(f"Loaded {len(samples)} validation samples")
            return np.array(samples), np.array(rewards)
            
        print(f"Unknown data type: {data_type}")
        return None, None
        
    except Exception as e:
        print(f"Error processing {event_dir}: {str(e)}")
        import traceback
        traceback.print_exc()
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
            train_path = os.path.join(logdir, f'seed{seed}', 'train')
            steps, vals = process_event_file(train_path, 'train')
            
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
        
        curve_name = FLAGS.curve_names[i] if FLAGS.curve_names and i < len(FLAGS.curve_names) else f"Run {i+1}"
        label = f"{curve_name} ({len(samples)} seeds)"
        ax.plot(samples[0,:], mean_rewards, linewidth=2, label=label)
        ax.fill_between(samples[0,:], 
                       mean_rewards-std_rewards, 
                       mean_rewards+std_rewards, 
                       alpha=0.3)
    
    ax.set_title('DQN Variants Training Progress')
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

# # Test one directory
# test_dir = "LOGDIR/seed0/val"
# event_acc = EventAccumulator(test_dir)
# event_acc.Reload()

# # Print available tags
# print("Available tags:", event_acc.Tags())