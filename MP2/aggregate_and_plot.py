from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from absl import flags, app
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
FLAGS = flags.FLAGS

flags.DEFINE_spaceseplist('logdirs', [], 
    'Space separated list of directories to plot results from.')
flags.DEFINE_string('output_file_name', 'out.pdf', 
    'Output file to generate plot.')
flags.DEFINE_integer('seeds', 5,
    'Number of seeds per run')

def process_event_file(logdir, seed):
    """Process a single event file and return steps and values."""
    logdir_ = Path(logdir) / f'seed{seed}' / 'val'
    print(f"Processing: {logdir_}")
    
    event_acc = EventAccumulator(str(logdir_), size_guidance={'scalars': 0})
    try:
        event_acc.Reload()
        if 'val-mean_reward' not in event_acc.Tags()['scalars']:
            print(f"No val-mean_reward found in seed {seed}")
            return None, None
            
        scalar_events = event_acc.Scalars('val-mean_reward')
        if not scalar_events:
            print(f"No scalar events found in seed {seed}")
            return None, None
            
        steps = [event.step for event in scalar_events]
        vals = [event.value for event in scalar_events]
        print(f"Found {len(steps)} events for seed {seed}")
        return steps, vals
    except Exception as e:
        print(f"Error processing seed {seed}: {str(e)}")
        return None, None

def main(_):
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['blue', 'red']  # Different colors for different runs
    for i, logdir in enumerate(FLAGS.logdirs):
        print(f"\nProcessing logdir: {logdir}")
        samples = []
        rewards = []
        
        # Process each seed
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
            
        print(f"Successfully processed {valid_seeds} seeds")
        samples = np.array(samples)
        rewards = np.array(rewards)
        
        # Verify all seeds have same steps
        if not np.all(samples == samples[0,:]):
            print(f"Warning: Seeds have different step values in {logdir}")
            continue
            
        mean_rewards = np.mean(rewards, 0)
        std_rewards = np.std(rewards, 0)
        
        label = f"Run {i+1} (mean of {valid_seeds} seeds)"
        ax.plot(samples[0,:], mean_rewards, linewidth=2, 
               label=label, color=colors[i])
        ax.fill_between(samples[0,:], 
                       mean_rewards-std_rewards, 
                       mean_rewards+std_rewards, 
                       alpha=0.3, color=colors[i])

    # Improve plot appearance
    ax.set_title('DQN Training Progress', fontsize=14, pad=20)
    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_ylabel('Average Return', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 210])
    ax.grid(True, alpha=0.3)
    
    # Add target line
    ax.axhline(y=195, color='green', linestyle='--', alpha=0.5, label='Target (195)')
    
    # Tight layout
    plt.tight_layout()
    
    # Save with high DPI
    output_path = os.path.join(os.getcwd(), FLAGS.output_file_name)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")

if __name__ == '__main__':
    app.run(main)

# Test one directory
test_dir = "LOGDIR/seed0/val"
event_acc = EventAccumulator(test_dir)
event_acc.Reload()

# Print available tags
print("Available tags:", event_acc.Tags())
