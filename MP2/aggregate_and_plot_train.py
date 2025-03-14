from pathlib import Path
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from absl import flags, app
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
FLAGS = flags.FLAGS

flags.DEFINE_spaceseplist('logdirs', [], 
    'Space separated list of directories to plot results from.')
flags.DEFINE_string('output_file_name', 'out.png',
    'Output file to generate plot.')
flags.DEFINE_integer('seeds', 5,
    'Number of seeds per run')
flags.DEFINE_spaceseplist('curve_names', None,
    'Names for the curves in the plot')

def main(_):
    sns.color_palette()
    fig = plt.figure(figsize=(10,6))  # Made figure slightly larger
    ax = fig.gca()
    print(FLAGS.logdirs)
    
    for i, logdir in enumerate(FLAGS.logdirs):
        print(logdir)
        samples = []
        bellman_errors = []
        for seed in range(FLAGS.seeds):
            logdir_ = Path(logdir) / f'seed{seed}'
            logdir_ = logdir_ / 'train'
            event_acc = EventAccumulator(str(logdir_))
            event_acc.Reload()
            
            # Get training samples and Bellman error
            try:
                sample_events = event_acc.Scalars('train-samples')
                error_events = event_acc.Scalars('train-bellman-error')
                
                step_nums = [event.step * 50 for event in sample_events]  # Scale steps by 50
                errors = [event.value for event in error_events]
                
                samples.append(step_nums)
                bellman_errors.append(errors)
                print(f"Successfully loaded data from {logdir_}")
                
            except Exception as e:
                print(f"Error processing {logdir_}: {str(e)}")
                continue
            
        if not samples or not bellman_errors:
            print(f"No valid data found for {logdir}")
            continue
            
        samples = np.array(samples)
        bellman_errors = np.array(bellman_errors)
        mean_errors = np.mean(bellman_errors, 0)
        std_errors = np.std(bellman_errors, 0)
        
        # Use custom curve name if provided
        label = FLAGS.curve_names[i] if FLAGS.curve_names else logdir
        ax.plot(samples[0,:], mean_errors, label=label)
        ax.fill_between(samples[0,:], 
                       mean_errors-std_errors, mean_errors+std_errors, alpha=0.2)

    ax.set_title('DQN Variants Training Loss')
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Bellman Error')
    ax.legend(loc='upper right')
    ax.grid('major')
    
    # Set x-axis limit to 400k steps
    ax.set_xlim([0, 400000])
    
    fig.savefig(FLAGS.output_file_name, format='png', bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    app.run(main)