# ------------------------------------------------------ 1 ----------------------------------------------------
#------------------------------------------------Importing Libraries-------------------------------------------


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os



# ------------------------------------------------------ 2 ----------------------------------------------------
#----------------------Reading and Plotting the performance (Cumulative Reward) for each method-------------------

def plot_multiple_files(file_paths, title, x_label, y_label, legend_labels):
    plt.figure(figsize=(18, 9))
    
    num_files = len(file_paths)
    color_sets = [['#B0E0E6', '#4169E1', '#000080'], 
                  ['#FFA07A', '#FF0000', '#8B0000'], 
                  ['#98FB98', '#00FF00', '#008000']]
    
    colors = []
    for i in range(num_files):
        colors.append(color_sets[i//3][i%3])
    
    for file_path, label, color in zip(file_paths, legend_labels, colors):
        data = read_list_from_file(file_path)
        cumulative_avg = [sum(data[:i+1]) / (i+1) for i in range(len(data))]
        plt.plot(cumulative_avg, label=label, color=color)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.grid(True)
    
    save_path = os.path.join('.', f"{title}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def read_list_from_file(filename):
    lst = []
    with open(filename, 'r') as file:
        for line in file:
            item = line.strip().split()[-1]  # Assuming the last item in each line is the score
            item_float = float(item)
            lst.append(item_float)
    return lst



# ------------------------------------------------------ 3 ----------------------------------------------------
#-----------------------------Reading and Plotting the Running Time for each method----------------------------

# This function is used after completing the run to read the log file and plot the running times.

def read_time_list_from_file(file_path):
    running_times = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            method = ' '.join(parts[:-1])  
            time_taken = float(parts[-1])  
            running_times[method] = time_taken
    return running_times

# This function is used during the execution of the code.
def read_running_times(file_path):
    running_times = {}
    with open(file_path, 'r') as file:
        for line in file:
            method, time_taken = line.strip().split()
            running_times[method] = float(time_taken)
    return running_times

# This function is used for plotting the running times.
def plot_running_times(running_times):
    methods = list(running_times.keys())
    times = list(running_times.values())

    plt.figure(figsize=(15, 6))
    bars = plt.bar(methods, times, color=['#B0E0E6', '#4169E1', '#000080', '#FFA07A', '#FF0000', '#8B0000', '#98FB98', '#00FF00', '#008000'])
    plt.title('Running Times for Each Optimization Method')
    plt.xlabel('Optimization Method')
    plt.ylabel('Running Time (seconds)')
    plt.xticks(rotation=45)

    # Add text annotations to each bar
    for bar, time in zip(bars, times):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{time:.2f} s', ha='center', va='bottom')

    plt.tight_layout()
    
    save_path = os.path.join('.', "Running Times for Each Optimization Method.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()



# ------------------------------------------------------ 4 ----------------------------------------------------
#--------------Reading and Plotting the performance (Cumulative Reward) for each method  SEPARATELY---------------


def plot_separated_methods(file_groups, titles, x_label, y_label, legend_groups):
    color_sets = [
        ['#B0E0E6', '#4169E1', '#000080'],  # Colors for Zeroth-Order Optimization
        ['#FFA07A', '#FF0000', '#8B0000'],  # Colors for Population Methods
        ['#98FB98', '#00FF00', '#008000']   # Colors for Policy Gradient
    ]

    for files, title, legends, colors in zip(file_groups, titles, legend_groups, color_sets):
        plt.figure(figsize=(18, 9))
        
        for file_path, label, color in zip(files, legends, colors):
            data = read_list_from_file(file_path)
            cumulative_avg = [sum(data[:i+1]) / (i+1) for i in range(len(data))]
            plt.plot(cumulative_avg, label=label, color=color)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        plt.grid(True)

        save_path = os.path.join('.', f"{title}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()