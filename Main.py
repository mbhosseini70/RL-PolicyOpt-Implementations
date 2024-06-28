# ------------------------------------------------------ 1 ----------------------------------------------------
#------------------------------------------------Importing Libraries-------------------------------------------
import gymnasium as gym
import torch
import numpy as np
import time

from policy_NN import PolicyNetwork
from zeroth_order_optimization import zeroth_order_optimization
from population_method import population_method
from policy_gradient_optimization import policy_gradient_optimization
from plotting import plot_multiple_files
from plotting import plot_running_times, plot_separated_methods



# ------------------------------------------------------ 2 ----------------------------------------------------
#--------------------------------------------------Initializations---------------------------------------------

# Set the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment setup
env = gym.make('LunarLanderContinuous-v2')

# Number of the episodes
Number_of_episodes = 1000

# Run the training with different settings
learning_rates = [0.001, 0.005, 0.01]
population_sizes = [10, 20, 30]


# Dictionary to store running times
running_times = {}


# ------------------------------------------------------ 3 ----------------------------------------------------
#-----------------------------------------------------Running--------------------------------------------------



#-----------------------------------------------------3.1-------------------------------------------------
#--------------------------------------------zeroth order optimization------------------------------------

# Run zeroth order optimization for different learning rates
for lr in learning_rates:
    policy = PolicyNetwork().to(device)  # Reinitialize the policy
    log_file_name = f'zeroth_order_lr_{lr}.txt'
    
    start_time = time.time()  
    zeroth_order_optimization(policy, env, Number_of_episodes, lr, log_file_name)
    end_time = time.time()  
    
    running_time = end_time - start_time
    running_times[f'Zeroth Order LR {lr}'] = running_time



#-----------------------------------------------------3.2-------------------------------------------------
#----------------------------------------------Population method------------------------------------------


# Run population method for different population sizes
for size in population_sizes:
    policy = PolicyNetwork().to(device)  
    log_file_name = f'population_size_{size}.txt'
    
    start_time = time.time()  
    population_method(policy, env, Number_of_episodes, size, log_file_name)
    end_time = time.time()  
    
    running_time = end_time - start_time
    running_times[f'Population Size {size}'] = running_time


#-----------------------------------------------------3.3-------------------------------------------------
#------------------------------------------Policy Gradient Optimization-----------------------------------


# Run policy gradient optimization for different learning rates
for lr in learning_rates:
    policy = PolicyNetwork().to(device)  
    log_file_name = f'policy_gradient_lr_{lr}.txt'
    
    start_time = time.time()  
    policy_gradient_optimization(policy, env, Number_of_episodes, lr, log_file_name)
    end_time = time.time()  
    
    running_time = end_time - start_time
    running_times[f'Policy Gradient LR {lr}'] = running_time


# ------------------------------------------------------ 4 ----------------------------------------------------
#-----------------------------------------------Saving the Running Times---------------------------------------

# Save running times to a file
with open('running_times.txt', 'w') as file:
    for method, time_taken in running_times.items():
        file.write(f"{method} {time_taken}\n")    



# ------------------------------------------------------ 5 ----------------------------------------------------
#-----------------------------------------------------Plotting-------------------------------------------------



# ------------------------------------------------------ 5.1 --------------------------------------------------
#------------------------------------------Defining the Path of Log Files--------------------------------------


# Define paths to all experiment log files
experiment_files = [
    f'zeroth_order_lr_{lr}.txt' for lr in learning_rates
] + [
    f'population_size_{size}.txt' for size in population_sizes
] + [
    f'policy_gradient_lr_{lr}.txt' for lr in learning_rates
]

# Corresponding labels for the legend in the plot
experiment_labels = [
    f'Zeroth Order LR {lr}' for lr in learning_rates
] + [
    f'Population Size {size}' for size in population_sizes
] + [
    f'Policy Gradient LR {lr}' for lr in learning_rates
]


# ------------------------------------------------------ 5.2 --------------------------------------------------
#--------------------------------Plotting all results (Average reward) in One Figure---------------------------

# Plot all experiments on the same graph
plot_multiple_files(
    file_paths=experiment_files,
    title='Comparison of Zeroth-Order Optimization, Population Methods, and Policy Gradient',
    x_label='Episodes',
    y_label='Cumulative Reward',
    legend_labels=experiment_labels
)

# ------------------------------------------------------ 5.3 --------------------------------------------------
#---------------------------------------------Plotting Running Times ------------------------------------------

# Plot the running times for each method
plot_running_times(running_times)



# ------------------------------------------------------ 5.4 --------------------------------------------------
#-------------------------Plotting (Average reward) Separately for Each Optimization Method ------------------------------------------


zeroth_order_files = [f'zeroth_order_lr_{lr}.txt' for lr in learning_rates]
population_files = [f'population_size_{size}.txt' for size in population_sizes]
policy_gradient_files = [f'policy_gradient_lr_{lr}.txt' for lr in learning_rates]

# Corresponding labels for the legend in the plot
zeroth_order_labels = [f'Zeroth Order LR {lr}' for lr in learning_rates]
population_labels = [f'Population Size {size}' for size in population_sizes]
policy_gradient_labels = [f'Policy Gradient LR {lr}' for lr in learning_rates]

file_groups = [zeroth_order_files, population_files, policy_gradient_files]
titles = [
    'Zeroth-Order Optimization Comparison',
    'Population Methods Comparison',
    'Policy Gradient Comparison'
]
legend_groups = [zeroth_order_labels, population_labels, policy_gradient_labels]

# Call the function to plot and save the figures
plot_separated_methods(file_groups, titles, 'Episodes', 'Average Reward', legend_groups)