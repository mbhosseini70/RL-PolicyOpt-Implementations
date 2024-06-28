# RL-PolicyOpt-Implementations

## Introduction
This project is part of a Reinforcement Learning (RL) course, with the primary objective of implementing an RL agent to perform the LunarLanderContinuous task using OpenAI Gym. The project explores various optimization techniques to enhance the agent's learning process, focusing on Policy Gradient Optimization and Gradient-Free Optimization methods.

The project is structured into several tasks that investigate different optimization approaches. This README provides an overview of the project, its environment, and the files included.

## Environment
### LunarLanderContinuous-v2
LunarLanderContinuous-v2 is an environment in OpenAI Gym designed to simulate a lunar landing scenario. The goal is to land a spacecraft safely on the moon's surface. The environment provides a rich platform for testing reinforcement learning algorithms with continuous control challenges.

### Goal
The ultimate goal is to land the spacecraft gently on the designated landing pad without crashing.

## Project Description

The project consists of several Python files that implement different components of the RL agent and its optimization methods:

- **`policy_NN.py`**: Implements the Neural Network Policy for LunarLanderContinuous-v2.
- **`policy_gradient_optimization.py`**: Implements policy gradient descent for parametric policy optimization.
- **`zeroth_order_optimization.py`**: Implements Zeroth-order Optimization as a gradient-free method.
- **`population_method.py`**: Implements a basic Population Method for gradient-free optimization.
- **`policy_evaluation_utils.py`**: Provides utilities for evaluating gradient-free optimizations.
- **`plotting.py`**: Contains functions for plotting the results.
- **`main.py`**: The main script to run the entire codebase.

## Usage
To run the project, execute the main.py script. This will initiate the training and evaluation process using the specified optimization methods.

## Results
In the report file, we compares the results obtained from different optimization methods, providing insights into their performance and efficiency in solving the LunarLanderContinuous task.
