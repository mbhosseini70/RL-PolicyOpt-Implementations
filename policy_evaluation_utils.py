import torch

def evaluate_policy(policy, env, perturbed_params, device):

    # Save the original parameters of the policy
    old_params = [param.data.clone() for param in policy.parameters()]

    # Replace the policy parameters with the perturbed parameters
    for param, perturbed_param in zip(policy.parameters(), perturbed_params):
        param.data.copy_(perturbed_param)

    state, _ = env.reset()    # Reset the environment to start a new episode
    done = False
    total_reward = 0
    max_steps = 1000  # Maximum steps to prevent infinite loops
    step_count = 0


    # Run the episode until it's done or until the maximum number of steps is reached
    while not done and step_count < max_steps:
        # Convert state to a tensor and move to the appropriate device
        state_tensor = torch.from_numpy(state).float().to(device)

        # Get the action from the policy
        action = policy(state_tensor).cpu().detach().numpy()

        # Take a step in the environment with the chosen action
        result = env.step(action)
        state = result[0]   # Get the new state
        reward = result[1]  # Get the reward
        done = result[2]   # Check if the episode is done
        total_reward += reward   # Accumulate the reward
        step_count += 1   # Increment the step count



    # Restore the original parameters of the policy
    for param, old_param in zip(policy.parameters(), old_params):
        param.data.copy_(old_param)


    # Return the total reward obtained in this episode
    return total_reward
