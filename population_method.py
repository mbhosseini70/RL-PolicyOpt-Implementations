import torch
import numpy as np
from policy_evaluation_utils import evaluate_policy

def population_method(policy, env, episodes, population_size, log_file):

    # Select the device to run the calculations on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    # Open the log file to write the results
    with open(log_file, 'w') as file:

        # Loop through each episode
        for episode in range(episodes):
            print(f"Population Methods, Episode number: {episode + 1}, Population size: {population_size}")

            # Initialize lists to store scores and perturbed parameters
            scores = []
            perturbed_params_list = []

            # Step 2: Produce N perturbations of the policy parameters (theta)
            for _ in range(population_size):

                # Create a perturbation vector and add it to the current parameters
                perturbed_params = [p.data + torch.randn_like(p) for p in policy.parameters()]

                # Step 3: Evaluate each perturbed policy in the environment
                score = evaluate_policy(policy, env, perturbed_params, device)
                scores.append(score)
                perturbed_params_list.append(perturbed_params)

            # Step 4: Select the perturbed parameters with the highest score
            best_idx = np.argmax(scores)
            best_params = perturbed_params_list[best_idx]


            # Step 5: Update the policy parameters with the best perturbed parameters
            with torch.no_grad():
                for param, best_param in zip(policy.parameters(), best_params):
                    param.data.copy_(best_param)

            file.write(f"RETURN {episode + 1} {scores[best_idx]}\n")



