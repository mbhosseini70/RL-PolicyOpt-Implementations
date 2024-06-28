import torch
import numpy as np
from policy_evaluation_utils import evaluate_policy

def zeroth_order_optimization(policy, env, episodes, learning_rate, log_file):

    # Select the device to run the calculations on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)


    # Open the log file to write the results
    with open(log_file, 'w') as file:
        # Loop through each episode
        for episode in range(episodes):
            print(f"Zeroth-order Optimization, Episode number: {episode + 1}, Learning rate: {learning_rate}")


            # Step 1: Get the current parameters of the policy (theta)
            theta = [p.clone().detach() for p in policy.parameters()]

            # Step 2: Create a perturbation vector with the same shape as theta
            perturbation = [torch.randn_like(p) for p in theta]

            # Step 3: Create theta+ and theta- by adding and subtracting the perturbation vector
            theta_plus = [p + d for p, d in zip(theta, perturbation)]
            theta_minus = [p - d for p, d in zip(theta, perturbation)]


            # Step 4: Evaluate the policy with perturbed parameters
            score_plus = evaluate_policy(policy, env, theta_plus, device)
            score_minus = evaluate_policy(policy, env, theta_minus, device)


            # Step 6: Compute the gradient approximation
            gradient = [0.5 * (score_plus - score_minus) * d for d in perturbation]

            
            # Step 7: Update the parameters of the policy using the computed gradient
            with torch.no_grad():
                for param, grad in zip(policy.parameters(), gradient):
                    param.add_(learning_rate * grad)

            file.write(f"RETURN {episode + 1} {max(score_plus, score_minus)}\n")

