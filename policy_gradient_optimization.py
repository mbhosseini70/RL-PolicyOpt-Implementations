import torch
import torch.optim as optim


def policy_gradient_optimization(policy, env, episodes, learning_rate, log_file):
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)


    with open(log_file, 'w') as file:
        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            log_probs = []
            rewards = []
            
            while not done:
                # Converts the state to a PyTorch tensor, moves it to the device, and adds a batch dimension.
                state_tensor = torch.from_numpy(state).float().to(device).unsqueeze(0)
                # Passes the state through the policy network to get the mean of the action distribution.
                action_mean = policy(state_tensor)  # Neural network output is the mean (μ)

                #Creates a normal distribution for the action with the mean from the policy and the fixed standard deviation.
                action_dist = torch.distributions.Normal(action_mean, torch.ones_like(action_mean))  # Gaussian distribution
                
                #Samples an action from the distribution, converts it to a numpy array.
                action = action_dist.sample().cpu().numpy()  

                # Clips the action to ensure it is within the valid action space.
                action = action.clip(env.action_space.low, env.action_space.high)
                
                next_state, reward, terminated, truncated, _ = env.step(action.squeeze(0))  # Remove batch dimension
                
                log_prob = action_dist.log_prob(torch.tensor(action)).sum()  # Log likelihood of the action
                
                log_probs.append(log_prob)
                rewards.append(reward)
                state = next_state
                total_reward += reward
                done = terminated or truncated
            
            # Compute the discounted rewards
            discounted_rewards = []
            for t in range(len(rewards)):
                Gt = sum([r * (0.99 ** i) for i, r in enumerate(rewards[t:])])
                discounted_rewards.append(Gt)
            
            discounted_rewards = torch.tensor(discounted_rewards).to(device)
            
            # Normalize rewards (advantage)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

            # Compute loss
            policy_loss = []
            for log_prob, Gt in zip(log_probs, discounted_rewards):
                policy_loss.append(-log_prob * Gt)  # Loss is negative log likelihood weighted by the advantage
            
            optimizer.zero_grad()
            policy_loss = torch.stack(policy_loss).sum()  # Stack and sum the losses
            policy_loss.backward()
            optimizer.step()
            
            file.write(f"RETURN {episode + 1} {total_reward}\n")
            print(f"Policy Gradient, Episode: {episode + 1}, Total Reward: {total_reward}")

