import torch.nn as nn



# Hardcoded
"""

inputs = 8
hidden neurons = 128
outputs = 2

"""

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(8, 128)  # Input to hidden layer
        self.relu = nn.ReLU()  # Activation function for hidden layer
        self.fc2 = nn.Linear(128, 2)  # Hidden to output layer
        self.tanh = nn.Tanh()  # Bounded activation function for output, 
        #                        Ensure output is bounded between -1 and 1
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.tanh(x)  