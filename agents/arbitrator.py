# agents/arbitrators/ceo_arbitrator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class arbitrator_ceo(nn.Module):
    """CEO Arbitrator - decides which module to listen to."""
    
    def __init__(self, input_dim, action_dim, lr, gamma=0.99):
        super(arbitrator_ceo, self).__init__()
        self.hidden_dim = 512
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.gamma = gamma  # ✅ Fixed
        self.lr = lr
        self.eps = 1e-20  # ✅ Fixed (was 10e-20, which is 1e-19)
        
        # Neural network layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.action_dim)
        self.relu = nn.ReLU()
        
        # For training
        self.saved_log_probs = []
        self.rewards = []
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        """Input: internal stats, Output: probability for each module"""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        action_scores = self.fc3(x)
        return F.softmax(action_scores, dim=1)

    def update(self, device):
        """Learn from episode rewards using Policy Gradient"""
        R = 0
        policy_loss = []
        returns = []
        
        # Calculate cumulative returns (backwards)
        for r in self.rewards[::-1]:
            R = r + self.gamma * R  # ✅ Fixed: self.gamma
            returns.insert(0, R)
        
        # Normalize returns
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + self.eps)
        
        # Calculate policy loss
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Update weights
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear buffers
        del self.rewards[:]
        del self.saved_log_probs[:]