
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=2.99):
        super(ActorNetwork, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)

        self.max_action = max_action
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.002)
    def forward(self, x, u):
        temp_ = torch.cat([x, u], 0)
        temp = self.l1(temp_)
        x = F.relu(temp)
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

