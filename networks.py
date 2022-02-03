# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F


# # class ActorNetwork(nn.Module):
# #     def __init__(self, s_dim, a_dim):
# #         super(ActorNetwork, self).__init__()
# #         self.fc1 = nn.Linear(s_dim, 30)
# #         self.fc1.weight.data.normal_(0, 0.1)  # initialization of FC1
# #         self.out = nn.Linear(30, a_dim)
# #         self.out.weight.data.normal_(0, 0.1)  # initilizaiton of OUT

# #         self.optimizer = torch.optim.Adam(self.parameters(), lr=0.002)

# #     def forward(self, x):
# #         x = self.fc1(x)
# #         x = F.relu(x)
# #         x = self.out(x)
# #         x = torch.tanh(x)
# #         action = x * 2  # for the game "Pendulum-v0", action range is [-2, 2]
# #         return action


# # class CriticNetwork(nn.Module):
# #     def __init__(self, s_dim, a_dim):
# #         super(CriticNetwork, self).__init__()
# #         self.fcs = nn.Linear(s_dim, 30)
# #         self.fcs.weight.data.normal_(0, 0.1)
# #         self.fca = nn.Linear(a_dim, 30)
# #         self.fca.weight.data.normal_(0, 0.1)
# #         self.out = nn.Linear(30, 1)
# #         self.out.weight.data.normal_(0, 0.1)
# #         self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

# #     def forward(self, s, a):
# #         x = self.fcs(s)
# #         y = self.fca(a)
# #         actions_value = self.out(F.relu(x + y))
# #         return actions_value



# # class Actor(nn.Module):
# #     def __init__(self, input_size, hidden_size, output_size):
# #         super(Actor, self).__init__()
# #         self.linear1 = nn.Linear(input_size, hidden_size)
# #         self.linear2 = nn.Linear(hidden_size, hidden_size)
# #         self.linear3 = nn.Linear(hidden_size, output_size)
        
# #     def forward(self, s):
# #         x = F.relu(self.linear1(s))
# #         x = F.relu(self.linear2(x))
# #         x = torch.tanh(self.linear3(x))

# #         return x


# # class Critic(nn.Module):
# #     def __init__(self, input_size, hidden_size, output_size):
# #         super().__init__()
# #         self.linear1 = nn.Linear(input_size, hidden_size)
# #         self.linear2 = nn.Linear(hidden_size, hidden_size)
# #         self.linear3 = nn.Linear(hidden_size, output_size)

# #     def forward(self, s, a):
# #         x = torch.cat([s, a], 1)
# #         x = F.relu(self.linear1(x))
# #         x = F.relu(self.linear2(x))
# #         x = self.linear3(x)

# #         return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=2):
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

