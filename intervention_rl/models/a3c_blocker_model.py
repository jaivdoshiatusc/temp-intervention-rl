import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, action_size):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=2)  
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=2)  

        self.fc1 = nn.Linear(4 * 25 * 19 + action_size, 10) 
        self.fc2 = nn.Linear(10, 10)
        self.fc_out = nn.Linear(10, 2) 

    def forward(self, obs, action):
        x = torch.relu(self.conv1(obs))
        x = torch.relu(self.conv2(x))

        x = x.view(x.size(0), -1)

        x = torch.cat([x, action], dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc_out(x)
