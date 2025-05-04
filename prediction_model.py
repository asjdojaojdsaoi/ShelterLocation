import torch
import torch.nn as nn

class PredictionModel(nn.Module):
    def __init__(self, surge_dim, building_dim, wind_dim, hidden_dim):
        super(PredictionModel, self).__init__()
        self.fc1 = nn.Linear(surge_dim + building_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)  # ← 添加 BatchNorm1d
        self.fc2 = nn.Linear(wind_dim + building_dim, hidden_dim)
        # self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim, hidden_dim)
        # self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.1, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, surge_feat, building_feat, wind_feat):
        x1 = torch.cat((surge_feat, building_feat), dim=1)
        x1 = self.fc1(x1)
        # x1 = self.bn1(x1)
        x1 = torch.tanh(x1)  # BatchNorm → 激活

        x2 = torch.cat((wind_feat, building_feat), dim=1)
        x2 = self.fc2(x2)
        # x2 = self.bn2(x2)
        x2 = torch.tanh(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc3(x)
        # x = self.bn3(x)
        x = torch.tanh(x)

        logits = self.fc4(x)
        return logits