import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from prediction_model import *


class ShelterDataset(Dataset):
    def __init__(self, df, surge_cols, building_cols, wind_cols, label_col):
        self.surge = torch.tensor(df[surge_cols].values, dtype=torch.float32)
        self.building = torch.tensor(df[building_cols].values, dtype=torch.float32)
        self.wind = torch.tensor(df[wind_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(df[label_col].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.surge[idx], self.building[idx], self.wind[idx], self.labels[idx]

df = pd.read_csv("./data/cleaned_data.csv")

surge_cols = ['water']
building_cols = ['age', 'roof_cover_converted', 'roof_shape_converted', 'wall_cladding_converted', 'wall_structure_converted', 'roof_slope']
wind_cols = ['wind']
label_col = 'label'

dataset = ShelterDataset(df, surge_cols, building_cols, wind_cols, label_col)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = PredictionModel(1, 6, 1, hidden_dim=60)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4, betas=(0.85, 0.999))

epochs = 2000
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for surge, building, wind, labels in loader:
        logits = model(surge, building, wind)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(loader):.4f}")

torch.save(model, 'full_model.pth')