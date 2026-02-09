import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from model import ChessNet

def train():
    # Load data from the processed folder
    X = np.load("python/data/X_train.npy")
    y = np.load("python/data/y_train.npy")

    # Reshape flattened bitboards back to 8x8 grids
    X = X.reshape(-1, 12, 8, 8)
    
    X_train = torch.from_numpy(X).float()
    y_train = torch.from_numpy(y).long()

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {device}...")
    model.train()
    
    for epoch in range(10):
        running_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            policy, value = model(batch_x)
            
            loss = criterion(policy, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(loader):.4f}")

    os.makedirs("python/models", exist_ok=True)
    torch.save(model.state_dict(), "python/models/chess_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()