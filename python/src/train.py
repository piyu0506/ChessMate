import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import ChessNet

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Detected Device: {device}")

    # Load massive dataset
    X = np.load("python/data/X_train.npy")
    y = np.load("python/data/y_train.npy")
    z = np.load("python/data/z_train.npy")

    dataset = TensorDataset(torch.from_numpy(X).float(), 
                            torch.from_numpy(y).long(), 
                            torch.from_numpy(z).float().unsqueeze(1))
    
    # Larger batch size for GPU efficiency
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    model = ChessNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # RL-Loss: Actor (Policy) + Critic (Value)
    criterion_actor = nn.CrossEntropyLoss()
    criterion_critic = nn.MSELoss()

    model.train()
    for epoch in range(20): # 20 epochs on 1M positions is very strong
        running_loss = 0.0
        for bx, by, bz in loader:
            bx, by, bz = bx.to(device), by.to(device), bz.to(device)
            optimizer.zero_grad()
            
            policy, value = model(bx)
            loss = criterion_actor(policy, by) + 0.5 * criterion_critic(value, bz)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Loss: {running_loss/len(loader):.5f}")

    torch.save(model.state_dict(), "python/models/chess_model.pth")

if __name__ == "__main__":
    train()