import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import ChessNet

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    try:
        X = np.load(os.path.join(data_dir, "X_train.npy"))
        y = np.load(os.path.join(data_dir, "y_train.npy"))
        z = np.load(os.path.join(data_dir, "z_train.npy"))
    except:
        print("Run data_prepare.py first!")
        return

    dataset = TensorDataset(
        torch.from_numpy(X).float(), 
        torch.from_numpy(y).long(), 
        torch.from_numpy(z).float().unsqueeze(1)
    )
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    model = ChessNet(num_res_blocks=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    p_loss_fn = nn.CrossEntropyLoss()
    v_loss_fn = nn.MSELoss()
    
    print("Starting Training...")
    model.train()
    for epoch in range(15):
        total_p, total_v = 0, 0
        for bx, by, bz in loader:
            bx, by, bz = bx.to(device), by.to(device), bz.to(device)
            optimizer.zero_grad()
            p, v = model(bx)
            
            lp = p_loss_fn(p, by)
            lv = v_loss_fn(v, bz)
            loss = lp + (0.5 * lv)
            
            loss.backward()
            optimizer.step()
            total_p += lp.item(); total_v += lv.item()
            
        print(f"Epoch {epoch+1} | Policy: {total_p/len(loader):.4f} | Value: {total_v/len(loader):.4f}")
    
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "chess_model.pth")
    torch.save(model.state_dict(), save_path)
    print("Model Saved.")

if __name__ == "__main__":
    train()