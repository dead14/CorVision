import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import CorrosionDataset
from model import create_model
import os

def train():
    # 1. Setup Device (Gunakan GPU jika ada)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Menggunakan device: {device}")

    # 2. Siapkan Dataset
    dataset_path = "YoloV8Corrosion.v1i.png-mask-semantic/test"
    full_dataset = CorrosionDataset(dataset_path)
    
    if len(full_dataset) == 0:
        print("Dataset kosong! Pastikan path benar.")
        return

    # Untuk contoh ini, karena kita menggunakan CPU (Python 3.14 belum ada versi GPU), 
    # kita batasi dataset ke 150 gambar pertama agar training awal cepat selesai
    subset_size = min(150, len(full_dataset))
    indices = torch.randperm(len(full_dataset))[:subset_size]
    subset_dataset = torch.utils.data.Subset(full_dataset, indices)

    # Bagi menjadi Train (80%) dan Validation (20%)
    train_size = int(0.8 * len(subset_dataset))
    val_size = len(subset_dataset) - train_size
    train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # 3. Buat Model
    model = create_model().to(device)

    # 4. Tentukan Loss Function & Optimizer
    # Kombinasi BCE (Binary Cross Entropy) yang bagus untuk biner (korosi / tidak korosi)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 5. Training Loop
    epochs = 5 # Ubah ini menjadi 20 atau 50 jika ingin akurasi maksimal nanti
    
    print(f"\nMulai melatih model dengan {train_size} data train dan {val_size} data validasi selama {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"==> Epoch {epoch+1} Selesai! | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}\n")

    # 6. Simpan Model
    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/corrosion_unet.pth")
    print("Training Selesai! Model berhasil disimpan di 'models/corrosion_unet.pth'")

if __name__ == "__main__":
    train()
