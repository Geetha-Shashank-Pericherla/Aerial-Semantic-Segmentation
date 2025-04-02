import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
from dataset import AerialDataset
from model import UNet
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = AerialDataset()
train_set, val_set = torch.utils.data.random_split(dataset, [320, 80])

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader = DataLoader(val_set, batch_size=4)

model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = smp.losses.SoftBCEWithLogitsLoss()

# Training loop
for epoch in range(10):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader)}")

    torch.save(model.state_dict(), "models/unet_model.pth")
