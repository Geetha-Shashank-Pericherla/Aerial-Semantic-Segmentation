import torch
import matplotlib.pyplot as plt
from dataset import AerialDataset
from model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = UNet().to(device)
model.load_state_dict(torch.load("models/unet_model.pth"))
model.eval()

dataset = AerialDataset()
test_img, test_label = dataset[0]
test_img = test_img.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(test_img)

output = output.squeeze(0).cpu().numpy()

# Display original and predicted image
fig, ax = plt.subplots(1, 2)
ax[0].imshow(test_img.cpu().squeeze().permute(1, 2, 0))
ax[0].set_title("Original Image")
ax[1].imshow(output.argmax(axis=0))
ax[1].set_title("Predicted Segmentation")
plt.show()
