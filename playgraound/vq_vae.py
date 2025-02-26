import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, latent_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)  # Output shape: (batch_size, latent_dim, H, W)
        return x

# Define Vector Quantization Layer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64):
        super().__init__()
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z_e):
        # Flatten spatial dimensions
        B, C, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, C)

        # Compute L2 distance to all embeddings
        distances = (z_e_flat.unsqueeze(1) - self.codebook.weight.unsqueeze(0)).pow(2).sum(2)
        encoding_indices = distances.argmin(1)

        # Retrieve quantized vectors
        z_q = self.codebook(encoding_indices).view(B, H, W, C).permute(0, 3, 1, 2)
        return z_q, encoding_indices

# Define Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))  # Sigmoid for pixel values (0-1)
        return x

# Define VQ-VAE Model
class VQVAE(nn.Module):
    def __init__(self, latent_dim=64, num_embeddings=512):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, indices = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, z_q, indices

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Load MNIST Dataset
transform = transforms.Compose([transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True),
    batch_size=64,
    shuffle=True
)

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    for images, _ in train_loader:
        images = images.to(device)
        optimizer.zero_grad()

        # Forward pass
        x_recon, _, _ = model(images)

        # Compute loss
        loss = loss_fn(x_recon, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Visualize Reconstructions
def visualize_reconstructions(model, loader):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(loader))
        images = images.to(device)
        recon_images, _, _ = model(images)

        fig, axes = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(10):
            axes[0, i].imshow(images[i].cpu().squeeze(), cmap="gray")
            axes[1, i].imshow(recon_images[i].cpu().squeeze(), cmap="gray")
            axes[0, i].axis("off")
            axes[1, i].axis("off")
        plt.show()

visualize_reconstructions(model, train_loader)
