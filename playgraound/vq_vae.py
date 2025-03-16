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

# Define Vector Quantization Layer with proper gradient handling
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize the embedding table
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z_e):
        # Flatten spatial dimensions
        B, C, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        
        # Calculate distances
        distances = torch.sum(z_e_flat ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.codebook.weight ** 2, dim=1) - \
                    2 * torch.matmul(z_e_flat, self.codebook.weight.t())
                    
        # Find nearest encodings
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Quantize and reshape
        z_q = self.codebook(encoding_indices).view(B, H, W, C)
        z_q = z_q.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Compute loss terms
        # 1. Codebook loss: moves codebook vectors towards encoder outputs
        q_loss = torch.mean((z_q - z_e.detach()) ** 2)
        # 2. Commitment loss: encourages encoder outputs to be close to chosen codebook vectors
        e_loss = torch.mean((z_e - z_q.detach()) ** 2)
        loss = q_loss + self.commitment_cost * e_loss
        
        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()  # Gradient flows from z_q to z_e
        
        return z_q, encoding_indices, loss

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
    def __init__(self, latent_dim=64, num_embeddings=512, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, indices, vq_loss = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, z_q, indices, vq_loss

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
    total_recon_loss = 0
    total_vq_loss = 0
    total_loss = 0
    
    for images, _ in train_loader:
        images = images.to(device)
        optimizer.zero_grad()

        # Forward pass
        x_recon, _, _, vq_loss = model(images)

        # Compute reconstruction loss
        recon_loss = torch.mean((x_recon - images)**2)  # MSE loss
        
        # Combined loss
        loss = recon_loss + vq_loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_loss += loss.item()

    # Print metrics
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_vq_loss = total_vq_loss / len(train_loader)
    avg_loss = total_loss / len(train_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Loss: {avg_loss:.4f}, "
          f"Recon Loss: {avg_recon_loss:.4f}, "
          f"VQ Loss: {avg_vq_loss:.4f}")

# Visualize Reconstructions
def visualize_reconstructions(model, loader):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(loader))
        images = images.to(device)
        recon_images, _, _, _ = model(images)

        fig, axes = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(10):
            axes[0, i].imshow(images[i].cpu().squeeze(), cmap="gray")
            axes[1, i].imshow(recon_images[i].cpu().squeeze(), cmap="gray")
            axes[0, i].axis("off")
            axes[1, i].axis("off")
        plt.title("Original (top) vs. Reconstructed (bottom)")
        plt.tight_layout()
        plt.show()

visualize_reconstructions(model, train_loader)
