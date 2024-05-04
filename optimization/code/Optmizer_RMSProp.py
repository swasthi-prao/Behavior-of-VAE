
from dataset.mnist_loader import MnistDataset
from tqdm import tqdm
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=2):
        """
        Initialize the Variational Autoencoder (VAE) model.

        Args:
            latent_dim (int): Dimensionality of the latent space.
        """
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Mean and log variance layers for the latent space
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid()  # Sigmoid activation for pixel intensity output (0 to 1 range)
        )
    
    def ENCODE(self, x):
        """
        Encode input images into the latent space.

        Args:
            x (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Latent mean.
            torch.Tensor: Latent log variance.
        """
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        latent_mean = self.fc_mean(x)
        latent_logvar = self.fc_logvar(x)
        return latent_mean, latent_logvar
    
    def REPARAMETERIZE(self, mean, logvar):
        """
        Reparameterization trick to sample from the latent distribution.

        Args:
            mean (torch.Tensor): Latent mean.
            logvar (torch.Tensor): Latent log variance.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        sampled_latent = mean + epsilon * std
        return sampled_latent
    
    def DECODE(self, z):
        """
        Decode latent vectors back into image space.

        Args:
            z (torch.Tensor): Latent vectors.

        Returns:
            torch.Tensor: Reconstructed images.
        """
        reconstructed_images = self.decoder(z)
        return reconstructed_images
    
    def FORWARD(self, x):
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Reconstructed images.
            torch.Tensor: Latent mean.
            torch.Tensor: Latent log variance.
        """
        latent_mean, latent_logvar = self.ENCODE(x)
        sampled_latent = self.REPARAMETERIZE(latent_mean, latent_logvar)
        reconstructed_images = self.DECODE(sampled_latent)
        return reconstructed_images, latent_mean, latent_logvar


def TRAIN_VAE():
    """
    Train the Variational Autoencoder (VAE) model using RMSprop optimizer.
    """
    torch.manual_seed(42)  # Set random seed for reproducibility
    
    # Create the dataset and data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_dataset = MnistDataset('train', transform=transform)
    mnist_loader = DataLoader(mnist_dataset, batch_size=128, shuffle=True, num_workers=4)
    
    # Instantiate the VAE model
    vae_model = VariationalAutoencoder().to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.RMSprop(vae_model.parameters(), lr=1e-3)
    reconstruction_criterion = nn.BCELoss(reduction='sum')  # Binary Cross Entropy Loss
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        vae_model.train()
        total_loss = 0.0
        
        for images, _ in tqdm(mnist_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            
            # Forward pass
            reconstructed_images, latent_mean, latent_logvar = vae_model(images)
            
            # Compute reconstruction loss and KL divergence
            recon_loss = reconstruction_criterion(reconstructed_images, images.view(-1, 28*28))
            kl_divergence = -0.5 * torch.sum(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
            
            # Total loss
            loss = recon_loss + 0.001 * kl_divergence
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print average loss for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(mnist_loader):.4f}")
    
    # Generate and save some reconstructed images
    vae_model.eval()
    with torch.no_grad():
        sample_images, _ = next(iter(mnist_loader))
        sample_images = sample_images.to(device)
        reconstructed_images, _, _ = vae_model(sample_images)
        
        # Denormalize images for visualization (assuming images are in range [-1, 1])
        reconstructed_images = (reconstructed_images * 0.5) + 0.5
        
        # Create a grid of original and reconstructed images
        comparison_images = torch.cat([sample_images[:8], reconstructed_images.view(-1, 1, 28, 28)[:8]])
        grid = torchvision.utils.make_grid(comparison_images, nrow=8, pad_value=1.0)
        
        # Save the grid as an image
        save_path = "RECONSTRUCTION_RESULT.png"
        torchvision.utils.save_image(grid, save_path)
        print(f"Reconstructed images saved at: {os.path.abspath(save_path)}")


if __name__ == '__main__':
    TRAIN_VAE()
