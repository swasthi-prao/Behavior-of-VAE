# basic imports
import torch
import cv2
import torchvision
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from einops import rearrange
from torch.optim import Adam
from dataset.mnist_loader import MnistDataset
from torch.utils.data import DataLoader
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.common_fc = nn.Sequential(
            nn.Linear(28*28, 196),
            nn.Tanh(),
            nn.Linear(196, 48),
            nn.Tanh(),
        )
        self.mean_fc = nn.Sequential(
            nn.Linear(48, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
        
        self.log_var_fc = nn.Sequential(
            nn.Linear(48, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
        
        self.decoder_fcs = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 48),
            nn.Tanh(),
            nn.Linear(48, 196),
            nn.Tanh(),
            nn.Linear(196, 28*28),
            nn.Tanh()
        )
        
    def forward(self, x):
        # B,C,H,W
        ## Encoder part
        mean, log_var = self.encoder(x)
        ## Sampling
        params = self.sample(mean, log_var)
        ## Decoder part
        res = self.decoder(params)
        return mean, log_var, res

    def encoder(self, x):
        res = self.common_fc(torch.flatten(x, start_dim=1))
        mean = self.mean_fc(res)
        log_var = self.log_var_fc(res)
        return mean, log_var
    
    def sample(self, mean, log_var):
        stdev = torch.exp(0.5 * log_var)
        res = torch.randn_like(stdev)
        res = res * stdev + mean
        return res
    
    def decoder(self, y):
        outp = self.decoder_fcs(y)
        outp = outp.reshape((y.size(0), 1, 28, 28))
        return outp
        
        

def generate_samples_vae(model, num_samples):
    with torch.no_grad():
        # Sample from the latent space
        latent_samples = torch.randn(num_samples, 2).to(device)
        # Decode the samples
        generated_images = model.decoder(latent_samples)
    return generated_images


def compare_and_save_samples(dataset, generated_images):
    # Select some random samples from the test dataset
    idxs = torch.randint(0, len(dataset) - 1, (100,))
    original_images = torch.cat([dataset[idx][0][None, :] for idx in idxs]).float()

    # Preprocess the original and generated images for visualization
    original_images = (original_images + 1) / 2
    generated_images = (generated_images + 1) / 2

    # Save original samples
    original_grid = torchvision.utils.make_grid(original_images, nrow=10)
    original_image = torchvision.transforms.ToPILImage()(original_grid)
    original_image.save('original_samples.png')
    print('Original samples image saved.')

    # Save generated samples
    generated_grid = torchvision.utils.make_grid(generated_images, nrow=10)
    generated_image = torchvision.transforms.ToPILImage()(generated_grid)
    generated_image.save('generated_samples.png')
    print('Generated samples image saved.')


def train_vae():
    # Create the data set and the data loader
    mnist = MnistDataset('train', im_path='data/train/images')
    mnist_test = MnistDataset('test', im_path='data/test/images')
    mnist_loader = DataLoader(mnist, batch_size=64, shuffle=True, num_workers=0)
    
    # Instantiate the model
    model = VAE().to(device)
    
    # Specify training parameters
    n_epochs = 10
    optim = Adam(model.parameters(), lr=1E-3)
    criterion = torch.nn.MSELoss()
    
    recon_l = []
    kl_l = []
    losses = []
    # Run training for 10 epochs
    start_time = time.time()
    for epoch_idx in range(n_epochs):
        for im, label in tqdm(mnist_loader):
            im = im.float().to(device)
            optim.zero_grad()
            mean, log_var, out = model(im)
            
            klloss = torch.mean(0.5* torch.sum(torch.exp(log_var) + mean**2 - 1 -log_var, dim=-1))
            reconloss = criterion(out, im)
            loss = reconloss + 0.00001 * klloss
            recon_l.append(reconloss.item())
            losses.append(loss.item())
            kl_l.append(klloss.item())
            loss.backward()
            optim.step()
        print('Finished epoch:{} | Recon Loss : {:.4f} | KL Loss : {:4f}'.format(
            epoch_idx+1,
            np.mean(recon_l),
            np.mean(kl_l)
        ))
    end_time = time.time()
    print("Training time for training loop 1:", end_time - start_time, "seconds")
    print('Done Training ...')
    
    # Get the reconstruction image obtained by the VAE
    idxs = torch.randint(0, len(mnist_test) - 1, (100,))
    ims = torch.cat([mnist_test[idx][0][None, :] for idx in idxs]).float()

    _, _, generated_im = model(ims)

    ims = (ims + 1) / 2
    generated_im = 1 - (generated_im + 1) / 2
    out = torch.hstack([ims, generated_im])
    output = rearrange(out, 'b c h w -> b () h (c w)')
    grid = torchvision.utils.make_grid(output, nrow=10)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save('reconstruction.png')
    print('Done Reconstruction 1...')
    
    # Generate new data samples from the latent space of the VAE
    generated_images = generate_samples_vae(model, num_samples=100)

    # Concatenate original and generated samples and train the VAE again
    concatenated_images = torch.cat([mnist[idx][0][None, :] for idx in idxs] + [generated_images])
    concatenated_loader = DataLoader(concatenated_images, batch_size=64, shuffle=True)
    
    # Store the original samples and show a comparison grid
    compare_and_save_samples(mnist, generated_images)
    
    
    # Train the VAE again
    recon_l = []
    kl_l = []
    losses = []
    start_time = time.time()
    for epoch_idx in range(n_epochs):
        for im in tqdm(concatenated_loader):
            im = im.float().to(device)
            optim.zero_grad()
            mean, log_var, out = model(im)
            
            klloss = torch.mean(0.5 * torch.sum(torch.exp(log_var) + mean ** 2 - 1 - log_var, dim=-1))
            reconloss = criterion(out, im)
            loss = reconloss + 0.00001 * klloss
            recon_l.append(reconloss.item())
            losses.append(loss.item())
            kl_l.append(klloss.item())
            loss.backward()
            optim.step()
        print('Finished epoch:{} | Recon Loss : {:.4f} | KL Loss : {:4f}'.format(
            epoch_idx + 1,
            np.mean(recon_l),
            np.mean(kl_l)
        ))
    
    end_time = time.time()
    print("Training time for training loop 2 (with augmentation):", end_time - start_time, "seconds")
    print('Done Training Second Time ...')
    
    # Get the second reconstruction image
    idxs = torch.randint(0, len(mnist_test) - 1, (100,))
    ims = torch.cat([mnist_test[idx][0][None, :] for idx in idxs]).float()

    _, _, generated_im = model(ims)

    ims = (ims + 1) / 2
    generated_im = 1 - (generated_im + 1) / 2
    out = torch.hstack([ims, generated_im])
    output = rearrange(out, 'b c h w -> b () h (c w)')
    grid = torchvision.utils.make_grid(output, nrow=10)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save('reconstructionwithaug.png')
    print('Done Reconstruction 2...')

    original_samples = torch.cat([mnist[idx][0][None, :] for idx in idxs])
    original_loader = DataLoader(original_samples, batch_size=64, shuffle=True)

    # Train the VAE again
    recon_l = []
    kl_l = []
    losses = []
    start_time = time.time()
    for epoch_idx in range(n_epochs):
        for im in tqdm(original_loader):
            im = im.float().to(device)
            optim.zero_grad()
            mean, log_var, out = model(im)
            
            klloss = torch.mean(0.5 * torch.sum(torch.exp(log_var) + mean ** 2 - 1 - log_var, dim=-1))
            reconloss = criterion(out, im)
            loss = reconloss + 0.00001 * klloss
            recon_l.append(reconloss.item())
            losses.append(loss.item())
            kl_l.append(klloss.item())
            loss.backward()
            optim.step()
        print('Finished epoch:{} | Recon Loss : {:.4f} | KL Loss : {:4f}'.format(
            epoch_idx + 1,
            np.mean(recon_l),
            np.mean(kl_l)
        ))
    
    end_time = time.time()
    print("Training time for training loop 3 (without augmentation):", end_time - start_time, "seconds")
    print('Done Training Third Time ...')
    
    # Get the second reconstruction image
    idxs = torch.randint(0, len(mnist_test) - 1, (100,))
    ims = torch.cat([mnist_test[idx][0][None, :] for idx in idxs]).float()

    _, _, generated_im = model(ims)

    ims = (ims + 1) / 2
    generated_im = 1 - (generated_im + 1) / 2
    out = torch.hstack([ims, generated_im])
    output = rearrange(out, 'b c h w -> b () h (c w)')
    grid = torchvision.utils.make_grid(output, nrow=10)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save('reconstructionwithoutaug.png')
    print('Done Reconstruction 3...')


if __name__ == '__main__':
    train_vae()