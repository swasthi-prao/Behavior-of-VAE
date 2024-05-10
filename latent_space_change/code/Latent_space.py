import torch
import cv2
import torchvision
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from einops import rearrange
import os
from torch.optim import Adam
from dataset.mnist_loader import MnistDataset
from torch.utils.data import DataLoader
import os
import skimage

device = torch.device('cpu')

class VAEModel(nn.Module):
    def __init__(self, latent_size):
        super(VAEModel, self).__init__()
        self.latent_size = latent_size
        
        self.common_fc = nn.Sequential(
            nn.Linear(28*28, 196),
            nn.Tanh(),
            nn.Linear(196, 48),
            nn.Tanh(),
        )
        self.mean_fc = nn.Sequential(
            nn.Linear(48, 16),
            nn.Tanh(),
            nn.Linear(16, latent_size)
        )
        
        self.log_var_fc = nn.Sequential(
            nn.Linear(48, 16),
            nn.Tanh(),
            nn.Linear(16, latent_size)
        )
        
        self.decoder_fcs = nn.Sequential(
            nn.Linear(latent_size, 16),
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
        mean, log_var = self.encode(x)
        ## Sampling
        z = self.sample(mean, log_var)
        ## Decoder part
        out = self.decode(z)
        return mean, log_var, out

    def encode(self, x):
        out = self.common_fc(torch.flatten(x, start_dim=1))
        mean = self.mean_fc(out)
        log_var = self.log_var_fc(out)
        return mean, log_var
    
    def sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        z = torch.randn_like(std)
        z = z * std + mean
        return z
    
    def decode(self, z):
        out = self.decoder_fcs(z)
        out = out.reshape((z.size(0), 1, 28, 28))
        return out

def train_vae(latent_size):
    # Create the data set and the data loader
    mnist = MnistDataset('train', im_path='data/train/images')
    mnist_test = MnistDataset('test', im_path='data/test/images')
    mnist_loader = DataLoader(mnist, batch_size=64, shuffle=True, num_workers=0)
    
    # Instantiate the model
    model = VAEModel(latent_size).to(device)
    
    # Specify training parameters
    num_epochs = 10
    optimizer = Adam(model.parameters(), lr=1E-3)
    criterion = torch.nn.MSELoss()
    
    recon_losses = []
    kl_losses = []
    losses = []
    # Run training for 10 epochs
    for epoch_idx in range(num_epochs):
        for im, label in tqdm(mnist_loader):
            im = im.float().to(device)
            optimizer.zero_grad()
            mean, log_var, out = model(im)
            
            cv2.imwrite(f'input{latent_size}.png', 255*((im+1)/2).detach().cpu().numpy()[0, 0])
            cv2.imwrite(f'output{latent_size}.png', 255 * ((out + 1) / 2).detach().cpu().numpy()[0, 0])
            
            kl_loss = torch.mean(0.5* torch.sum(torch.exp(log_var) + mean**2 - 1 -log_var, dim=-1))
            recon_loss = criterion(out, im)
            loss = recon_loss + 0.00001 * kl_loss
            recon_losses.append(recon_loss.item())
            losses.append(loss.item())
            kl_losses.append(kl_loss.item())
            loss.backward()
            optimizer.step()
        print('Finished epoch:{} | Recon Loss : {:.4f} | KL Loss : {:4f}'.format(
            epoch_idx+1,
            np.mean(recon_losses),
            np.mean(kl_losses)
        ))
        
    print('Done Training ...')
    # Run a reconstruction for some sample test images
    idxs = torch.randint(0, len(mnist_test)-1, (100, ))
    ims = torch.cat([mnist_test[idx][0][None, :] for idx in idxs]).float()
    
    _, _, generated_im = model(ims)
    # Calculate SSIM and PSNR
    ssim_values = []
    psnr_values = []
    for i in range(len(ims)):
        input_image = ims[i].permute(1, 2, 0).cpu().numpy()
        output_image = generated_im[i].permute(1, 2, 0).detach().cpu().numpy()
        ssim_value = skimage.metrics.structural_similarity(input_image, output_image, multichannel=True)
        psnr_value = skimage.metrics.peak_signal_noise_ratio(input_image, output_image)
        ssim_values.append(ssim_value)
        psnr_values.append(psnr_value)
        
    avg_ssim = np.mean(ssim_values)
    avg_psnr = np.mean(psnr_values)
    
    print(f'Average SSIM: {avg_ssim}, Average PSNR: {avg_psnr}')
    
    # Save reconstruction grid  
    ims = (ims + 1)/ 2
    generated_im = 1- (generated_im + 1) / 2
    out = torch.hstack([ims, generated_im])
    output = rearrange(out, 'b c h w -> b () h (c w)')
    grid = torchvision.utils.make_grid(output, nrow=10)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(f'reconstruction_latent_{latent_size}.png')
    #print('Done Reconstruction...')
    print(avg_ssim, avg_psnr, kl_loss, recon_loss)
    results.append((avg_ssim, avg_psnr, kl_loss, recon_loss))

results = []
if __name__ == '__main__':
    for latent_size in range(1, 41):
        train_vae(latent_size)