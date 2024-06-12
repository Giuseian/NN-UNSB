from torch.utils.data import DataLoader
from preprocessing.dataset import ImageDataset
from models.generator.adaptive_conditioned_gen import *
from models.discriminator.ncsn_discriminator import *
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path_trainA = 'datasets/horse2zebra/trainA'
train_datasetA = ImageDataset(img_dir=path_trainA)
train_dataloaderA = DataLoader(train_datasetA, batch_size=1, shuffle=True)

path_trainB = 'datasets/horse2zebra/trainB'
train_datasetB = ImageDataset(img_dir=path_trainB)
train_dataloaderB = DataLoader(train_datasetB, batch_size=1, shuffle=True)

# Initialize generators and discriminators
gen_A_to_B = ResnetGenerator_cond(input_nc=3, output_nc=3, ngf=64, n_blocks=9, norm_layer=nn.InstanceNorm2d).to(device)
gen_B_to_A = ResnetGenerator_cond(input_nc=3, output_nc=3, ngf=64, n_blocks=9, norm_layer=nn.InstanceNorm2d).to(device)
disc_A = NLayerDiscriminator_ncsn_new(input_nc=3, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d).to(device)
disc_B = NLayerDiscriminator_ncsn_new(input_nc=3, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d).to(device)

# Define optimizers
optimizer_gen = optim.Adam(list(gen_A_to_B.parameters()) + list(gen_B_to_A.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_disc_A = optim.Adam(disc_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_disc_B = optim.Adam(disc_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Define loss function
criterion = nn.BCEWithLogitsLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for i, (real_images_A,real_images_B) in enumerate(zip(train_dataloaderA, train_dataloaderB)):
        # Move real images to the device
        real_images_A = real_images_A.to(device)
        real_images_B = real_images_B.to(device)
        
        # Sample random time embeddings and latent vectors
        t_emb_A = torch.randint(0, 1000, (real_images_A.size(0),), device=device)  # Example time embeddings for domain A
        z_A = torch.randn(real_images_A.size(0), 4 * 64, device=device)  # Example latent vectors for domain A
        t_emb_B = torch.randint(0, 1000, (real_images_B.size(0),), device=device)  # Example time embeddings for domain B
        z_B = torch.randn(real_images_B.size(0), 4 * 64, device=device)  # Example latent vectors for domain B

    
        #  TRAIN DISCRIMINATOR A
        
        # Zero the parameter gradients
        optimizer_disc_A.zero_grad()
        
        # Forward pass real images from domain A through disc_A
        output_real_A = disc_A(real_images_A, t_emb_A)
        real_labels_A = torch.ones_like(output_real_A, device=device)  # Match shape to discriminator output
        loss_D_real_A = criterion(output_real_A, real_labels_A)
        
        # Generate fake images for domain A to B translation and forward pass through disc_A
        fake_images_B = gen_A_to_B(real_images_A, t_emb_A, z_A)
        output_fake_B = disc_A(fake_images_B.detach(), t_emb_A)
        fake_labels_A = torch.zeros_like(output_fake_B, device=device)  # Match shape to discriminator output
        loss_D_fake_A = criterion(output_fake_B, fake_labels_A)
        
        # Total discriminator A loss
        loss_D_A = (loss_D_real_A + loss_D_fake_A) / 2
        
        # Backward pass and optimize
        loss_D_A.backward()
        optimizer_disc_A.step()
        

        #  TRAIN DISCRIMINATOR B
        
        # Zero the parameter gradients
        optimizer_disc_B.zero_grad()
        
        # Forward pass real images from domain B through disc_B
        output_real_B = disc_B(real_images_B, t_emb_B)
        real_labels_B = torch.ones_like(output_real_B, device=device)  # Match shape to discriminator output
        loss_D_real_B = criterion(output_real_B, real_labels_B)
        
        # Generate fake images for domain B to A translation and forward pass through disc_B
        fake_images_A = gen_B_to_A(real_images_B, t_emb_B, z_B)
        output_fake_A = disc_B(fake_images_A.detach(), t_emb_B)
        fake_labels_B = torch.zeros_like(output_fake_A, device=device)  # Match shape to discriminator output
        loss_D_fake_B = criterion(output_fake_A, fake_labels_B)
        
        # Total discriminator B loss
        loss_D_B = (loss_D_real_B + loss_D_fake_B) / 2
        
        # Backward pass and optimize
        loss_D_B.backward()
        optimizer_disc_B.step()
                
        
        # TRAIN GENERATORS
        
        # Zero the parameter gradients
        optimizer_gen.zero_grad()
        
        # Forward pass fake images through respective discriminators to compute adversarial loss
        output_fake_B = disc_A(fake_images_B, t_emb_A)
        output_fake_A = disc_B(fake_images_A, t_emb_B)
        
        # Compute generator losses
        adv_loss_A = criterion(output_fake_B, real_labels_A)
        adv_loss_B = criterion(output_fake_A, real_labels_B)

        # Compute cycle consistency losses
        reconstructed_images_A = gen_B_to_A(fake_images_B, t_emb_A, z_A)
        reconstructed_images_B = gen_A_to_B(fake_images_A, t_emb_B, z_B)
        cycle_loss_A = criterion(reconstructed_images_A, real_images_A)
        cycle_loss_B = criterion(reconstructed_images_B, real_images_B)

        # Compute total generator losses
        total_gen_loss_A = adv_loss_A + 10 * cycle_loss_A
        total_gen_loss_B = adv_loss_B + 10 * cycle_loss_B

        # Backward pass and optimize generators
        total_gen_loss_A.backward(retain_graph=True)
        total_gen_loss_B.backward()
        optimizer_gen.step()

        # Print losses
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_dataloaderA)}]")
            print(f"Discriminator A Loss: {loss_D_A.item()}, Discriminator B Loss: {loss_D_B.item()}")
            print(f"Generator A Loss: {total_gen_loss_A.item()}, Generator B Loss: {total_gen_loss_B.item()}")
            
        # Print fake images generated at the end of each epoch
        with torch.no_grad():
            fake_images_A = gen_B_to_A(real_images_B, t_emb_B, z_B)
            fake_images_B = gen_A_to_B(real_images_A, t_emb_A, z_A)

        # Convert tensors to numpy arrays
        fake_images_A = fake_images_A.cpu().numpy()
        fake_images_B = fake_images_B.cpu().numpy()

        # Plot fake images
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(np.transpose(fake_images_A[0], (1, 2, 0)))
        plt.title('Fake Images A to B')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(np.transpose(fake_images_B[0], (1, 2, 0)))
        plt.title('Fake Images B to A')
        plt.axis('off')

        plt.show()
