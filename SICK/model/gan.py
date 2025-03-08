import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np
from utility.data import get_loader
class GANs:
    def __init__(
        self,
        train_ds,
        batch_size,
        X_train,
        y_train,
        train_y_full,
        latent_dim,
        condition_dim,
        device,
        gan_epochs,
        ):
        self.train_ds = train_ds
        self.batch_size = batch_size
        self.X_train = X_train
        self.y_train = y_train
        self.train_y_full = train_y_full
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.device = device
        self.gan_epochs = gan_epochs
    def generate(self, size, generation_size):
        train_subset = Subset(self.train_ds, range(size))
        gan_loader = get_loader(train_subset, batch_size=self.batch_size, shuffle=True)
        input_dim = self.X_train.shape[1]  
        unique_labels = np.unique(self.train_y_full)
        num_classes_gan = len(unique_labels)

        # Create and train GAN model
        generator = Generator(
            latent_dim=self.latent_dim,
            condition_dim=self.condition_dim,
            num_classes=num_classes_gan,
            start_dim=self.latent_dim * 2,
            n_layer=3,
            output_dim=input_dim
        ).to(self.device)
        discriminator = Discriminator(
            condition_dim=self.condition_dim,
            num_classes=num_classes_gan,
            start_dim=256,
            n_layer=3,
            input_dim=input_dim
        ).to(self.device)

        manager = GANManager(generator, discriminator, gan_loader, self.gan_epochs, self.latent_dim, self.device)
        manager.train()
        synthetic_x, synthetic_y = manager.generate(unique_labels, generation_size)
        return synthetic_x, synthetic_y


class GANManager:
    def __init__(
            self,
            generator: nn.Module,
            discriminator: nn.Module,
            gan_loader: DataLoader,
            gan_epochs: int,
            latent_dim: int,
            device: torch.device,
            verbosity: int = 1  # Default verbosity level is 1 (basic logging)
        ):
        self.generator = generator
        self.discriminator = discriminator
        self.gan_loader = gan_loader
        self.gan_epochs = gan_epochs
        self.device = device
        self.latent_dim = latent_dim
        self.verbosity = verbosity  # Level of verbosity for printing logs
    def generate(self, unique_labels, generation_size):
            
        synthetic_data_list = []
        synthetic_labels_list = []
        for lab in unique_labels:
            lab_tensor = torch.full((generation_size,), lab, dtype=torch.long, device=self.device)
            z = torch.randn(generation_size, self.latent_dim, device=self.device)
            synth = self.generator(z, lab_tensor).cpu().detach().numpy()
            synthetic_data_list.append(synth)
            synthetic_labels_list.append(np.full((generation_size,), lab))
                
        synthetic_x = np.concatenate(synthetic_data_list, axis=0)
        synthetic_y = np.concatenate(synthetic_labels_list, axis=0)
        return synthetic_x, synthetic_y
    
    def train(self) -> list:
        '''
        Train the GAN model and return a list of losses (D_loss, G_loss).
        
        Arguments:
        - verbosity: (int) controls the verbosity of the training output
            - 0: silent (no output)
            - 1: basic output (epoch-wise losses)
            - 2: detailed output (including per-batch losses)
        '''
        self.generator.train()
        self.discriminator.train()

        adversarial_loss = nn.BCELoss().to(self.device)
        optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        losses = []  # List to store losses

        # Loop through the number of epochs
        for epoch in range(self.gan_epochs):
            d_loss_epoch, g_loss_epoch = 0.0, 0.0  # Initialize epoch losses

            # Loop through each batch in the DataLoader
            for embeddings, labels in self.gan_loader:
                embeddings = embeddings.to(self.device)  # Move data to device (GPU/CPU)
                labels = labels.clone().detach().to(self.device).long()
                b_size = embeddings.size(0)  # Get batch size

                valid = torch.ones(b_size, 1, device=self.device)  # Labels for real data
                fake = torch.zeros(b_size, 1, device=self.device)  # Labels for fake data

                # Train Generator
                optimizer_G.zero_grad()  # Zero the gradients for the generator
                z = torch.randn(b_size, self.latent_dim, device=self.device)  # Generate random latent vector
                gen_data = self.generator(z, labels)  # Generate fake data using the generator
                g_loss = adversarial_loss(self.discriminator(gen_data, labels), valid)  # Calculate generator loss
                g_loss.backward()  # Backpropagate the generator loss
                optimizer_G.step()  # Update generator parameters

                # Train Discriminator
                optimizer_D.zero_grad()  # Zero the gradients for the discriminator
                real_loss = adversarial_loss(self.discriminator(embeddings, labels), valid)  # Real data loss
                fake_loss = adversarial_loss(self.discriminator(gen_data.detach(), labels), fake)  # Fake data loss
                d_loss = (real_loss + fake_loss) / 2  # Average loss for discriminator
                d_loss.backward()  # Backpropagate the discriminator loss
                optimizer_D.step()  # Update discriminator parameters

                d_loss_epoch += d_loss.item()  # Accumulate discriminator loss
                g_loss_epoch += g_loss.item()  # Accumulate generator loss

                # Verbosity: print detailed per-batch loss if verbosity level is 2
                if self.verbosity == 2:
                    print(f"  [Batch] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")

            # Verbosity: print epoch-wise losses based on verbosity level
            if self.verbosity > 0:
                print(f"[GAN Epoch {epoch+1}/{self.gan_epochs}] D loss: {d_loss_epoch/len(self.gan_loader):.4f}, G loss: {g_loss_epoch/len(self.gan_loader):.4f}")

            losses.append((d_loss_epoch / len(self.gan_loader), g_loss_epoch / len(self.gan_loader)))  # Store average losses for each epoch

        self.generator.eval()
        self.discriminator.eval()

        return losses


class Generator(nn.Module):
    def __init__(self, latent_dim=100, condition_dim=10, num_classes=10, start_dim=128, n_layer=3, output_dim=512):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, condition_dim)
        input_dim = latent_dim + condition_dim
        layers = []

        layers.append(nn.Linear(input_dim, start_dim))
        layers.append(nn.ReLU(inplace=True))
        current_dim = start_dim

        for i in range(1, n_layer):
            next_dim = current_dim * 2
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.BatchNorm1d(next_dim, momentum=0.8))
            layers.append(nn.ReLU(inplace=True))
            current_dim = next_dim
            
        layers.append(nn.Linear(current_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, z, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([z, label_embedding], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, condition_dim=10, num_classes=10, start_dim=128, n_layer=3, input_dim=512):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, condition_dim)
        # input_dim + condition_dim
        input_dim = input_dim + condition_dim
        hidden_dim = start_dim * (2 ** (n_layer - 1))
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        current_dim = hidden_dim
        for i in range(1, n_layer):
            next_dim = current_dim // 2
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_dim = next_dim
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([x, label_embedding], dim=1)
        return self.model(x)