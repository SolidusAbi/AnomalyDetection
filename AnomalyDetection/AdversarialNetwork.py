import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader
from torch.distributions import MultivariateNormal

from autoencoder import SDAE, SDAE_TYPE
from sklearn import svm

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=True):
        r'''
            Parameters
            ----------
                input_dim: int
                    Indicate the dimensionality of latent space Z

                hidden_dim: int
        '''
        super(Discriminator, self).__init__()
        self.D = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            *( nn.Dropout(.2), nn.ReLU(inplace=True) ) if dropout else ( nn.ReLU(), ),
            nn.Linear(hidden_dim, hidden_dim),
            *( nn.Dropout(.2), nn.ReLU(inplace=True) ) if dropout else ( nn.ReLU(), ),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )  
    def forward(self, x):
        return self.D(x)

class RobustAnomalyDetector():
    def __init__(self, z_dim, dropout=True):
        self.sdae = SDAE([28*28, 1024, 1024, z_dim], SDAE_TYPE.linear, dropout=dropout, 
            activation_func=[nn.ReLU(), nn.ReLU(), nn.Identity()])
        self.Q = self.sdae.encode 
        self.P = self.sdae.decode 

        self.D = Discriminator(z_dim, 512, dropout=dropout)
        # Defining prior as Multivariate Gaussian: $p(z) \sim \mathcal{N}_n(\mu, \sum)$
        self.prior = MultivariateNormal(loc=torch.zeros(z_dim), covariance_matrix=torch.eye(z_dim))

        self.gen_lr = 1e-3
        self.reg_lr = 5e-4

        # Encoder/Decoder optimization
        self.optim_sdae = torch.optim.Adam(self.sdae.parameters(), lr=self.gen_lr)
        #regularizing optimizers
        self.optim_Q_gen = torch.optim.Adam(self.Q.parameters(), lr=self.reg_lr)
        self.optim_D = torch.optim.Adam(self.D.parameters(), lr=self.reg_lr)

        self.recons_criterion = nn.MSELoss()

    def test(self, test_dataset, epoch, tb_writer, device='cpu'):
        self.Q.eval()
        self.P.eval()

        normal_loader = DataLoader(test_dataset[0], batch_size=1024, shuffle=False)
        anomaly_loader = DataLoader(test_dataset[1], batch_size=1024, shuffle=False)
        with torch.no_grad():
            loss_normal = []
            loss_anomaly = []
            for inputs, _ in normal_loader:
                inputs = inputs.to(device).flatten(1)
                z = self.Q(inputs)
                out = self.P(z)
                
                loss_normal.append(self.recons_criterion(out, inputs).item())

            for inputs, _ in anomaly_loader:
                inputs = inputs.to(device).flatten(1)
                z = self.Q(inputs)
                out = self.P(z)
                
                loss_anomaly.append(self.recons_criterion(out, inputs).item())
            
            tb_writer.add_scalar('test/reconstruction/normal', np.array(loss_normal).mean(), epoch)
            tb_writer.add_scalar('test/reconstruction/anomaly',np.array(loss_anomaly).mean(), epoch)

    def refinement(self, train_dataset, device):
        clf = svm.OneClassSVM(nu=0.02, kernel="rbf", gamma=0.1)
        refinement_loader =  DataLoader(train_dataset, 4096, shuffle=False)
        result = None
        self.Q.eval()
        with torch.no_grad():
            for inputs, _ in refinement_loader:
                inputs = inputs.flatten(1).to(device)
                z = self.Q(inputs)
                if result is not None:
                    result = torch.cat([result, z], dim=0)
                else:
                    result = z

        clf.fit(result.cpu())
        anomaly_detection = clf.predict(result.cpu())
        normal_idx = torch.where((torch.tensor(anomaly_detection) == 1))[0]
        return Subset(train_dataset, normal_idx)

    def train(self, train_dataset, batch_size, n_epoch, test_dataset: list, refinament_iter: list, tb_writer, device='cpu'):
        eps = 1e-12
        self.sdae = self.sdae.to(device)
        self.D = self.D.to(device)
        train_loader =  DataLoader(train_dataset, batch_size, shuffle=True)

        for epoch in range(n_epoch):
            recons_loss_ = []
            disc_loss_ = []
            gen_loss_ = []
            self.sdae.train()
            self.D.train()
            for idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.flatten(1)

                self.optim_sdae.zero_grad()
                self.optim_Q_gen.zero_grad()
                self.optim_D.zero_grad()

                out = self.sdae(inputs)
                recon_loss = self.recons_criterion(out, inputs)
                recons_loss_.append(recon_loss.item())

                recon_loss.backward()
                self.optim_sdae.step()

                # Discriminator
                # Prior as an multivariate gaussian function $p(z) \sim \mathcal{N}_n(\mu, \sum)$
                # this is constraining the Z-projection to be normal!
                self.Q.eval()
                z_real_gauss = self.prior.sample((inputs.size(0),)).to(device)
                D_real_gauss = self.D(z_real_gauss)

                z_fake_gauss = self.Q(inputs)
                D_fake_gauss = self.D(z_fake_gauss)

                D_loss = -torch.mean(torch.log(D_real_gauss + eps) + torch.log(1 - D_fake_gauss + eps))
                disc_loss_.append(D_loss.item())

                D_loss.backward()
                self.optim_D.step()

                # Generator
                self.Q.train()
                z_fake_gauss = self.Q(inputs)
                D_fake_gauss = self.D(z_fake_gauss)
                
                G_loss = -torch.mean(torch.log(D_fake_gauss + eps))
                gen_loss_.append(G_loss.item())

                G_loss.backward()
                self.optim_Q_gen.step()
                
            tb_writer.add_scalar('train/reconstruction', np.array(recons_loss_).mean(), epoch)
            tb_writer.add_scalar('train/Discrimator', np.array(disc_loss_).mean(), epoch)
            tb_writer.add_scalar('train/Generator', np.array(gen_loss_).mean(), epoch)

            self.test(test_dataset, epoch, tb_writer)

            if epoch in refinament_iter:
                print('Refinement epoch:{}'.format(epoch))
                train_dataset = self.refinement(train_dataset, device)
                train_loader =  DataLoader(train_dataset, batch_size, shuffle=True)

        self.sdae = self.sdae.cpu()
        self.D = self.D.cpu()

