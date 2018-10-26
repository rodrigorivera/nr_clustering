
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import numpy as np

# Semi-Supervised VAE model

class SSVAE(nn.Module):
    def __init__(self, in_channels, M, L, device, K=12, P=5, mp=3, hidden_lsz=2, channels=10, lstm_sz=10,
                clf_sz=50, is_semi_supervised=True):
        super(SSVAE, self).__init__()
        self.M = M
        self.L = L
        self.device = device
        self.channels = channels
        self.is_semi_supervised = is_semi_supervised
        r = hidden_lsz + M * int(is_semi_supervised)
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=channels, kernel_size=K, padding=P)
        self.mp1 = nn.MaxPool1d(kernel_size=mp, stride=mp, return_indices=False)
        conv_sz = self.conv1.out_channels * ((L + 1 - (K - 2*P)) // mp)
        print('conv size:', conv_sz)
        self.fc21 = nn.Linear(conv_sz, hidden_lsz)
        self.fc22 = nn.Linear(conv_sz, hidden_lsz)
        
        self.fc3 = nn.Linear(r, channels*r)
        print('fc3', r, channels*r)
        self.fc3b = nn.Linear(channels*r, conv_sz)
        self.ump4 = nn.MaxUnpool1d(kernel_size=mp, stride=mp)
        self.ump4_inds_base = torch.tensor(np.arange(mp//2, (L + 1 - (K - 2*P)) - 1, mp)).float()
        self.deconv4 = nn.ConvTranspose1d(in_channels=channels, out_channels=in_channels, kernel_size=K, padding=P)
        self.polish = nn.LSTM(in_channels, lstm_sz, num_layers=1, bidirectional=True)
        self.fc4 = nn.Linear(2*lstm_sz, in_channels)
        
        self.clf_conv1 = nn.Conv1d(in_channels=in_channels, out_channels=channels, kernel_size=K, padding=P)
        self.clf_mp1 = nn.MaxPool1d(kernel_size=mp, stride=mp, return_indices=False)
        self.clf_linear = nn.Linear(conv_sz, clf_sz)
        self.clf_final = nn.Linear(clf_sz, M)

    def encode(self, x):
        h1 = self.mp1(F.leaky_relu(self.conv1(x)))
        h1 = h1.view(h1.shape[0], -1)
        return self.fc21(h1), self.fc22(h1)
    
    def classify(self, x):
        h1 = self.clf_mp1(F.leaky_relu(self.clf_conv1(x)))
        h1 = h1.view(h1.shape[0], -1)
        h2 = F.leaky_relu(self.clf_linear(h1))
        return torch.log_softmax(self.clf_final(h2), dim=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) + 0.1
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, y):
        if self.is_semi_supervised:
            zy = torch.cat((z, y), dim=1)
        else:
            zy = z
        zyb = F.leaky_relu(self.fc3(zy))
        ump4_inds = (torch.ones(z.shape[0], self.channels, len(self.ump4_inds_base)) * self.ump4_inds_base).long().to(self.device)
        h3 = self.ump4(F.leaky_relu(self.fc3b(zyb).unsqueeze(1).view(zy.shape[0], self.deconv4.in_channels, -1)),
                       ump4_inds)
        
        h4 = F.leaky_relu(self.deconv4(h3))
        h4 = h4.transpose(1, 0).transpose(2, 0) 
        h5, _ = self.polish(h4)
        h5 = F.leaky_relu(h5)
        return self.fc4(h5).transpose(1, 0).squeeze()
    
    def forward(self, x, variational_sample=1):
        mu, logvar = self.encode(x.transpose(1, 2))
        y_pred = self.classify(x.transpose(1, 2))
        for i in range(variational_sample):
            z = self.reparameterize(mu, logvar)
            recon_x = self.decode(z, y_pred)
            if i == 0:
                reconstruction = torch.zeros_like(recon_x)
            reconstruction += recon_x
        reconstruction /= variational_sample
        return reconstruction, mu, logvar, y_pred
