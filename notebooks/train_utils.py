import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.cm as cm
from sklearn.metrics import roc_auc_score

from model import SSVAE

def loss_component_KLD(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

def semi_supervised_cross_entorpy(y, log_p):
    # for missing labels all one-hot-y is 0
    return -torch.sum(y * log_p)/(torch.sum(y) + 1e-5)

def mse_loss(rx, x):
    return torch.sum((rx - x)**2)/(x.shape[0] * x.shape[2])

def torch_var(x):
    return torch.autograd.Variable(torch.from_numpy(np.array(x)).float())

class SemiSupervisedLoss(torch.nn.Module):
    
    def __init__(self, mse_weight=100, ce_weight=600, kl_weight=0.5, is_semi_supervised=True):
        super(SemiSupervisedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.ce_weight = ce_weight * int(is_semi_supervised)
        self.kl_weight = kl_weight
        
    def forward(self, recon_x, x, y_true, y_pred, mu, logvar):
        MSE = mse_loss(recon_x, x) * self.mse_weight
        CrossEntropy = semi_supervised_cross_entorpy(y_true, y_pred)
        return MSE + CrossEntropy*self.ce_weight + loss_component_KLD(mu, logvar)*self.kl_weight

def loss_function(SS_MOD, recon_x, x, y_true, y_pred, mu, logvar):
    return SemiSupervisedLoss(is_semi_supervised=(SS_MOD is not None))(recon_x, x, y_true, y_pred, mu, logvar)

def one_hot_encoding(y, num_labels):
    one_hot_labels = torch.zeros(len(y), num_labels + 1)
    one_hot_labels.scatter_(1, torch.tensor(y).view(-1, 1), 1)
    return one_hot_labels[:, :-1]

def train(SS_MOD, train_loader, model, optimizer, epoch, num_labels, device, max_batches=2, VERBOSE=False):
    model.train()
    train_loss = 0
    train_mse_loss = 0
    crossentopy_loss = 0
    crossentopy_loss_full = 0
    kld = 0
    for batch_idx, (data, y_true, y_full) in enumerate(train_loader):
        one_hot_labels_full = one_hot_encoding(y_full, num_labels).to(device)
        if SS_MOD == 'full':
            one_hot_labels = one_hot_encoding(y_full, num_labels)
        elif SS_MOD == 'semi':
            one_hot_labels = one_hot_encoding(y_true, num_labels)
        else:
            one_hot_labels = one_hot_encoding(y_true, num_labels) * 0
        data = data.to(device)
        cur_batch_size = len(data)
        one_hot_labels = one_hot_labels.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, y_pred = model(data)
        loss = loss_function(SS_MOD, recon_batch, 
                             data[:, :recon_batch.shape[1]],
                             one_hot_labels, y_pred,
                             mu, logvar)
        if VERBOSE:
            train_mse_loss += mse_loss(recon_batch, data[:, :recon_batch.shape[1]])
            if y_pred is not None:
                crossentopy_loss += semi_supervised_cross_entorpy(one_hot_labels, y_pred)
                crossentopy_loss_full += semi_supervised_cross_entorpy(one_hot_labels_full, y_pred)
            kld += loss_component_KLD(mu, logvar)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx >= max_batches:
            break
    train_loss /= (batch_idx + 1)
    if VERBOSE:
        train_mse_loss /= (batch_idx + 1)
        crossentopy_loss /= (batch_idx + 1)
        crossentopy_loss_full /= (batch_idx + 1)
        kld /= (batch_idx + 1)
        if epoch % 10 == 0:
            print('====> Train set loss: {:.4f}, MSE: {:.4f}, CE: {:.4f}, CE(full): {:.4f}, KL: {:.4f}'.format(train_loss, train_mse_loss, crossentopy_loss, crossentopy_loss_full, kld))
    
    return train_loss


def safe_roc_auc_score(y, p):
    if len(np.unique(y)) == 1:
        return np.nan
    else:
        return roc_auc_score(y, p)

def test(SS_MOD, test_loader, model, epoch, num_labels, device, target_features, prefix, recon_samples=50, VERBOSE=False):
    model.eval()
    test_loss = 0
    test_iwae = 0
    test_mse_loss = 0
    crossentopy_loss = 0
    crossentopy_loss_full = 0
    kld = 0
    full_vals = []
    pred_vals = []
    with torch.no_grad():
        for batch_idx, (data, y_true, y_full) in enumerate(test_loader):
            if SS_MOD == 'full':
                one_hot_labels = one_hot_encoding(y_full,num_labels)
            elif SS_MOD == 'semi':
                one_hot_labels = one_hot_encoding(y_true,num_labels)
            else:
                one_hot_labels = one_hot_encoding(y_true,num_labels) * 0
            one_hot_labels_full = one_hot_encoding(y_full,num_labels).to(device)
            full_vals.extend(one_hot_labels_full.to(torch.device("cpu")).data.numpy().tolist())
            data = data.to(device)
            one_hot_labels = one_hot_labels.to(device)
            cur_batch_size = len(data)
            for r in range(recon_samples):
                recon_batch_part, mu, logvar, y_pred = model(data)
                if r == 0:
                    recon_batch = recon_batch_part
                else:
                    recon_batch += recon_batch_part
                # test_iwae += loss_function(SS_MOD, recon_batch_part, 
                #                        data[:, :recon_batch_part.shape[1]], 
                #                        one_hot_labels, y_pred,
                #                        mu, logvar)
            recon_batch /= recon_samples
            test_iwae /= recon_samples
            if y_pred is not None:
                pred_vals.extend(np.exp(y_pred.to(torch.device("cpu")).data.numpy()).tolist())
            test_loss += loss_function(SS_MOD, recon_batch, 
                                       data[:, :recon_batch.shape[1]], 
                                       one_hot_labels, y_pred,
                                       mu, logvar)
            test_mse_loss += mse_loss(recon_batch, data[:, :recon_batch.shape[1]])
            if True:
                if y_pred is not None:
                    crossentopy_loss += semi_supervised_cross_entorpy(one_hot_labels, y_pred)
                    crossentopy_loss_full += semi_supervised_cross_entorpy(one_hot_labels_full, y_pred)
                kld += loss_component_KLD(mu, logvar)
                if batch_idx == 0 and VERBOSE:
                    n = min(data.size(0), 8)
                    if y_pred is None:
                        y_pred = torch.zeros(n, num_labels)
                    plot_comparison_batch(data[:n, :recon_batch.shape[1]].to(torch.device("cpu")), 
                                          recon_batch[:n].to(torch.device("cpu")), 
                                          y_full[:n].to(torch.device("cpu")), 
                                          y_pred[:n].to(torch.device("cpu")), 
                             'results/'+prefix+'_recon_' + str(epoch) + '.png', target_features)

    test_loss /= (batch_idx + 1)
    test_mse_loss /= (batch_idx + 1)
    if True:    
        crossentopy_loss /= (batch_idx + 1)
        crossentopy_loss_full /= (batch_idx + 1)
        kld /= (batch_idx + 1)
        print('====> Test set loss: {:.4f}, MSE: {:.4f}, CE: {:.4f}, CE(full): {:.4f}, KL: {:.4f}'.format(test_loss, test_mse_loss, crossentopy_loss, crossentopy_loss_full, kld))
        if len(pred_vals) == len(full_vals):
            full_vals = np.array(full_vals)
            pred_vals = np.array(pred_vals)
            for c in range(num_labels):
                print(safe_roc_auc_score(full_vals[:, c], pred_vals[:, c]))
    return test_loss, test_mse_loss, test_iwae


def plot_comparison(x, x_recon, target_features):
    plt.figure(figsize=(5, (1 + 1.25*x.shape[1])))
    for f in range(x.shape[1]):
        plt.subplot(x.shape[1], 1, f + 1)
        plt.plot(x[:, f].data.numpy().ravel())
        plt.plot(x_recon[:, f].data.numpy().ravel())
        plt.ylabel(target_features[f])
    plt.tight_layout()

def plot_comparison_batch(X, X_recon, y_true, y_pred, file_path, target_features):
    plt.figure(figsize=(5*len(X), 1.25*X.shape[2]))
    for i in range(len(X)):
        x_recon = X_recon[i]
        x = X[i, :len(x_recon)]
        
        for f in range(x.shape[1]):
            plt.subplot(x.shape[1], len(X), f*len(X) + i + 1)
            if f == 0:
                mse = nn.MSELoss(reduction='sum')(x.view(-1, 1), x_recon.view(-1, 1)).item()/len(x)
                plt.title('example {}\nMSE: {:.3f}, y_true: {}\n y_pred: '.format(i + 1, mse, y_true[i]) + ', '.join(['%.4f' % np.exp(y) for y in y_pred[i].data.numpy()]))
            plt.plot(x[:, f].data.numpy().ravel())
            plt.plot(x_recon[:, f].data.numpy().ravel())
            plt.ylabel(target_features[f])
            plt.ylim(X[:, :, f].min().item(), X[:, :, f].max().item())
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight', dpi=100)
    plt.close()

def fit_model_report(SS_MOD, VERBOSE, target_features, num_labels, latent_dim, L, device, epochs, train_loader, test_loader, prefix, test_rate=100, channels=5):
    model = SSVAE(in_channels=len(target_features), M=num_labels, L=L, device=device, K=20, P=10, hidden_lsz=latent_dim, channels=channels, 
              mp=3, lstm_sz=10, clf_sz=10, is_semi_supervised=(SS_MOD is not None)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    if VERBOSE:
        print('number of model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    train_losses = []
    test_losses = []
    mse_losses = []
    test_iwaes = []
    for epoch in range(1, epochs + 1):
        t = train(SS_MOD, train_loader, model, optimizer, epoch, num_labels, device, max_batches=30, VERBOSE=VERBOSE)
        if epoch % test_rate == 0:
            print('epoch:', epoch)
            train_losses.append(t)
            test_loss, test_mse_loss, test_iwae = test(SS_MOD, test_loader, model, epoch, num_labels, device, target_features, prefix, VERBOSE=VERBOSE)
            test_losses.append(test_loss)
            mse_losses.append(test_mse_loss)
            test_iwaes.append(test_iwae)
    
    return model, train_losses, test_losses, mse_losses, test_iwaes