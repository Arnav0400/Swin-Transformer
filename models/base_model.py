import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.prunable_modules = []

    def calculate_prune_threshold(self, Vc_attn, Vc_mlp):
        zetas_attn, zetas_mlp = self.give_zetas()
        zetas_attn = sorted(zetas_attn)
        zetas_mlp = sorted(zetas_mlp)
        prune_threshold_attn = zetas_attn[int((1.-Vc_attn)*len(zetas_attn))]
        prune_threshold_mlp = zetas_mlp[int((1.-Vc_mlp)*len(zetas_mlp))]
        return prune_threshold_attn, prune_threshold_mlp
    
    def calculate_global_prune_threshold(self, Vc):
        zetas_attn, zetas_mlp = self.give_zetas()
        zetas_all = zetas_attn + zetas_mlp
        zetas_all = sorted(zetas_all)
        prune_threshold = zetas_all[int((1.-Vc)*len(zetas_all))]
        return prune_threshold
    
    def n_remaining(self, m):
        return (m.pruned_zeta if m.is_pruned else m.get_zeta()).sum()
    
    def is_all_pruned(self, m):
        return self.n_remaining(m) == 0
    
    def get_remaining(self):
        """return the fraction of active zeta""" 
        n_rem_attn = 0
        n_total_attn = 0
        n_rem_mlp = 0
        n_total_mlp = 0
        for l_block in self.prunable_modules:
            if hasattr(l_block, 'num_heads'):
                attn = self.n_remaining(l_block)
                n_rem_attn += attn
                n_total_attn += l_block.num_gates*l_block.num_heads
            else:
                n_rem_mlp += self.n_remaining(l_block)
                n_total_mlp += l_block.num_gates
        return n_rem_attn/n_total_attn, n_rem_mlp/n_total_mlp

    def get_sparsity_loss(self, device):
        loss_attn = torch.FloatTensor([]).to(device)
        loss_mlp = torch.FloatTensor([]).to(device)
        for l_block in self.prunable_modules:
            if hasattr(l_block, 'num_heads'):
                zeta_attn = l_block.get_zeta()
                loss_attn = torch.cat([loss_attn, torch.abs(zeta_attn.view(-1))])
            else:
                loss_mlp = torch.cat([loss_mlp, torch.abs(l_block.get_zeta().view(-1))])
        return torch.sum(loss_attn).to(device), torch.sum(loss_mlp).to(device)

    def give_zetas(self):
        zetas_attn = []
        zetas_mlp = []
        for l_block in self.prunable_modules:
            if hasattr(l_block, 'num_heads'):
                zetas_attn.append(l_block.get_zeta().cpu().detach().reshape(-1).numpy().tolist())
            else:
                zetas_mlp.append(l_block.get_zeta().cpu().detach().reshape(-1).numpy().tolist())
        zetas_attn = [z for k in zetas_attn for z in k ]
        zetas_mlp = [z for k in zetas_mlp for z in k ]
        return zetas_attn, zetas_mlp

    def plot_zt(self):
        """plots the distribution of zeta_t and returns the same"""
        zetas_attn, zetas_mlp = self.give_zetas()
        zetas = zetas_attn + zetas_mlp
        exactly_zeros = np.sum(np.array(zetas)==0.0)
        exactly_ones = np.sum(np.array(zetas)==1.0)
        plt.hist(zetas)
        plt.show()
        return exactly_zeros, exactly_ones

    def prune(self, Vc_attn, Vc_mlp, Vc_patch, constrain_patch=True):
        """prunes the network to make zeta_t exactly 1 and 0"""
        thresh_attn, thresh_mlp = self.calculate_prune_threshold(Vc_attn, Vc_mlp)
                
        for l_block in self.prunable_modules:
            if hasattr(l_block, 'num_heads'):
                l_block.prune(thresh_attn)
            else:
                l_block.prune(thresh_mlp)
        problem = self.check_abnormality()
        return thresh_attn, thresh_mlp, problem
    
    def prune_global(self, Vc_attn_mlp, Vc_patch, constrain_patch=True):
        """prunes the network to make zeta_t exactly 1 and 0"""
        thresh = self.calculate_global_prune_threshold(Vc_attn_mlp)
                
        for l_block in self.prunable_modules:
            if hasattr(l_block, 'num_heads'):
                l_block.prune(thresh)
            else:
                l_block.prune(thresh)
        problem = self.check_abnormality()
        return thresh, problem

    def correct_require_grad(self, w1, w2):
        if w1==0:
            for l_block in self.prunable_modules:
                if hasattr(l_block, 'num_heads'):
                    l_block.zeta.requires_grad = False
        if w2==0:
            for l_block in self.prunable_modules:
                if not hasattr(l_block, 'num_heads'):
                    l_block.zeta.requires_grad = False

    def unprune(self):
        for l_block in self.prunable_modules:
            l_block.unprune()
    
    def get_channels(self):
        active_channels_attn = []
        active_channels_mlp = []
        for l_block in self.prunable_modules:
            if hasattr(l_block, 'num_heads'):
                active_channels_attn.append(l_block.pruned_zeta.numpy())
            else:
                active_channels_mlp.append(l_block.pruned_zeta.sum().item())
        return np.squeeze(np.array(active_channels_attn)), np.array(active_channels_mlp)

    def get_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pruned_params = total_params
        for l_block in self.prunable_modules:
            pruned_params-=l_block.get_params_count()[0]
            pruned_params+=l_block.get_params_count()[1]
        return total_params, pruned_params.item()
    
    def check_abnormality(self):
        isbroken = self.check_if_broken()
        if isbroken:
            return 'broken'
        
    def check_if_broken(self):
        for attn in self.prunable_modules:
            if attn.pruned_zeta.sum()==0:
                return True
        return False