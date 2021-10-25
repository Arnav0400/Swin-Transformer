import torch
from torch.nn import functional as F

class PruningLoss(torch.nn.Module):
    def __init__(self, base_criterion, device, attn_w=0.0001, mlp_w=0.0001):
        super().__init__()
        self.base_criterion = base_criterion
        self.w1 = attn_w
        self.w2 = mlp_w
        self.device = device

    def forward(self, inputs, outputs, labels, model):
        base_loss = self.base_criterion(inputs, outputs, labels)
        sparsity_loss_attn, sparsity_loss_mlp = model.module.get_sparsity_loss(self.device)
        return  base_loss + self.w1*sparsity_loss_attn + self.w2*sparsity_loss_mlp