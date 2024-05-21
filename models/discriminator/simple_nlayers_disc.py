import torch
import torch.nn as nn
from models.helper_functions import * 

# simple Discrminator used with the Injection conditioned simple generator

class D_NLayersMulti(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.layers = nn.ModuleList()
        self.timestep_embedding_transforms = nn.ModuleList()

        kw = 4
        padw = 1

        # First layer
        self.layers.append(nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ))
        self.timestep_embedding_transforms.append(nn.Linear(ndf * 4, ndf))

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.layers.append(nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ))
            self.timestep_embedding_transforms.append(nn.Linear(ndf * 4, ndf * nf_mult))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.layers.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ))
        self.timestep_embedding_transforms.append(nn.Linear(ndf * 4, ndf * nf_mult))

        self.final_layer = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)

    def forward(self, input, t_emb):
        result = input
        for layer, t_emb_transform in zip(self.layers, self.timestep_embedding_transforms):
            result = layer(result)
            t_emb_channel_specific = t_emb_transform(t_emb).view(t_emb.size(0), -1, 1, 1)
            t_emb_channel_specific = t_emb_channel_specific.expand(-1, -1, result.size(2), result.size(3))
            result = result + t_emb_channel_specific  
        return self.final_layer(result)