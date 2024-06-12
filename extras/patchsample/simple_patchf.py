import torch
import torch.nn as nn
import numpy as np
from models.helper_functions import *


""" Simplified version of PatchsampleF """
class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.mlps = nn.ModuleList()

    def create_mlp(self, feats):
        for feat in feats:
            input_nc = feat.shape[0]
            mlp = nn.Sequential(
                nn.Linear(input_nc, self.nc),
                nn.LeakyReLU(0.2),
                nn.Linear(self.nc, self.nc)
            ).to(feat.device)
            self.mlps.append(mlp)
        init_net(self, self.init_gain, self.gpu_ids)

    def forward(self, feats, num_patches=64, patch_ids=None):
        if self.use_mlp and len(self.mlps) == 0:
            self.create_mlp(feats)

        return_ids = []
        return_feats = []
        
        for feat_id, feat in enumerate(feats):
            if len(feat.shape) == 3:
                feat = feat.unsqueeze(0)
                
            B, C, H, W = feat.size()
            feat_reshape = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = np.random.permutation(feat_reshape.shape[1])[:num_patches]
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].reshape(-1, C)
            else:
                x_sample = feat_reshape
                patch_id = torch.arange(H * W, device=feat.device)

            if self.use_mlp:
                x_sample = self.mlps[feat_id](x_sample)

            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.view(B, H, W, -1).permute(0, 3, 1, 2)

            return_ids.append(patch_id)
            return_feats.append(x_sample)

        return return_feats, return_ids
