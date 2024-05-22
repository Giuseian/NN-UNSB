from models.helper_functions import *
import numpy as np 

class PatchSampleF(nn.Module):
    # using different patch_ids for different images in the batch 
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[1]):
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            print("input_nc", input_nc.shape)
            mlp = nn.Sequential(
                nn.Linear(input_nc, self.nc),
                nn.LeakyReLU(0.2),
                nn.Linear(self.nc, self.nc)
            )
            mlp.cuda()
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)

        return_ids = []
        return_feats = []

        for feat_id, feat in enumerate(feats):
            B, C, H, W = feat.shape
            feat_reshape = feat.permute(0, 2, 3, 1).reshape(B, -1, C)

            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = [np.random.permutation(feat_reshape.shape[1])[:num_patches] for _ in range(B)]
                patch_id = [torch.tensor(pid, dtype=torch.long, device=feat.device) for pid in patch_id]
                x_sample = [feat_reshape[i, pid, :] for i, pid in enumerate(patch_id)]
                x_sample = torch.cat(x_sample, dim=0)
            else:
                x_sample = feat_reshape.reshape(-1, C)
                patch_id = [torch.tensor([], dtype=torch.long, device=feat.device) for _ in range(B)]

            if self.use_mlp:
                mlp = getattr(self, f'mlp_{feat_id}')
                x_sample = mlp(x_sample)

            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.view(B, H, W, -1).permute(0, 3, 1, 2)

            return_ids.extend(patch_id)
            return_feats.append(x_sample)

        return return_feats, return_ids



