from models.helper_functions import *

# our dynamic PatchF  

class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
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
            input_nc = feat.shape[0]
            mlp = nn.Sequential(
                nn.Linear(input_nc, self.nc),
                nn.LeakyReLU(0.2),
                nn.Linear(self.nc, self.nc)
            )
            mlp.cuda()
            setattr(self, f'mlp_{mlp_id}', mlp)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)

        return_feats = []
        return_ids = []

        for feat_id, feat in enumerate(feats):
            # Add batch dimension if missing
            if len(feat.shape) == 3:
                feat = feat.unsqueeze(0)

            B, C, H, W = feat.shape
            feat_reshape = feat.permute(0, 2, 3, 1).reshape(B, -1, C)  # Reshape to [B, H*W, C]

            if num_patches > 0:
                if patch_ids is not None and len(patch_ids) > feat_id:
                    current_patch_ids = patch_ids[feat_id]
                else:
                    # Generate random patch indices if none provided
                    current_patch_ids = [torch.randperm(feat_reshape.shape[1])[:num_patches].to(feat.device) for _ in range(B)]
                current_patch_ids = [torch.tensor(pid, dtype=torch.long, device=feat.device) for pid in current_patch_ids]
                # Sampling patches
                x_sample = torch.cat([feat_reshape[b, pid, :] for b, pid in enumerate(current_patch_ids)], dim=0)
                return_ids.append(current_patch_ids)
            else:
                x_sample = feat_reshape.reshape(-1, C)
                current_patch_ids = [torch.tensor([], dtype=torch.long, device=feat.device) for _ in range(B)]
                return_ids.append(current_patch_ids)

            if self.use_mlp:
                mlp = getattr(self, f'mlp_{feat_id}')
                x_sample = mlp(x_sample)

            x_sample = self.l2norm(x_sample)

            return_feats.append(x_sample)

        # Since we add patches for each batch, we must handle the concatenation properly
        if num_patches == 0:
            return_feats = [f.view(B, H, W, -1).permute(0, 3, 1, 2) for f in return_feats]

        return return_feats, return_ids