# Helper Functions
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 


def get_pad_layer(pad_type):
    if pad_type in ['reflect', 'refl']:
        return nn.ReflectionPad2d
    elif pad_type in ['replicate', 'repl']:
        return nn.ReplicationPad2d
    elif pad_type == 'zero':
        return nn.ZeroPad2d
    else:
        raise NotImplementedError(f'Padding type {pad_type} not recognized')

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)
        
def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    return emb
                                  
class AdaptiveLayer(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.style = nn.Linear(style_dim, in_channel * 2)
        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        gamma, beta = self.style(style).chunk(2, 1)
        gamma, beta = gamma.unsqueeze(2).unsqueeze(3), beta.unsqueeze(2).unsqueeze(3)
        return gamma * input + beta

class ConvBlock_cond(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim, kernel_size=3, stride=1, padding=1, use_bias=True, norm_layer=nn.BatchNorm2d, downsample=True):
        super(ConvBlock_cond, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias)
        self.norm = norm_layer(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.downsample = downsample

    def forward(self, x, t_emb):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        if self.downsample:
            out = nn.functional.avg_pool2d(out, kernel_size=2, stride=2)
        return out