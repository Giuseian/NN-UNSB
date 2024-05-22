# Helper Functions
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 
import torch.nn.init as init

# helper functions
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
    

class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.ReLU()):
        super(TimestepEmbedding, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.act = act
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t = t.float()  # Ensure t is of type float
        t = self.fc1(t)
        t = self.act(t)
        t = self.fc2(t)
        return t


# helper functions for PatchSample_F
def init_weights(net, init_gain=0.02, debug=False):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('LayerNorm') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
    if initialize_weights:
        init_weights(net, init_gain=init_gain, debug=debug)
    return net

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out