import torch.nn as nn
from models.helper_functions import *

# simple resnet Generator (simplified version of UNSB Resnet Generator)

# Basic Block for Generator
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        padding = get_pad_layer(padding_type)(1) if padding_type != 'zero' else 0
        layers = [
            padding,
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            norm_layer(dim),
            nn.ReLU(True),
            nn.Dropout(0.5) if use_dropout else nn.Identity(),
            padding,
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            norm_layer(dim)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv_block(x)


# Generator
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        model = [get_pad_layer('reflect')(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        
        # Downsampling
        for i in range(2):
            mult = 2 ** i
            model.extend([
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)])
        
        # ResNet blocks
        mult = 2 ** 2
        for i in range(n_blocks):
            model.append(ResnetBlock(ngf * mult, 'reflect', norm_layer, use_dropout))
        
        # Upsampling
        for i in range(2, 0, -1):
            mult = 2 ** i
            model.extend([
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf * mult // 2),
                nn.ReLU(True)])
        
        model.extend([
            get_pad_layer('reflect')(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()])
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)