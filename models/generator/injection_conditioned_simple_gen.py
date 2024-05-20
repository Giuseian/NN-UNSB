import torch.nn as nn
from models.helper_functions import *

# Conditional ResNet Generation with feature injection

class ConditionalResNetBlock(nn.Module):
    """
    Conditional ResNet block with external conditioning using style and time embeddings.
    """
    def __init__(self, dim, style_dim, time_dim, padding_type, norm_layer, use_dropout):
        super(ConditionalResNetBlock, self).__init__()
        self.style_dim = style_dim
        self.time_dim = time_dim
        self.padding_type = padding_type
        self.use_dropout = use_dropout

        # First conv layer with padding
        if padding_type == 'reflect':
            self.pad1 = nn.ReflectionPad2d(1)
        elif padding_type == 'replicate':
            self.pad1 = nn.ReplicationPad2d(1)
        else:
            self.pad1 = nn.ZeroPad2d(1)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, bias=False)
        self.norm1 = norm_layer(dim)
        self.inject1 = FeatureInjection(style_dim, dim)
        self.inject2 = FeatureInjection(time_dim, dim)
        self.relu = nn.ReLU(True)

        # Optional dropout
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()

        # Second conv layer with padding
        if padding_type == 'reflect':
            self.pad2 = nn.ReflectionPad2d(1)
        elif padding_type == 'replicate':
            self.pad2 = nn.ReplicationPad2d(1)
        else:
            self.pad2 = nn.ZeroPad2d(1)

        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, bias=False)
        self.norm2 = norm_layer(dim)

    def forward(self, x, style, time):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.inject1(out, style)
        out = self.inject2(out, time)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return x + out

class FeatureInjection(nn.Module):
    """
    Injects an external feature into the feature maps.
    """
    def __init__(self, external_dim, feature_dim):
        super(FeatureInjection, self).__init__()
        self.fc = nn.Linear(external_dim, feature_dim)

    def forward(self, x, external_feature):
        external_feature = self.fc(external_feature)
        # Expand dimensions to match feature map dimensions (batch, channels, height, width)
        external_feature = external_feature.view(external_feature.shape[0], -1, 1, 1)
        return x + external_feature

class ResNetGeneratorWithCondBlocks(nn.Module):
    """
    ResNet-based generator that includes downsampling, conditional ResNet blocks, and upsampling layers.
    
    """
    def __init__(self, input_nc, output_nc, ngf=64, style_dim=100, time_dim=100, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        super(ResNetGeneratorWithCondBlocks, self).__init__()

        # Initial convolutional block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # Downsampling layers
        for i in range(2):
            mult = 2 ** i
            model.extend([
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ])

        # Conditional ResNet blocks
        mult = 2 ** 2
        for i in range(n_blocks):
            model.append(ConditionalResNetBlock(ngf * mult, style_dim, time_dim, padding_type, norm_layer, use_dropout))

        # Upsampling layers
        for i in range(2):
            mult = 2 ** (2 - i)
            model.extend([
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ])

        # Final convolutional block
        model.extend([
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ])

        self.model = nn.Sequential(*model)

    def forward(self, input, style, time):
        x = input
        for layer in self.model:
            if isinstance(layer, ConditionalResNetBlock):
                x = layer(x, style, time)
            else:
                x = layer(x)
        return x
