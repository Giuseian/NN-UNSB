from torch import nn
import torch
from models.helper_functions import *

# Discriminator with conditional convolution blocks for processing input images
class NLayerDiscriminator_ncsn_new(nn.Module):
    """Discriminator that uses conditional convolution blocks to process input images."""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Initialize the discriminator with conditional convolution blocks."""
        super(NLayerDiscriminator_ncsn_new, self).__init__()
        # Determine if bias should be used based on the type of normalization layer
        use_bias = norm_layer == nn.InstanceNorm2d

        # List of modules that make up the main discriminator model
        self.model_main = nn.ModuleList()
        
        # First convolution block that processes the initial input layer
        self.model_main.append(
            ConvBlock_cond(input_nc, ndf, 4 * ndf, kernel_size=4, stride=1, padding=1, use_bias=use_bias))

        # Dynamically add intermediate convolution blocks with increasing feature depth
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.model_main.append(
                ConvBlock_cond(ndf * nf_mult_prev, ndf * nf_mult, 4 * ndf, kernel_size=4, stride=1, padding=1, use_bias=use_bias, norm_layer=norm_layer)
            )

        # Add the last convolution block without downsampling to maintain spatial dimensions
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.model_main.append(
            ConvBlock_cond(ndf * nf_mult_prev, ndf * nf_mult, 4 * ndf, kernel_size=4, stride=1, padding=1, use_bias=use_bias, norm_layer=norm_layer, downsample=False)
        )
        
        # Final convolution layer that outputs a single channel for discrimination
        self.final_conv = nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        # Time embedding layer that prepares the timestep embedding for integration into convolution blocks
        self.t_embed = TimestepEmbedding(
            embedding_dim=1,
            hidden_dim=4 * ndf,
            output_dim=4 * ndf,
            act=nn.LeakyReLU(0.2)
        )

    def forward(self, input, t_emb, input2=None):
        """Forward pass through the discriminator with optional dual inputs and timestep embedding."""
        t_emb = t_emb.float()  # Convert timestep embedding to float for processing
        t_emb = self.t_embed(t_emb)  # Apply embedding transformation
        # If a second input is provided, concatenate it with the first input
        out = torch.cat([input, input2], dim=1) if input2 is not None else input
        
        # Process each convolution block with the current output and timestep embedding
        for layer in self.model_main:
            out = layer(out, t_emb) if isinstance(layer, ConvBlock_cond) else layer(out)
        
        return self.final_conv(out)  # Apply the final convolution layer to produce the discriminator's output