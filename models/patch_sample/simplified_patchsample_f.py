import torch
import torch.nn as nn
import numpy as np
from models.helper_functions import *

class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, output_shape=(1, 12, 256, 256), device=None):
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc
        self.output_shape = output_shape
        self.device = device if device is not None else torch.device('cpu')  # Default to CPU if not specified
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain

    def create_mlp(self, input_nc):
        # The last layer's output size must be the total number of elements in the target output shape
        output_size = 12 * 256 * 256  # This matches the total elements for a 12x256x256 image
        self.mlp = nn.Sequential(
            nn.Linear(input_nc, self.nc),  # First layer
            nn.LeakyReLU(),
            nn.Linear(self.nc, output_size)  # Adjust this layer to match the reshaping requirement
        ).to(self.device)
        self.mlp_init = True

    def forward(self, input):
        B, C, H, W = input.shape
        if self.use_mlp:
            if not self.mlp_init:
                self.create_mlp(C * H * W)  # Initialize MLP based on the input dimensions

            input_flat = input.view(B, -1)  # Flatten the input
            input = self.mlp(input_flat)  # Pass through MLP

        input = input.view(*self.output_shape)  # Reshape to the desired output shape
        input = self.l2norm(input)  # Normalize the output

        return input