from models.helper_functions import * 
import functools 
import torch.nn as nn

# Conditional ResNet Generator with Adaptive Conditioning
# The generator uses a series of ResNet blocks with adaptive conditioning to generate images.
# The generator also uses time-based conditioning to generate images at different time steps.

class ResnetGenerator_cond(nn.Module):
    # Initialization of the conditional ResNet generator
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, n_blocks=9):
        super(ResnetGenerator_cond, self).__init__()
        
        # Ensuring the number of blocks is non-negative
        assert(n_blocks >= 0)
        # Determine if bias is needed based on the type of normalization layer
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        # Initial convolution module to process input image
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),  # Padding before initial convolution
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),  # Initial convolution to transform input channel
            norm_layer(ngf),  # Normalization layer
            nn.ReLU(inplace=False)  # Activation function
        )
        
        self.ngf = ngf  # Number of generator filters
        
        # List of residual blocks with conditional inputs
        self.model_res = nn.ModuleList([])
        # Downsampling part of the model
        self.model_downsample = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(inplace=False)
        )
        
        # Add multiple ResnetBlockCond instances for intermediate processing
        for i in range(n_blocks):
            self.model_res += [ResnetBlockCond(ngf * 4, norm_layer, temb_dim=4 * ngf, z_dim=4 * ngf)]
       
        # Upsampling part of the model to restore original image size
        self.model_upsample = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()  # Output activation to ensure output values are between -1 and 1
        )
        
        # Define a transformation for the latent vector z
        mapping_layers = [PixelNorm(),
                          nn.Linear(self.ngf * 4, self.ngf * 4),
                          nn.LeakyReLU(0.2)]
        self.z_transform = nn.Sequential(*mapping_layers)
        
        # Time embedding layers
        modules_emb = [nn.Linear(self.ngf, self.ngf * 4)]
        nn.init.zeros_(modules_emb[-1].bias)  # Initialize the bias to zero for stability
        modules_emb += [nn.LeakyReLU(0.2), nn.Linear(self.ngf * 4, self.ngf * 4)]
        nn.init.zeros_(modules_emb[-1].bias)  # Again, initialize the bias to zero
        modules_emb += [nn.LeakyReLU(0.2)]
        self.time_embed = nn.Sequential(*modules_emb)
                                
    # Define the forward pass with conditional inputs time_cond and z
    def forward(self, x, time_cond, z):
        z_embed = self.z_transform(z)  # Transform z before feeding it to the ResNet blocks
        temb = get_timestep_embedding(time_cond, self.ngf)  # Embedding the time steps
        time_embed = self.time_embed(temb)  # Applying the time embedding
        out = self.model(x)  # Initial processing of input
        out = self.model_downsample(out)  # Apply downsampling
        for layer in self.model_res:  # Apply each ResNet block sequentially
            out = layer(out, time_embed, z_embed)
        out = self.model_upsample(out)  # Final upsampling and output layer
        return out
     