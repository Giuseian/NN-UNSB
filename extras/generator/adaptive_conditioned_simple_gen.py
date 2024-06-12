from models.helper_functions import *

"""Simple Resnet with Adaptive Conditioning"""
# It's a simplified version of the UNSB generator

class ResnetBlockS_Cond(nn.Module):
    # Constructor for a single ResNet block with conditional inputs
    def __init__(self, dim, padding_type, norm_layer, use_dropout, temb_dim, z_dim):
        super(ResnetBlockS_Cond, self).__init__()
        
        # Setup padding based on specified type
        padding = get_pad_layer(padding_type)(1) if padding_type != 'zero' else 0
        # Define the convolutional block
        self.conv_block = nn.Sequential(
            padding,
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            norm_layer(dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5) if use_dropout else nn.Identity()  # Optionally add dropout for regularization
        )
        
        # Style-based adaptive layer for applying learned affine transformations
        self.adaptive = AdaptiveLayer(dim, z_dim)
        
        # Final convolutional layer of the block
        self.conv_fin = nn.Sequential(
            padding,
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            norm_layer(dim)
        )
        
        # Dense layer for embedding time conditioning
        self.dense_time = nn.Linear(temb_dim, dim)
        nn.init.zeros_(self.dense_time.bias)  # Initialize bias to zero for stability
        # Style affine transformation setup similar to adaptive layer
        self.style = nn.Linear(z_dim, dim * 2)
        self.style.bias.data[:dim] = 1
        self.style.bias.data[dim:] = 0

    # Forward method to process input through the ResNet block
    def forward(self, x, time_cond, z):
        time_input = self.dense_time(time_cond)  # Process time conditioning
        out = self.conv_block(x)  # Pass input through the convolutional block
        out = out + time_input[:, :, None, None]  # Add time embedding
        out = self.adaptive(out, z)  # Apply style-based adaptive layer
        out = self.conv_fin(out)  # Pass through final convolution
        out = x + out  # Include residual connection
        return out


class ResnetGeneratorS_Cond(nn.Module): 
    # Constructor for the generator network with spatial and temporal conditioning
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, n_blocks=9, use_dropout=False):
        super(ResnetGeneratorS_Cond, self).__init__()
        self.input_nc = input_nc  # Number of input channels
        self.output_nc = output_nc  # Number of output channels
        self.ngf = ngf  # Number of generator filters
        
        # Initial model layers including reflective padding and convolution
        model = [get_pad_layer('reflect')(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(inplace=False)]
        
        # Downsampling part
        model_downsample = []
        for i in range(2):
            mult = 2 ** i
            model_downsample += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(inplace=False)]
        
        # Sequential blocks for deep residual processing
        self.model_res = nn.ModuleList()
        mult = 2 ** 2
        for i in range(n_blocks):
            self.model_res += [ResnetBlockS_Cond(ngf * mult, 'reflect', norm_layer, use_dropout, temb_dim=4*ngf, z_dim=4*ngf)]
        
        # Upsampling part to reconstruct the image resolution
        model_upsample = []
        for i in range(2, 0, -1):
            mult = 2 ** i
            model_upsample += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf * mult // 2),
                nn.ReLU(True)]
        
        # Final model output layers with tanh activation to normalize the output
        self.final_model = nn.Sequential(
            get_pad_layer('reflect')(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh())
        
        # Full model concatenation of initial, downsampling, upsampling layers
        self.model = nn.Sequential(*model)
        self.model_downsample = nn.Sequential(*model_downsample)
        self.model_upsample = nn.Sequential(*model_upsample)

        # Latent vector transformation layers
        mapping_layers = [PixelNorm(),
                          nn.Linear(ngf*4, ngf*4),
                          nn.LeakyReLU(0.2)]
        
        self.z_transform = nn.Sequential(*mapping_layers)
        
        # Embedding layers for time conditioning
        modules_emb = [nn.Linear(ngf, ngf*4)]
        nn.init.zeros_(modules_emb[-1].bias)
        modules_emb += [nn.LeakyReLU(0.2), nn.Linear(ngf*4, ngf*4)]
        nn.init.zeros_(modules_emb[-1].bias)
        modules_emb += [nn.LeakyReLU(0.2)]
        self.time_embed = nn.Sequential(*modules_emb)
                                
    # Forward method to process input through the entire generator network
    def forward(self, x, time_cond, z):
        z_embed = self.z_transform(z)  # Transform latent vector
        temb = get_timestep_embedding(time_cond, self.ngf)  # Get timestep embeddings
        time_embed = self.time_embed(temb)  # Embed time conditioning
        out = self.model(x)  # Process input through the initial model
        out = self.model_downsample(out)  # Apply downsampling
        for layer in self.model_res:  # Apply all ResNet blocks
            out = layer(out, time_embed, z_embed)
        out = self.model_upsample(out)  # Upsample to the original resolution
        out = self.final_model(out)  # Final output processing
        return out

