from models.helper_functions import *

# Simple Resnet with Adaptive Conditioning : simple_gen with Adaptive Conditioning

class ResnetBlockS_Cond(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, temb_dim, z_dim):
        super(ResnetBlockS_Cond, self).__init__()
        
        padding = get_pad_layer(padding_type)(1) if padding_type != 'zero' else 0
        self.conv_block = nn.Sequential(
            padding,
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            norm_layer(dim),
            nn.ReLU(inplace = False),
            nn.Dropout(0.5) if use_dropout else nn.Identity()
        )
        
        self.adaptive = AdaptiveLayer(dim, z_dim)
        
        self.conv_fin = nn.Sequential(
            padding,
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            norm_layer(dim)
        )
        
        self.dense_time = nn.Linear(temb_dim, dim)
        nn.init.zeros_(self.dense_time.bias)
        self.style = nn.Linear(z_dim, dim * 2)
        self.style.bias.data[:dim] = 1
        self.style.bias.data[dim:] = 0

    def forward(self, x, time_cond, z):
        time_input = self.dense_time(time_cond) 
        #print("z",z.shape,"time_input",time_input.shape) 
        out = self.conv_block(x)
        #print("out_conv",out.shape)
        out = out + time_input[:, :, None, None]
        #print("out_inplace",out.shape)
        out = self.adaptive(out, z)
        #print("out_adaptive",out.shape)
        out = self.conv_fin(out)
        #print("out_fin",out.shape)
        out = x + out  # add skip connections
        return out


class ResnetGeneratorS_Cond(nn.Module): 
    def __init__(self, input_nc, output_nc, ngf=64,norm_layer=nn.BatchNorm2d, n_blocks=9, use_dropout=False):
        super(ResnetGeneratorS_Cond, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        
        model = [get_pad_layer('reflect')(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(inplace = False)]
        
        # Downsampling
        model_downsample = []
        for i in range(2):
            mult = 2 ** i
            model_downsample +=[
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(inplace=False)]
        
        # ResNet blocks
        self.model_res = nn.ModuleList()
        mult = 2 ** 2
        for i in range(n_blocks):
            self.model_res += [ResnetBlockS_Cond(ngf * mult, 'reflect', norm_layer, use_dropout, temb_dim=4*ngf,z_dim=4*ngf)]
        
        # Upsampling
        model_upsample = []
        for i in range(2, 0, -1):
            mult = 2 ** i
            model_upsample += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf * mult // 2),
                nn.ReLU(True)]
        
        
        self.final_model = nn.Sequential(
            get_pad_layer('reflect')(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh())
            
        self.model = nn.Sequential(*model)
        self.model_downsample = nn.Sequential(*model_downsample)
        self.model_upsample = nn.Sequential(*model_upsample)

        mapping_layers = [PixelNorm(),
                      nn.Linear(self.ngf*4, self.ngf*4),
                      nn.LeakyReLU(0.2)]
        
        self.z_transform = nn.Sequential(*mapping_layers)
        modules_emb = []
        modules_emb += [nn.Linear(self.ngf,self.ngf*4)]
        
        nn.init.zeros_(modules_emb[-1].bias)
        modules_emb += [nn.LeakyReLU(0.2)]
        modules_emb += [nn.Linear(self.ngf*4,self.ngf*4)]
        
        nn.init.zeros_(modules_emb[-1].bias)
        modules_emb += [nn.LeakyReLU(0.2)]
        self.time_embed = nn.Sequential(*modules_emb)
                                
    def forward(self, x, time_cond, z):
        #print("x",x.shape)   # [1,3,256,256]  # [B,C,H,W]
        z_embed = self.z_transform(z)
        #print(z_embed.shape)    #[1,256]
        temb = get_timestep_embedding(time_cond, self.ngf)
        #print(temb.shape)
        time_embed = self.time_embed(temb)
        #print(time_embed.shape)   # [1,256]
        out = self.model(x)                    # As input to the model,you give input_nc = 3,and the CNN goes from 3 to 64 -> resulting : [1,64,256,256]
        #print("model",out.shape)     # [1,64,256,256]
        out = self.model_downsample(out)
        #print("down",out.shape)
        for layer in self.model_res:
            out = layer(out, time_embed, z_embed)
        #print("res",out.shape)    # [1, 256, 128, 128]
        out = self.model_upsample(out)
        #print("upsample",out.shape)
        out = self.final_model(out)
        #print("final", out.shape)
        #print(out.shape)    #  [1, 3, 256, 256]
        return out
