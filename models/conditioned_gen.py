from models.helper_functions import * 
import functools 

# Resnet with conditioning : Simplified version of UNSB Conditional Resnet 


class ResnetBlockCond(nn.Module):
    def __init__(self, dim, norm_layer, temb_dim, z_dim):
        super(ResnetBlockCond, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            norm_layer(dim),
            nn.ReLU(inplace = False)
        ) 
        
        self.adaptive = AdaptiveLayer(dim, z_dim)
        
        self.conv_fin = nn.Sequential(
            nn.ReflectionPad2d(1),
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
        #print(z.shape,time_input.shape)
        out = self.conv_block(x)
        out = out + time_input[:, :, None, None]
        out = self.adaptive(out, z)
        out = self.conv_fin(out)
        out = x + out  # add skip connections
        return out

class ResnetGenerator_cond(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, n_blocks =9):
        super(ResnetGenerator_cond, self).__init__()
        
        assert(n_blocks >= 0)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(inplace = False)
        )
        
        self.ngf = ngf
        
        self.model_res = nn.ModuleList([])
        self.model_downsample = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * 4),
                          nn.ReLU(inplace = False)
        
        )
        
        for i in range(n_blocks):
            self.model_res += [ ResnetBlockCond(ngf*4, norm_layer, temb_dim=4*ngf,z_dim=4*ngf)] 
            
       
        self.model_upsample = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias = use_bias),
            norm_layer(ngf),
            nn.ReLU(inplace = False),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )
        
        
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
        #print(out.shape)     # [1,64,256,256]
        out = self.model_downsample(out)
        for layer in self.model_res:
            out = layer(out, time_embed, z_embed)
        #print(out.shape)    # [1, 256, 128, 128]
        out = self.model_upsample(out)
        #print(out.shape)    #  [1, 3, 256, 256]
        return out