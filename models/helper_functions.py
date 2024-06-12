""" Here we define helper functions that we will use throughout the project """

# Define padding layer based on type for use in convolutional layers
def get_pad_layer(pad_type):
    if pad_type in ['reflect', 'refl']:
        return nn.ReflectionPad2d
    elif pad_type in ['replicate', 'repl']:
        return nn.ReplicationPad2d
    elif pad_type == 'zero':
        return nn.ZeroPad2d
    else:
        raise NotImplementedError(f'Padding type {pad_type} not recognized')

# Module to normalize pixel values in images for stable training
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)
        
# Generate embeddings for timesteps in models that incorporate time dynamics
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
                                  
# Class to embed timestep information into network inputs
class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.ReLU()):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)  # First layer: embedding to hidden dimension
        self.act = act  # Activation function
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Second layer: hidden dimension to output

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t = t.float()  # Ensure t is of type float
        t = self.fc1(t)
        t = self.act(t)
        t = self.fc2(t)
        return t
    
# Initialize network weights using a specific strategy for better training performance
def init_weights(net, init_gain=0.02, debug=False):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)  # Print class name during debugging
            init_gain = 0.02
            init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('LayerNorm') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>

# Set up network for use, optionally initialize weights, and set GPU configuration if available
def init_net(net, init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    if initialize_weights:
        init_weights(net, init_gain=init_gain, debug=debug)
    return net

# Module to normalize tensors based on a power rule, useful for data and feature normalization
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
    
# Function to convert normalized image data back to standard image format
def denormalize(tensor):
    return tensor.mul(0.5).add(0.5)  # Converts from [-1, 1] to [0, 1]

# Visualize a batch of images using a grid layout
def visualize_images(images, title="Generated Images"):
    images = images.cpu()  # Move images to CPU for visualization
    images = denormalize(images)  # Denormalize images to bring them to displayable format
    grid = vutils.make_grid(images, padding=2, normalize=True)  # Create a grid of images
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(grid, (1, 2, 0)))  # Display images
    plt.show()