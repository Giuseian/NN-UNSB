import os 
import torch 
from torch.utils.data import Dataset 
from torchvision import transforms
from PIL import Image
import glob
import random
from torch.autograd import Variable
import torchvision.transforms as transforms

# Image transformations utilities
img_height = 256
img_width = 256
debug_mode = False

# Convert grayscale images to rgb
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))
    
# Image transformations 
transforms_ = [
    transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Image dataset class
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        print("root",root)
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        
        print(os.path.join(root,f"{mode}A"))
        self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}A") + "/*.*"))
        print("self.files_A", self.files_A)
        self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}B") + "/*.*"))
        print("self.files_B", self.files_B)
        if debug_mode:
            self.files_A = self.files_A[:100]
            self.files_B = self.files_B[:100]

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
