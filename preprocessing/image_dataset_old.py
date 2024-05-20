import os 
import cv2 
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.image_paths = []
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):  # Checking file extension if needed
                    self.image_paths.append(os.path.join(root, file))
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy arrays to PIL Image
            transforms.Resize((256, 256)),  # Resize to 256x256
            transforms.ToTensor(),  # Convert to Tensor
            #transforms.Lambda(lambda x: add_gaussian_noise(x, std=0.1)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        if self.transform:
            image = self.transform(image)
        return image 




        