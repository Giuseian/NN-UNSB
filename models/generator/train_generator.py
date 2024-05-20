from torch.utils.data import DataLoader
from preprocessing.image_dataset_old import ImageDataset

path_trainA = 'datasets/horse2zebra/trainA'
train_datasetA = ImageDataset(img_dir=path_trainA)
train_dataloaderA = DataLoader(train_datasetA, batch_size=1, shuffle=True)

path_trainB = 'datasets/horse2zebra/trainB'
train_datasetB = ImageDataset(img_dir=path_trainB)
train_dataloaderB = DataLoader(train_datasetB, batch_size=1, shuffle=True)

