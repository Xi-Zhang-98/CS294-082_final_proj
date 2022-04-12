from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np


class ISBI2021Dataset(Dataset):
    def __init__(self, root_dir, img_size, loader):

        self.path = os.path.join(root_dir, loader)

        self.file_path = sorted(make_dataset(self.path))

        # self.transform = {
        #     'training':transforms.Compose([
        #     transforms.RandomRotation(30),
        #     transforms.RandomResizedCrop(224),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225]
        #     )]),
        #     'testing':transforms.Compose([
        #     transforms.Resize(img_size),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225]
        #     )])

        # }
        self.transform = transforms.Compose([
            transforms.Resize(img_size), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(), 
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )])
        

    def __getitem__(self, index):
        file = self.file_path[index]
        img = Image.open(file).convert('RGB')
        img_transform = self.transform(img)
        A_label = file.split('/')[-1].split('_')[0]
        if A_label == 'normal':
            label = 0
        elif A_label == 'AMD':
            label = 1
        else:
            label = 2
        return img_transform, label
    def __len__(self):
        return len(self.file_path)

class ODIRDataset(Dataset):
    def __init__(self, root_dir, img_size, loader):

        self.path = os.path.join(root_dir, loader)

        self.file_path = sorted(make_dataset(self.path))

        self.transform = transforms.Compose([
            transforms.Resize(img_size), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(), 
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )])
        

    def __getitem__(self, index):
        file = self.file_path[index]
        img = Image.open(file).convert('RGB')
        img_transform = self.transform(img)
        A_label = file.split('/')[-1].split('_')[0]
        if A_label == 'normal':
            label = 0
        else:
            label = 1
        return img_transform, label
    def __len__(self):
        return len(self.file_path)

class STAREDataset(Dataset):
    def __init__(self, root_dir, img_size, loader):

        self.path = os.path.join(root_dir, loader)

        self.file_path = sorted(make_dataset(self.path))

        self.transform = transforms.Compose([
            transforms.Resize(img_size), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(), 
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )])
        

    def __getitem__(self, index):
        file = self.file_path[index]
        img = Image.open(file).convert('RGB')
        img_transform = self.transform(img)
        A_label = file.split('/')[-1].split('_')[0]
        if A_label == 'normal':
            label = 0
        else:
            label = 1
        return img_transform, label
    def __len__(self):
        return len(self.file_path)

class iCHALLENGEDataset(Dataset):
    def __init__(self, root_dir, img_size, loader):

        self.path = os.path.join(root_dir, loader)

        self.file_path = sorted(make_dataset(self.path))

        self.transform = transforms.Compose([
            transforms.Resize(img_size), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(), 
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )])
        

    def __getitem__(self, index):
        file = self.file_path[index]
        img = Image.open(file).convert('RGB')
        img_transform = self.transform(img)
        A_label = file.split('/')[-1].split('_')[0]
        if A_label == 'normal':
            label = 0
        else:
            label = 1
        return img_transform, label
    def __len__(self):
        return len(self.file_path)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images[:min(max_dataset_size, len(images))]




