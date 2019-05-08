from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import random
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from utils import dense_optical_flow

class TestDataHandler(Dataset):

    def __init__(self, root_dir, test_images_dir=None, test_mask_images_dir=None):
        # Root directory that contains the dataset
        self.root_dir = root_dir
        self.dataset_mean = [0.0014861894323434117]
        self.dataset_std = [0.0020256241244931863]
        # test set
        test_images_path = os.path.join(self.root_dir,test_images_dir)
        test_mask_images_path = os.path.join(self.root_dir,test_mask_images_dir)
        file_names = sorted(os.listdir(test_images_path))
        mask_names = sorted(os.listdir(test_mask_images_path))
        self.test_file_names = [os.path.join(test_images_path,name) for name in file_names]
        self.test_mask_file_names = [os.path.join(test_mask_images_path,name) for name in mask_names]
        self.transform = self.create_transformation()

    @staticmethod
    def create_transformation():
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform

    def __len__(self):
        return len(self.test_file_names)

    def __getitem__(self, ind):
        # Open both the test images and the masks
        img = Image.open(self.test_file_names[ind]).convert('RGB')
        name = self.test_file_names[ind]
        mask = Image.open(self.test_mask_file_names[ind])
        # print(self.test_mask_file_names[ind])
        # If all pixels are white in the mask the image does not have any liver cells
        if np.mean(mask) == 255:
            label = 0
        else:
            label = 1
        if self.transform is not None:
            img = self.transform(img)

        return img,label,name


class DataHandler(Dataset):
    '''
    This is the data handler for the train and validation test set.
    '''
    def __init__(self, root_dir, gt_file_path, mode='train'):
        # Root directory that contains the dataset
        self.root_dir = root_dir
        self.file_names = sorted(os.listdir(self.root_dir),key=lambda x: int(x[:-4]))
        self.file_names = [os.path.join(self.root_dir,name) for name in self.file_names]
        self.dev_file_indices = list(np.asarray(random.sample(range(0,len(self.file_names)),int(0.2*len(self.file_names))))//2)
        self.train_file_indices = [i for i in range(len(self.file_names)//2)]
        self.train_file_indices = list(set(self.train_file_indices).difference(set(self.dev_file_indices)))
        self.dev_file_names = [os.path.join(self.root_dir,str(self.dev_file_indices[i])+".jpg") for i in range(len(self.dev_file_indices))]
        self.train_file_names = [os.path.join(self.root_dir, str(self.train_file_indices[i]) + ".jpg") for i in range(len(self.train_file_indices))]
        self.gt_file = np.loadtxt(gt_file_path)
        self.mode = mode
        # Train set mean and standard deviation caluculated before hand.
        self.dataset_mean = [0.0014861894323434117]
        self.dataset_std = [0.0020256241244931863]

        # Get the complete paths of all files in the dataset
        if mode == "train":
            self.names = self.train_file_names
            self.indices = self.train_file_indices
            # Assign label 0 to negative images and label 1 to positive images

        # creating a dev set with 20% of train data containing liver and no liver images
        elif mode == 'val':
            self.names = self.dev_file_names
            self.indices = self.dev_file_indices
        # The type of image transformations that we will try
        self.transform = self.create_transformation()

    # Use transformations for image augmentation.
    @staticmethod
    def create_transformation():
        transform = transforms.Compose([
            transforms.Resize((66,200)),
            transforms.ToTensor()
            ])
        return transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, ind):
        # Open the image corresponding to the index
        img1 = np.asarray(Image.open(self.file_names[self.indices[ind]*2]).convert('RGB'))
        img2 = np.asarray(Image.open(self.file_names[self.indices[ind]*2+1]).convert('RGB'))
        img = dense_optical_flow(img1,img2)
        img = Image.fromarray(img,'RGB')
        names = self.file_names[self.indices[ind]*2]
        # Apply transformation to image
        if self.transform is not None:
            img = self.transform(img)
        # Label of image
        label = self.gt_file[self.indices[ind]*2]

        return img, label, names

if __name__ == "__main__":
    data_handler = DataHandler("data/frames_train","data/train.txt","train")
    batch_size = 8
    num_workers = 1
    all_labels = []
    loader = DataLoader(data_handler, batch_size,shuffle=True,num_workers=num_workers, pin_memory=True)
    for i, batch in enumerate(loader):
        print(i)