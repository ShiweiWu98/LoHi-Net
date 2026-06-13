from torch.utils.data import Dataset
from torchvision import transforms as TF
import os.path as osp
import os
from PIL import Image
from natsort import natsorted
import torch
from functools import partial
from torch.utils.data import DataLoader, ConcatDataset
from .transforms import paired_transform
import pytorch_lightning as pl
from omegaconf import ListConfig

class TMI_DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_root=None,
                 val_root=None,
                 test_root=None,
                 image_size=256, #or [256,256]
                 training_transforms='paired_transform',
                 batch_size=4,
                 num_workers=4,
                 pin_memory=True,
                 prefetch_factor=2,
                 ):
        super().__init__()
        self.train_root = train_root
        self.val_root = val_root
        self.test_root = test_root
        self.training_transforms = training_transforms
        if isinstance(image_size, (list, tuple, ListConfig)):
            self.image_size = image_size
        else:
            self.image_size = [image_size,image_size]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        if image_size is None:
            self.default_transforms = TF.Compose([TF.ToTensor(),])
        else:
            self.default_transforms = TF.Compose([TF.Resize(self.image_size, antialias=True),
                                                                 TF.ToTensor(),])
    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = TMI_Dataset(
                smoky_dir= osp.join(self.train_root,'smoky'),
                smokeless_dir=osp.join(self.train_root,'smokeless'),
                image_size=self.image_size,
                transforms=self.training_transforms,
            )

            self.val_dataset = TMI_Dataset(
                smoky_dir= osp.join(self.val_root,'smoky'),
                smokeless_dir=osp.join(self.val_root,'smokeless'),
                transforms=self.default_transforms,
            )
        if stage == "test":
            self.test_dataset = TMI_Dataset(
                smoky_dir= osp.join(self.test_root,'smoky'),
                smokeless_dir=osp.join(self.test_root,'smokeless'),
                transforms=self.default_transforms,
            )

        if stage == "predict":
            self.predict_dataset = TMI_Dataset(
                smoky_dir= osp.join(self.test_root,'smoky'),
                smokeless_dir=osp.join(self.test_root,'smokeless'),
                transforms=self.default_transforms,
            )
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )    

    
class TMI_Dataset(Dataset):
    def __init__(self,smoky_dir=None,smokeless_dir=None,
                image_size=256,transforms=None):
        super().__init__()
        self.smoky_path = []  # smoky image
        self.smokeless_path = [] # somkeless image
        # if isinstance(image_size,list) or isinstance(image_size,tuple):
        #     self.image_size = image_size
        # else:
        #     self.image_size = [image_size,image_size]
        self.image_size = image_size
        
        self.transforms = transforms
        if transforms is None:
            self.trans = TF.Compose([TF.Resize(self.image_size,antialias=True),
                                                                 TF.ToTensor(),])
            print(f'ToTensor() is used, image_size={image_size}')
        elif transforms=='paired_transform':
            self.trans = partial(paired_transform,crop_size=image_size,hflip=True,rotation=True)
            print(f'paired_transform is used, crop_size={image_size}, hflip=True, rotation=True')
        else:
            self.trans = transforms
        self.smoky_path = self.get_smoky_img(smoky_dir) 
        self.smokeless_path = self.get_smoky_less(smokeless_dir)
    
    def __getitem__(self, idx):
        smoky = Image.open(self.smoky_path[idx]).convert('RGB')
        smokeless = Image.open(self.smokeless_path[idx]).convert('RGB')
        if self.transforms =='paired_transform':
            smoky, smokeless = self.trans(smoky, smokeless)
        else:
            seed = torch.random.seed()
            smoky = self._aug(smoky,seed=seed)
            smokeless = self._aug(smokeless,seed=seed)
        return {'lq_files': smoky, 
                'gt_files': smokeless, 
                'lq_paths': self.smoky_path[idx]
                }

    def _aug(self, img: Image, seed=None):
        if seed is not None:
            torch.random.manual_seed(seed)
        trans_img = self.trans(img)
        return trans_img 
    
    def __len__(self,):
        return len(self.smoky_path)            
    
    def get_smoky_img(self,smoky_path):
        if not osp.exists(smoky_path):
            raise "Path is not exist!"
        smoky_list = []
        for dir, _, files in os.walk(smoky_path):
            for file in files:
                smoky_list.append(osp.join(dir,file))
        return natsorted(smoky_list)
    
    def get_smoky_less(self,smokeless_dir):
        if not osp.exists(smokeless_dir):
            raise FileNotFoundError(f"Path -{smokeless_dir}- does not exist!")
        smokeless_list = []
        for dir, _, files in os.walk(smokeless_dir):
            for file in files:
                smokeless_list.append(osp.join(dir,file))
        return natsorted(smokeless_list)