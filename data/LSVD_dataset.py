from torch.utils.data import Dataset
from torchvision import transforms as TF
import os.path as osp
import os
from PIL import Image
from natsort import natsorted
import torch
from functools import partial
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from .transforms import paired_transform
import pytorch_lightning as pl
from omegaconf import ListConfig

class LSVD_DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_root=None,
                 val_root=None,
                 test_root=None,
                 image_size=256,
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
        self.image_size = image_size
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
            self.train_dataset = LSVD_Dataset(
                video_folder=self.train_root,
                image_size=self.image_size,
                transforms=self.training_transforms,
            )
            self.val_dataset = LSVD_Dataset(
                video_folder=self.val_root,
                transforms=self.default_transforms,
                filter_corrupted=True,
            )

        if stage == "test":
            self.test_dataset = LSVD_Dataset(
                video_folder=self.test_root,
                transforms=self.default_transforms,
                filter_corrupted=True,
            )

        if stage == "predict":
            self.predict_dataset = LSVD_Dataset(
                video_folder=self.test_root,
                transforms=self.default_transforms,
                filter_corrupted=True,
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



class LSVD_Dataset(Dataset):
    def __init__(self, video_folder=None, image_size=256, transforms=None, filter_corrupted=False):
        super().__init__()
        self.smoky_path = []
        self.smokeless_path = []
        if isinstance(image_size,list) or isinstance(image_size,tuple):
            self.image_size = image_size
        else:
            self.image_size = [image_size,image_size]
        self.transforms = transforms
        
        if transforms is None:
            self.trans = TF.Compose([TF.Resize(self.image_size, antialias=True),
                                     TF.ToTensor(),
                                     TF.RandomChoice([
                                        TF.Lambda(lambda x: x),
                                        TF.Lambda(lambda x: torch.rot90(x, 1, [1, 2])),
                                        TF.Lambda(lambda x: torch.rot90(x, 2, [1, 2])),
                                        TF.Lambda(lambda x: torch.rot90(x, 3, [1, 2])),
                                     ]),
                                     TF.RandomHorizontalFlip(p=0.5),])
            print(f'ToTensor(), RandomHorizontalFlip(), and RandomRotation() are used, image_size={image_size}')
        elif transforms == 'paired_transform':
            self.trans = partial(paired_transform, crop_size=image_size, hflip=True, rotation=True)
            print(f'paired_transform is used, crop_size={image_size}, hflip=True, rotation=True')
        else:
            self.trans = transforms
        
        self.smoky_path, self.smokeless_path = self._get_frames(Path(video_folder))
        # 过滤损坏的图片
        if filter_corrupted:
            self.smoky_path, self.smokeless_path = self._filter_corrupted(self.smoky_path, self.smokeless_path)

    def _is_valid_image(self, path):
        """检查图片是否可以正常加载"""
        try:
            with Image.open(path) as img:
                img.load()
            return True
        except Exception:
            return False

    def _filter_corrupted(self, lq_paths, gt_paths):
        """过滤掉损坏的图片对"""
        valid_lq, valid_gt = [], []
        corrupted_count = 0
        for lq, gt in zip(lq_paths, gt_paths):
            if self._is_valid_image(lq) and self._is_valid_image(gt):
                valid_lq.append(lq)
                valid_gt.append(gt)
            else:
                corrupted_count += 1
        if corrupted_count > 0:
            print(f"Filtered out {corrupted_count} corrupted image pairs")
        return valid_lq, valid_gt

    def __getitem__(self, idx):
        smoky = Image.open(self.smoky_path[idx]).convert('RGB')
        smokeless = Image.open(self.smokeless_path[idx]).convert('RGB')
        if self.transforms == 'paired_transform':
            smoky, smokeless = self.trans(smoky, smokeless)
        else:
            seed = torch.random.seed()
            smoky = self._aug(smoky, seed=seed)
            smokeless = self._aug(smokeless, seed=seed)
        return {'lq_files': smoky,
                'gt_files': smokeless,
                'lq_paths': str(self.smoky_path[idx])
                }

    def _aug(self, img, seed=None):
        if seed is not None:
            torch.random.manual_seed(seed)
        trans_img = self.trans(img)
        return trans_img

    def __len__(self):
        return len(self.smoky_path)

    def _get_frames(self, root_dir):
        """
        遍历所有视频序列文件夹，每个文件夹中：
        - 第一帧（按名称排序）为smokeless (gt)
        - 其余帧为smoky (lq)
        """
        gt_imgs = []
        lq_imgs = []

        for video_folder in sorted(root_dir.iterdir()):
            if not video_folder.is_dir():
                continue

            frame_paths = natsorted([
                p for p in video_folder.iterdir()
                if p.suffix.lower() in ['.jpg', '.png', '.jpeg']
            ])

            if len(frame_paths) > 1:
                # lq: 第1帧及之后的所有帧
                lq_imgs.extend(frame_paths[1:])
                # gt: 第0帧，重复和lq数量一样
                gt_imgs.extend([frame_paths[0]] * (len(frame_paths) - 1))

        return lq_imgs, gt_imgs
