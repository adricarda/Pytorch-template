import torch
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from dataloader.dataloader_cityscapes import CityScapesDataset
from albumentations import (
    HorizontalFlip,
    Compose,
    Resize,
    Normalize,
    RandomCrop
    )

def fetch_dataloader(root, txt_file, split, params):
    h, w = params.crop_h, params.crop_w
    mean = [0.286, 0.325, 0.283]
    std = [0.176, 0.180, 0.177]
    
    if split == 'train':
        transform_train = Compose([RandomCrop(h,w),
                    HorizontalFlip(p=0.5), 
                    Normalize(mean=mean,std=std)])

        dataset=CityScapesDataset(root, txt_file, transforms=transform_train, mean=mean, std=std)
        return DataLoader(dataset, batch_size=params.batch_size_train, shuffle=True, num_workers=params.num_workers, drop_last=True, pin_memory=True)

    else:
        transform_val = Compose( [Normalize(mean=mean,std=std)])

        dataset=CityScapesDataset(root, txt_file, transforms=transform_val, mean=mean, std=std)

        #reduce validation data to speed up training
        if "split_validation" in params.dict:
            ss = ShuffleSplit(n_splits=1, test_size=params.split_validation, random_state=42)
            indexes=range(len(dataset))
            split1, split2 = next(ss.split(indexes))
            dataset=Subset(dataset, split2)        

        return DataLoader(dataset, batch_size=params.batch_size_val, shuffle=False, num_workers=params.num_workers, drop_last=True, pin_memory=True)