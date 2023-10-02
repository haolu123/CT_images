import os
import glob
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    Flipd,
    RandAffined,
)
from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
# import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def get_loader(args):
    num_workers = 4
    # set the data directory
    data_dir = "/isilon/datalake/cialab/original/cialab/image_database/d00143/2022_dataset/original/"
    label_dir = "/isilon/datalake/cialab/original/cialab/image_database/d00143/2022_dataset/body_masks/"
    images = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(label_dir, "*.nii.tif.tif")))

    # set the training and validation split
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images, labels)]
    train_files, val_files = data_dicts[:-94], data_dicts[-94:]

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], reader = ["NibabelReader", Image.open]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-357,
                a_max=1000, 
                b_min=0.0, 
                b_max=1.0, 
                clip=True,
            ),
            Flipd(keys=["label"], spatial_axis=1),
            # ScaleIntensityRanged(
            #     keys=["label"],
            #     a_min=0,
            #     a_max=255, 
            #     b_min=0.0, 
            #     b_max=1.0, 
            #     clip=True,
            # ),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            # Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            # user can also add other random transforms
            RandAffined(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                prob=1.0, spatial_size=(96, 96),
                rotate_range=(0, 0, np.pi/15),
                scale_range=(0.1, 0.1, 0.1)),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], reader = ["NibabelReader", Image.open]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-357,
                a_max=1000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Flipd(keys=["label"], spatial_axis=1),
        ]
    )

    if args.cache_dataset:
        print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5, num_workers=num_workers)
    elif args.smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=train_files,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * args.batch_size * args.sw_batch_size,
        )
    else:
        print("Using generic dataset")
        train_ds = Dataset(data=train_files, transform=train_transforms)

    val_ds = Dataset(data=val_files, transform=val_transforms)

    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler, drop_last=True
    )

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, drop_last=True)

    return train_loader, val_loader