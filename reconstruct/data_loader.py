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
    ToTensord,
)
from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
# import matplotlib.pyplot as plt
import json
import numpy as np

def get_loader(num_workers, batch_size, train_sampler):
    data_dir = "/isilon/datalake/cialab/original/cialab/image_database/d00143/2022_dataset/original/"
    label_dir = "/isilon/datalake/cialab/original/cialab/image_database/d00143/2022_dataset/"
    label_file = "labels.json"
    with open(os.path.join(label_dir, label_file), 'r') as f:
        labels = json.load(f)

    # get the image file names and labels
    image_name_list = [os.path.join(data_dir, str(image_name['id'])+'.nii') for image_name in labels]


    label_list = [label['age'] for label in labels]
    max_age = max(label_list)
    min_age = min(label_list)
    label_list = [[(label-min_age)/(max_age-min_age)] for label in label_list] # Not sure it is still a good idea to normalize the label for spearman loss

    data_dicts = []
    for image_name, label in zip(image_name_list, label_list):
        if os.path.exists(image_name):
            data_dicts.append({"image": image_name, "label": label})
    train_files, val_files = data_dicts[:-200], data_dicts[-200:]
    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-500,
                a_max=1500, 
                b_min=0.0, 
                b_max=1.0, 
                clip=True,
            ),
            # RepeatChanneld(keys=["image"],repeats=3),
            ToTensord(keys=["image", "label"]),
            # RandCropByPosNegLabeld(
            #     keys=["image", "label"],
            #     label_key="label",
            #     spatial_size=(96, 96),
            #     pos=1,
            #     neg=1,
            #     num_samples=4,
            #     image_key="image",
            #     image_threshold=0,
            # ),
            # user can also add other random transforms
            # RandAffined(
            #     keys=['image', 'label'],
            #     mode=('bilinear', 'nearest'),
            #     prob=1.0, spatial_size=(96, 96),
            #     rotate_range=(0, 0, np.pi/15),
            #     scale_range=(0.1, 0.1, 0.1)),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-500,
                a_max=1500,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # # RepeatChanneld(keys=["image"],repeats=3),
            ToTensord(keys=["image", "label"]),
        ]
    )

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler,shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)


    # if args.cache_dataset:
    #     print("Using MONAI Cache Dataset")
    #     train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5, num_workers=num_workers)
    # elif args.smartcache_dataset:
    #     print("Using MONAI SmartCache Dataset")
    #     train_ds = SmartCacheDataset(
    #         data=train_files,
    #         transform=train_transforms,
    #         replace_rate=1.0,
    #         cache_num=2 * args.batch_size * args.sw_batch_size,
    #     )
    # else:
    #     print("Using generic dataset")
    #     train_ds = Dataset(data=train_files, transform=train_transforms)

    # val_ds = Dataset(data=val_files, transform=val_transforms)

    # if args.distributed:
    #     train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
    # else:
    #     train_sampler = None
    # train_loader = DataLoader(
    #     train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler, drop_last=True
    # )

    # val_ds = Dataset(data=val_files, transform=val_transforms)
    # val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, drop_last=True)

    return train_loader, val_loader