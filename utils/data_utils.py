######## lots of problems not finished yet


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
    RepeatChanneld,
)
from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
# import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json

def get_loader(args):
    num_workers = 4
    # set the data directory
    data_dir = "/isilon/datalake/cialab/original/cialab/image_database/d00143/2022_dataset/original/"
    label_dir = "/isilon/datalake/cialab/original/cialab/image_database/d00143/2022_dataset/"
    label_file = "labels.json"
    with open(os.path.join(label_dir, label_file), 'r') as f:
        labels = json.load(f)

    # get the image file names and labels
    image_name_list = [os.path.join(data_dir, str(image_name['id'])+'.nii') for image_name in labels]

    # get the labels
    # class 0: age 55-65, gender : Male, 
    # class 1: age 55-65, gender : Famale,
    # class 2: age 65-75, gender : Male,
    # class 3: age 65-75, gender : Famale,
    # class 4: age 75-85, gender : Male,
    # class 5: age 75-85, gender : Famale,
    # class 6: age 85-95, gender : Male,
    # class 7: age 85-95, gender : Famale,
    # generate the labels
    # label_list = []
    # for label in labels:
    #     age = label['age']
    #     gender = label['gender']
    #     if age >= 55 and age < 65:
    #         if gender == 'Male':
    #             label_list.append(0)
    #         else:
    #             label_list.append(1)
    #     elif age >= 65 and age < 75:
    #         if gender == 'Male':
    #             label_list.append(2)
    #         else:
    #             label_list.append(3)
    #     elif age >= 75 and age < 85:
    #         if gender == 'Male':
    #             label_list.append(4)
    #         else:
    #             label_list.append(5)
    #     elif age >= 85 and age <= 95:
    #         if gender == 'Male':
    #             label_list.append(6)
    #         else:
    #             label_list.append(7)

    
    map_three_categories = {'Frail': 0, 'Prefrail': 1, 'Not frail': 2,}
    map_two_categories = {'Abnormal': 0, 'Normal': 1}
    map_gender = {'Male': 0, 'Female': 1}

    label_list = [[
                    label['eFI'], 
                    map_three_categories[label['Three Categories']], 
                    map_two_categories[label['Two Categories']], 
                    map_gender[label['gender']],
                    label['age']
                ] for label in labels]
    max_age = max([label[4] for label in label_list])
    min_age = min([label[4] for label in label_list])
    label_list = [[label[0], label[1], label[2], label[3], (label[4]-min_age)/(max_age-min_age)] for label in label_list]

    # set the training and validation split
    data_dicts = []
    for image_name, label in zip(image_name_list, label_list):
        if os.path.exists(image_name):
            data_dicts.append({"image": image_name, "label": label})
    train_files, val_files = data_dicts[:-94], data_dicts[-94:]

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
            RepeatChanneld(keys=["image"],repeats=3),
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
            RepeatChanneld(keys=["image"],repeats=3),
            ToTensord(keys=["image", "label"]),
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

    return train_loader, val_loader, max_age, min_age

if __name__ == '__main__':
    from parser import parse_args
    # from matplotlib import pyplot as plt
    from monai.utils import first, set_determinism
    args = parse_args()
    train_loader, val_loader = get_loader(args)
    check_data = first(train_loader)
    image, label = (check_data["image"][0][0], check_data["label"])
    print(f"image shape: {image.shape}, label shape: {label.shape}")
    print(check_data["label"])