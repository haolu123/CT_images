from utils.parser import parse_args
from utils.MultiGPU import distributed_init, distributed_params, set_device, multigpu_dataloader
import torch
from monai.networks.nets import UNet,SEResNet50
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, MAEMetric
from monai.losses import DiceLoss, FocalLoss
import torch.nn as nn
from losses.losses import HausdorffDTLoss
from utils.data_utils import get_loader
from trains.train import training
import os
import json
from models.ResNet50 import Backbone
from models.Classifier import Classifier
from models.model import MyModel
from optimizers.lr_scheduler import WarmupCosineSchedule
import torch.distributed as dist

def main():
    # Parse the arguments
    args = parse_args()
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    if args.distributed:
        # initializing multi-node settings
        distributed_init()
        local_rank = distributed_params() 

        # setting the device
        device = set_device(local_rank_param=local_rank, multi_gpu=True)
        torch.cuda.set_device(device) # set the cuda device, this line doesn't included in Usman's code. But appears in MONAI tutorial
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (torch.distributed.get_rank(), torch.distributed.get_world_size())
        )
    else:
        device = torch.device("cuda:0")
        print("Training with a single process on 1 GPU.")
    # load the model
    # backbone = Backbone()
    # backbone.load_state_dict(torch.load("../RadImageNet_pytorch/ResNet50.pt"))
    # classifier = Classifier(num_class=1)
    # model = nn.Sequential(backbone, classifier)
    model = SEResNet50(spatial_dims=2, in_channels=3, pretrained=False, num_classes=8)
    # for param in backbone.parameters():
    #     param.requires_grad = False
    # model.cuda()


    model = model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank) # find_output_device=True is optional
    
    # loss functions
    # dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
    # facal_loss = FocalLoss(to_onehot_y=True,use_softmax=True)
    # hausdorff_loss = HausdorffDTLoss()
    # BCE_loss = nn.BCEWithLogitsLoss()
    CE_loss = nn.CrossEntropyLoss()
    # huber_loss = nn.SmoothL1Loss()
    loss_name = 'huber_loss'
    # losses = {
    #             "dice_loss": dice_loss,
    #             "facal_loss": facal_loss, 
    #             "hausdorff_loss": hausdorff_loss, 
    #             "BCE_loss": BCE_loss, 
    #             "CE_loss": CE_loss
    #         }
    # mae_metric = MAEMetric() # mean absolute error

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    # load the data
    train_loader, val_loader = get_loader(args)
    # training
    epoch_loss_values, metric_values = training(args, train_loader, val_loader, model, optimizer, device, CE_loss, args.model_save_dir, len(train_loader.dataset))
    # save hyper-parameters
    hyper_para = {"batch_size": args.batch_size,
                   "lr": args.lr, "epochs": args.epochs,
                    "val_interval": args.val_interval,
                    "distributed": args.distributed,
                    "cache_dataset": args.cache_dataset,
                    "smartcache_dataset": args.smartcache_dataset,
                    "losses": loss_name,
                }
    with open(os.path.join(args.result_dir, "hyper_para.txt"), "w") as f:
        f.write(str(hyper_para))
        f.write("\n")
        f.write("losses: ")
        f.write(str(epoch_loss_values))
        f.write("\n")
        f.write("metric: ")
        f.write(str(metric_values))
        f.write("\n")
    with open(os.path.join(args.result_dir,
                            "bs_{}_lr_{}_dist_{}_loss_{}_losses.json".format(
                                args.batch_size, args.lr, args.distributed, loss_name
                            )), "w") as f:
        json.dump(epoch_loss_values, f)
    with open(os.path.join(args.result_dir, "bs_{}_lr_{}_dist_{}_loss_{}_losses.json".format(
                                args.batch_size, args.lr, args.distributed, loss_name
                            )), "w") as f:
        json.dump(metric_values, f)
    
    

    
if __name__ == '__main__':
    main()