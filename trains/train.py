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
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.metrics import DiceMetric
import os

def training(args, train_loader,val_loader, model, optimizer, device, losses, metric_function, model_save_dir, len_ds):
    dice_loss = losses["dice_loss"]
    facal_loss = losses["facal_loss"]
    hausdorff_loss = losses["hausdorff_loss"]
    BCE_loss = losses["BCE_loss"]
    CE_loss = losses["CE_loss"]
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    lambda_ = (0.7, 0.3, 0.1)
    
    for epoch in range(args.epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss1 = dice_loss(outputs, labels)
            loss2 = facal_loss(outputs, labels)
            loss3 = CE_loss(outputs.view(args.batch_size,2,-1), labels.view(args.batch_size,-1).astype(torch.long))
            # loss3 = hausdorff_loss(outputs, labels)
            loss = lambda_[0]*loss1 + lambda_[1]*loss2  + lambda_[2]*loss3
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len_ds // (train_loader.batch_size*torch.distributed.get_world_size())}, " f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (-1, -1)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    metric_function(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = metric_function.aggregate().item()
                # reset the status for next validation round
                metric_function.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    if args.distributed:
                        if torch.distributed.get_rank() == 0:
                            torch.save(model.module.state_dict(), os.path.join(model_save_dir, "best_metric_model.pth"))
                            print("saved new best metric model")
                    else:
                        torch.save(model.state_dict(), os.path.join(model_save_dir, "best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
    print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
    return epoch_loss_values, metric_values