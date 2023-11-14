import torch
import matplotlib.pyplot as plt

def training(model, train_loader, val_loader, lr, batch_size, max_epochs, val_interval, device, optimizer, scheduler, spearman_loss, spearman_hard_eval, debug_flag=False):

    best_metric = -10
    metric_val = []
    loss_val = []
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
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
            output = model(inputs)
            # print(output)
            loss = spearman_loss(output.t(), labels.t())
            loss.backward()
            # print('output before sigmoid', outputs["SEResNet50_output"])
            # print('gradient of the las layer of SEResNet50', gradients["SEResNet50_output"])
            # break #####################################################################################################################

            optimizer.step()
            epoch_loss += loss.item()
            # if step ==2:#########################################################################################################
            #     break
            # # print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        # Step the scheduler
        if scheduler is not None:
            scheduler.step()
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        # break #####################################################################################################################
    
        if (epoch+1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_spearman_metric = 0
                val_step = 0
                for val_data in val_loader:
                    val_images, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    val_outputs = model(val_images)
                    val_loss = spearman_hard_eval(val_outputs.t(), val_labels.t())
                    val_spearman_metric += val_loss
                    val_step += 1
                val_spearman_metric /= val_step
                if val_spearman_metric > best_metric:
                    best_metric = val_spearman_metric
                    if scheduler is not None:
                        torch.save(model.state_dict(), 
                                   "spearman_loss_best_metric_model_lr_{}_batchsize_{}_lr_decay_{}_{}_{}.png".format(
                                       lr, batch_size, scheduler.__class__.__name__, scheduler.gamma, scheduler.step_size))
                    else:
                        torch.save(model.state_dict(), "spearman_loss_best_metric_model_lr_{}_batchsize_{}.pth".format(lr, batch_size))
                print(f"current epoch: {epoch + 1} current spearman: {val_spearman_metric:.4f} best spearman: {best_metric:.4f}")
                metric_val.append(val_spearman_metric.cpu())
                loss_val.append(epoch_loss)
    # break #########################################################################################################################
    # Plot the loss curve
    
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i*val_interval for i in range(len(loss_val))]
    plt.xlabel("epoch")
    plt.plot(x, loss_val, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Epoch Average Spearman")
    x = [i*val_interval for i in range(len(metric_val))]
    plt.xlabel("epoch")
    plt.plot(x, metric_val, color="red")
    if debug_flag:
        plt.show()
    if scheduler is not None:
        plt.savefig("spearman_loss_lr_{}_batchsize_{}_lr_decay_{}_{}_{}.png".format(lr, batch_size, scheduler.__class__.__name__, scheduler.gamma, scheduler.step_size))
    else:
        plt.savefig("spearman_loss_lr_{}_batchsize_{}.png".format(lr, batch_size))
    plt.close("train")

    # save the loss and metric
    if scheduler is not None:
        with open("spearman_loss_lr_{}_batchsize_{}_lr_decay_{}_{}_{}.txt".format(lr, batch_size, scheduler.__class__.__name__, scheduler.gamma, scheduler.step_size), "w") as f:
            for i in loss_val:
                f.write(str(i))
                f.write("\n")
        with open("spearman_metric_lr_{}_batchsize_{}_lr_decay_{}_{}_{}.txt".format(lr, batch_size, scheduler.__class__.__name__, scheduler.gamma, scheduler.step_size), "w") as f:
            for i in metric_val:
                f.write(str(i))
                f.write("\n")
    else:
        with open("spearman_loss_lr_{}_batchsize_{}.txt".format(lr, batch_size), "w") as f:
            for i in loss_val:

                f.write(str(i))
                f.write("\n")
        with open("spearman_metric_lr_{}_batchsize_{}.txt".format(lr, batch_size), "w") as f:
            for i in metric_val:
                f.write(str(i))
                f.write("\n")