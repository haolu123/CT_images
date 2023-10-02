import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

## copy the following functions in your code

def distributed_init():
    torch.distributed.init_process_group(
                                        backend='nccl',
                                        init_method="env://",
                                        world_size=int(os.environ['WORLD_SIZE']),
                                        rank=int(os.environ["RANK"])
                                        )
    
    torch.distributed.barrier()


def distributed_params():
    return int(os.environ['LOCAL_RANK'])


def set_device(local_rank_param, multi_gpu = True):
    """Returns the device

    Args:
        local_rank_param: Give the local_rank parameter output of distributed_params()
        multi_gpu: Defaults to True.

    Returns:
        Device: Name the output device value
    """
    
    if multi_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(local_rank_param))
            print('CUDA is available. Setting device to CUDA.')
        else:
            device = torch.device('cpu')
            print('CUDA is not available. Setting device to CPU.')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print('CUDA is available. Setting device to CUDA.')
        else:
            device = torch.device('cpu')
            print('CUDA is not available. Setting device to CPU.')
        
    return device


def multigpu_dataloader(in_data, batchsize):
    """
    Creates the dataloader object.

    Args:
        in_data: Input data object that returns (image, text) when indexed. This is the data which you normally give as parameter to Dataloader(in_data)
        batchsize (int): The batchsize on which dataloader will operate
    """
    
    train_sampler = DistributedSampler(dataset=in_data)
    dataloader_object = DataLoader(dataset=in_data, batch_size=batchsize, shuffle=False, sampler=train_sampler, num_workers=32, pin_memory=True)
    
    return dataloader_object
