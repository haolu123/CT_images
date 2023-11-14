import re
import torch
import torch.nn as nn
from monai.networks.nets import UNet,SEResNet50

def load_state_dict(model: nn.Module, pth_file: str):
    """
    This function is used to load pretrained models.
    """

    pattern_conv = re.compile(r"^(layer[1-4]\.\d\.(?:conv)\d\.)(\w*)$")
    pattern_bn = re.compile(r"^(layer[1-4]\.\d\.)(?:bn)(\d\.)(\w*)$")
    pattern_se = re.compile(r"^(layer[1-4]\.\d\.)(?:se_module.fc1.)(\w*)$")
    pattern_se2 = re.compile(r"^(layer[1-4]\.\d\.)(?:se_module.fc2.)(\w*)$")
    pattern_down_conv = re.compile(r"^(layer[1-4]\.\d\.)(?:downsample.0.)(\w*)$")
    pattern_down_bn = re.compile(r"^(layer[1-4]\.\d\.)(?:downsample.1.)(\w*)$")
    state_dict = torch.load(pth_file)
    for key in list(state_dict.keys()):
        new_key = None
        if pattern_conv.match(key):
            new_key = re.sub(pattern_conv, r"\1conv.\2", key)
        elif pattern_bn.match(key):
            new_key = re.sub(pattern_bn, r"\1conv\2adn.N.\3", key)
        elif pattern_se.match(key):
            state_dict[key] = state_dict[key].squeeze()
            new_key = re.sub(pattern_se, r"\1se_layer.fc.0.\2", key)
        elif pattern_se2.match(key):
            state_dict[key] = state_dict[key].squeeze()
            new_key = re.sub(pattern_se2, r"\1se_layer.fc.2.\2", key)
        elif pattern_down_conv.match(key):
            new_key = re.sub(pattern_down_conv, r"\1project.conv.\2", key)
        elif pattern_down_bn.match(key):
            new_key = re.sub(pattern_down_bn, r"\1project.adn.N.\2", key)
        if new_key:
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    model_dict = model.state_dict()
    state_dict = {
        k: v for k, v in state_dict.items() if (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)
    }
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

def SEResNet50_model(flag_pretrain, spatial_dims=2, in_channels=1, pretrained=False, num_classes=1):
    model = SEResNet50(spatial_dims, in_channels, pretrained, num_classes)
    if flag_pretrain:
        seresnet50_file = '/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project2_CT_image/codes/CT_age/RadImageNet_pytorch/se_resnet50-ce0d4300.pth'
        load_state_dict(model, seresnet50_file)
    
    return model