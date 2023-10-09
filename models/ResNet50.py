import torch
from torch import nn
from torchvision.models import resnet50

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50(pretrained=False)
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:9])
                        
    def forward(self, x):
        return self.backbone(x)
    
if __name__ == "__main__":
    model = Backbone()
    model.load_state_dict(torch.load("../../RadImageNet_pytorch/ResNet50.pt"))
    print(model)
    