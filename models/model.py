import torch.nn as nn
import torch
from torchvision.models import resnet50
from monai.networks.nets import SEResNet50

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50(pretrained=False)
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:9])
                        
    def forward(self, x):
        return self.backbone(x)
    

class Classifier(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.drop_out = nn.Dropout()
        self.linear = nn.Linear(2048, num_class)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.linear(x)
        #x = torch.softmax(x, dim=-1)
        return x


class MyModel(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.seresnet50 = SEResNet50(spatial_dims = 2, in_channels =3 ,pretrained=False)
        self.drop_out = nn.Dropout()
        self.linear = nn.Linear(1000, num_class)

    def forward(self, x):
        x = self.seresnet50(x)
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    model = MyModel(num_class=10)
    # model.load_state_dict(torch.load("../../RadImageNet_pytorch/ResNet50.pt"))
    print(model)
    
