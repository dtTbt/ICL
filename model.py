import torch.nn as nn
import torchvision.models as models
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ResnetClassifier(nn.Module):
    def __init__(self, num_kinds):
        super(ResnetClassifier, self).__init__()
        self.model_name = 'RES'
        self.resnet = models.resnet18(pretrained=True)
        self.net = nn.Sequential(
            nn.Linear(1000, num_kinds),
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.net(x)
        return x


class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetClassifier, self).__init__()
        self.model_name = 'MBN'
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        self.net = nn.Sequential(
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        x = self.mobilenet_v2(x)
        x = self.net(x)
        return x


class ShuffleNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ShuffleNetClassifier, self).__init__()
        self.model_name = 'SFN'
        self.shuffle_net = models.shufflenet_v2_x1_0(pretrained=True)
        self.net = nn.Sequential(
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        x = self.shuffle_net(x)
        x = self.net(x)
        return x
