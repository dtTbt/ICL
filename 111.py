import torch
import torchvision.models as models

# 加载预训练的ResNet-18模型
resnet = models.resnet18(pretrained=True)

# 创建一个随机输入，模拟一个图像批次
input_batch = torch.randn(1, 3, 224, 224)  # 假设批次大小为1，图像尺寸为224x224

# 将输入传递给ResNet-18模型
output = resnet(input_batch)

print("Output shape:", output.shape)
