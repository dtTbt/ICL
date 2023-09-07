import yaml
import os
import torchvision.transforms as transforms
from PIL import Image
import torch


def read_config_file(config_file_path):
    with open(config_file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_cpu_temperature():
    try:
        res = os.popen('vcgencmd measure_temp').readline()
        temp = float(res.replace("temp=", "").replace("'C\n", ""))
        return temp
    except:
        return None


transform_1 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def image_read(image_path, trs):
    image = Image.open(image_path).convert("RGB")
    image_tensor = trs(image)
    return image_tensor


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha  # 意义为学习率缩放倍数

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class SchedulerMachine:
    def __init__(self, schedulers, times):
        self.schedulers = schedulers
        self.times = times

    def step(self, epoch):
        for i in range(len(self.times)):
            if self.times[i][0] <= epoch < self.times[i][1]:
                self.schedulers[i].step()
