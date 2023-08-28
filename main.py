import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # 导入tqdm库


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx], self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ICL(nn.Module):
    def __init__(self):
        super(ICL, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.linear = nn.Linear(1000, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_path = '../垃圾/dataset/Kaggle_1/dataset'
dataset_path_train = os.path.join(dataset_path, 'train')
dataset_path_test = os.path.join(dataset_path, 'test')

preprocess = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
])

number_categories = 2
# 初始化列表用于暂存图像和标签数据
images_list = []
labels_list = []

# 遍历每个类别文件夹
for kind in range(number_categories):
    image_one_kind_folder_path = os.path.join(dataset_path_train, str(kind))
    image_name_list = [image_name for image_name in os.listdir(image_one_kind_folder_path) if image_name.endswith('.jpg') or image_name.endswith('.png')]
    for image_name in tqdm(image_name_list):
        image_path = os.path.join(image_one_kind_folder_path, image_name)
        image = Image.open(image_path).convert("RGB")  # 将图像转换为RGB
        image_tensor = transform(image)
        images_list.append(image_tensor)
        labels_list.append(kind)

# 将列表转换为张量
images = torch.stack(images_list)
labels = torch.tensor(labels_list)

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)


train_dataset = CustomDataset(images, labels)
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = ICL()
model.train()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 40
for epoch in range(epochs):
    for batch_index, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        print('epoch: {}, batch_index: {}, loss is: {}'.format(epoch, batch_index, loss.item()))