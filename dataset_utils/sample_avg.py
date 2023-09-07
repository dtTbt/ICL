import yaml
import os

def read_config_file(config_file_path):
    with open(config_file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

config = read_config_file('../config.yaml')

dataset_path = config['dataset_path']
pth_train = os.path.join(dataset_path, "train")
pth_val = os.path.join(dataset_path, "val")

folder_names = os.listdir(pth_train)

for folder_name in folder_names:
    folder_path_train = os.path.join(pth_train, folder_name)
    folder_path_val = os.path.join(pth_val, folder_name)
    img_names_train = os.listdir(folder_path_train)
    img_names_val = os.listdir(folder_path_val)
    img_names_train_n = len(img_names_train)
    img_names_val_n = len(img_names_val)
    print(f"{img_names_train_n} {img_names_val_n}")