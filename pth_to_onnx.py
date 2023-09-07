import os.path
import torch.onnx
from main import MobileNetClassifier
import yaml
import main

def read_config_file(config_file_path):
    with open(config_file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

config = read_config_file('config.yaml')
pth_path = config['pth_path']
split_result = pth_path.split("/")

# Initialize your model and load its weights
train_model = split_result[-1][:3]
number_categories = config['number_categories']
model = None
if train_model == 'MBN':
    model = main.MobileNetClassifier(number_categories)
if train_model == 'RES':
    model = main.ResnetClassifier(number_categories)
model.load_state_dict(torch.load(pth_path))
model.eval()
batch_size_onnx = config['batch_size_onnx']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn(batch_size_onnx, 3, 224, 224).to(device)
model.to(device)

onnx_name = split_result[-1][:-4] + '_bs' + str(batch_size_onnx) + '.onnx'
onnx_folder_path = config['onnx_folder_path']
if not os.path.exists(onnx_folder_path):
    os.makedirs(onnx_folder_path)
onnx_path = os.path.join(onnx_folder_path, onnx_name)
if os.path.exists(onnx_path):
    print(f"{onnx_path} already exists, please clean it up first.")
    exit()

torch.onnx.export(model, dummy_input, onnx_path, verbose=True)
print(f"ONNX model has been saved to {onnx_path}")

