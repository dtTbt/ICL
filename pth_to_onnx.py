from main import *

config = read_config_file('config.yaml')
pth_path = config['pth_path']
split_result = pth_path.split("/")

train_model = split_result[-1][:3]
number_categories = config['number_categories']
model = None
if test_model == 'MBN':
    model = MobileNetClassifier(number_categories)
if test_model == 'RES':
    model = ResnetClassifier(number_categories)
if test_model == 'SFN':
    model = ShuffleNetClassifier(number_categories)
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

