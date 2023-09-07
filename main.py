from torch.optim import lr_scheduler
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import *
from utils import *
from classify_utils import *
ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = read_config_file('config.yaml')

    dataset_path = config['dataset_path']
    number_categories = config['number_categories']  # 获取类别数
    mode = config['mode']
    pth_save_folder = config['pth_save_folder']
    class_name_file_path = config['classname_txt_path']

    kind_dic = {"可回收物": "1", "厨余垃圾": "2", "有害垃圾": "3", "其他垃圾": "4"}

    dataset_path_train = os.path.join(dataset_path, "train")
    dataset_path_val = os.path.join(dataset_path, "val")

    king_strs = []
    for i in range(number_categories):
        king_strs.append(str(i))
    king_strs = sorted(king_strs)
    kind_list_0 = {}
    kind_list_1 = {}
    with open(class_name_file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().split()
            index = int(line[0])
            kind_list_0[index], kind_list_1[index] = line[1], line[2]

    if mode == "test":
        image_test = config['image_test']
        pth_model_path = config['pth_model_path']
        model = None
        test_model = pth_model_path.split('/')[-1][:3]
        if test_model == 'MBN':
            model = MobileNetClassifier(number_categories)
        if test_model == 'RES':
            model = ResnetClassifier(number_categories)
        if test_model == 'SFN':
            model = ShuffleNetClassifier(number_categories)
        model.load_state_dict(torch.load(pth_model_path))
        model.to(device)
        model.eval()

        images_name_list = os.listdir(image_test)
        for image_name in images_name_list:
            image_path = os.path.join(image_test, image_name)
            image_tensor = image_read(image_path, transform_1)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            pred = output.argmax(dim=1, keepdim=True)
            index = pred.item()
            kind_0 = kind_list_0[int(king_strs[index])]
            kind_1 = kind_list_1[int(king_strs[index])]
            print(image_name, kind_0, kind_1)
    elif mode == 'train':
        train_model = config['train_model']
        model = None
        if train_model == 'MBN':
            model = MobileNetClassifier(number_categories)
        if train_model == 'RES':
            model = ResnetClassifier(number_categories)
        if train_model == 'SFN':
            model = ShuffleNetClassifier(number_categories)
        model.to(device)

        data_train = ImageFolder(dataset_path_train, transform=transform_1)
        data_val = ImageFolder(dataset_path_val, transform=transform_1)

        batch_size = config['batch_size']
        train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(data_val, batch_size=batch_size, shuffle=True, num_workers=4)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        epochs = config['epochs']
        val_start_epoch = 3
        warmup_epochs = 2
        warmup_iters = len(train_dataloader) * warmup_epochs  # 学习率线性warmup
        warmup_factor = 0.0003
        decay_gamma = 0.5  # 衰减率
        decay_epoch = 3  # 每多少个epoch衰减一次
        decay_step_size = len(train_dataloader) * decay_epoch
        scheduler_1 = warmup_lr_scheduler(optimizer, warmup_iters=warmup_iters, warmup_factor=warmup_factor)
        scheduler_2 = lr_scheduler.StepLR(optimizer, step_size=decay_step_size, gamma=decay_gamma)
        scheduler = [scheduler_1, scheduler_2]

        scheduler_machine = SchedulerMachine(scheduler, [[0, warmup_epochs], [warmup_epochs, epochs]])
        train_machine = ClassifyTrainMachine(model, device, epochs, val_start_epoch, train_dataloader, val_dataloader, loss_fn,
                                     optimizer, scheduler_machine, pth_save_folder)

        train_machine.train()
    elif mode == 'val':
        image_test = config['image_test']
        pth_model_path = config['pth_model_path']
        model = None
        test_model = pth_model_path.split('/')[-1][:3]
        if test_model == 'MBN':
            model = MobileNetClassifier(number_categories)
        if test_model == 'RES':
            model = ResnetClassifier(number_categories)
        if test_model == 'SFN':
            model = ShuffleNetClassifier(number_categories)
        model.load_state_dict(torch.load(pth_model_path))
        model.to(device)

        data_val = ImageFolder(dataset_path_val, transform=transform_1)
        val_dataloader = DataLoader(data_val, batch_size=100, shuffle=False, num_workers=4)
        loss_fn = nn.CrossEntropyLoss()

        classify_val(model, device, val_dataloader, loss_fn)

