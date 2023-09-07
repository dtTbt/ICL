import os
from PIL import Image
import torchvision.transforms as transforms
import concurrent.futures
import threading
import time

pth = "data_265"
pth_train = os.path.join(pth, "train")
pth_val = os.path.join(pth, "val")
pth_list = [pth_train, pth_val]

folder_names = os.listdir(pth_train)

transform_list =[]

transform_0 = transforms.Compose([
    transforms.RandomRotation(1)
])
transform_list.append(transform_0)
transform_1 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.RandomVerticalFlip(p=0.6),
    transforms.RandomRotation(45)
])
transform_list.append(transform_1)
transform_2 = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.35, contrast=0.3, saturation=0.3, hue=0.1)
])
transform_list.append(transform_2)
transform_3 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.RandomVerticalFlip(p=0.6),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.35, contrast=0.3, saturation=0.3, hue=0.1)
])
transform_list.append(transform_3)

total_n = 2 * len(folder_names) * len(transform_list)

class FolderCounter:
    def __init__(self):
        self.lock = threading.Lock()
        self.count = 0

    def increment(self):
        with self.lock:
            self.count += 1

    def get_count(self):
        with self.lock:
            return self.count

def is_image_corrupted(image_path):
    try:
        with Image.open(image_path) as img:
            return False
    except (OSError, ValueError):
        return True

def apply_transform(image_path, output_path, trs):
    if is_image_corrupted(image_path):
        os.remove(image_path)
    else:
        img = trs(Image.open(image_path).convert('RGB'))
        img.save(output_path)

def process_folder(folder_path, folder_path_out, counter, trs, index):
    start_time = time.time()
    img_names = os.listdir(folder_path)
    for img_name in img_names:
        input_path = os.path.join(folder_path, img_name)
        output_path = os.path.join(folder_path_out, img_name[:-4] + '_t' + str(index) + '.jpg')
        apply_transform(input_path, output_path, trs)
    elapsed_time = time.time() - start_time
    counter.increment()
    print(f"{counter.get_count()} finish. (total {total_n}) cost time: {elapsed_time}s")

if __name__ == "__main__":
    folder_counter = FolderCounter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for pth in pth_list:
            pth_tmp = pth.strip().split('/')
            pth_tmp[-2] = pth_tmp[-2] + '_x4'
            pth_out = './'
            for pth_a in pth_tmp:
                pth_out = os.path.join(pth_out, pth_a)
            for folder_name in folder_names:
                folder_path = os.path.join(pth, folder_name)
                folder_path_out = os.path.join(pth_out, folder_name)
                if not os.path.exists(folder_path_out):
                    os.makedirs(folder_path_out)
                for index, trs in enumerate(transform_list):
                    executor.submit(process_folder, folder_path, folder_path_out, folder_counter, trs, index)
