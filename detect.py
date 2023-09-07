import onnxruntime
import numpy as np
import serial
import cv2
from model import *
from utils import *
from classify_utils import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = read_config_file('config.yaml')

    onnx_model_path = config['onnx_model_path']
    detect_threshold = config['detect_threshold']
    number_categories = config['number_categories']
    class_name_file_path = config['classname_txt_path']
    session = onnxruntime.InferenceSession(
        onnx_model_path,
        providers=['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    )

    kind_dic = {"可回收物": "1", "厨余垃圾": "2", "有害垃圾": "3", "其他垃圾": "4"}

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

    ser = serial.Serial('/dev/ttyAMA0', 9600)
    if not ser.isOpen():
        ser.open()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    img_max_num = int(onnx_model_path.split('/')[-1][-6])

    print('Detecting...')
    while 1:
        start_time = time.time()
        frame_buffer = []
        img_num = 0
        while img_num < img_max_num:
            img_num += 1
            _, frame = cap.read()
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            transformed_frame = transform_1(frame_pil)
            frame_as_numpy = transformed_frame.unsqueeze(dim=0).cpu().numpy()
            frame_buffer.append(frame_as_numpy)
        batch_input = np.concatenate(frame_buffer)
        input_name = session.get_inputs()[0].name
        input_data = {input_name: batch_input}
        output = session.run(None, input_data)
        output = output[0]
        output = torch.from_numpy(output).to(device)
        pred = output.argmax(dim=1, keepdim=True)
        output = torch.nn.functional.softmax(output, dim=1)
        prt_message = None
        count_array = np.zeros(number_categories + 1, dtype=int)
        for i in range(img_max_num):
            index = pred[i].item()
            if output[i][index] > detect_threshold:
                count_array[index] += 1
            else:
                count_array[-1] += 1
        count_array_max_index = count_array.argmax()
        if count_array_max_index < number_categories:
            kind_index = int(king_strs[count_array_max_index])
            prt_message = kind_list_1[kind_index] + ' ' + kind_list_0[kind_index]
            message = kind_dic[kind_list_0[kind_index]]
            ser.write(message.encode())
        else:
            prt_message = 'nothing'
        temperature = get_cpu_temperature()
        if temperature is not None:
            temperature_out = temperature
        else:
            temperature_out = -99.9
        time_sp = time.time() - start_time
        print('{} time:{:.3f} temperature:{:.1f}'.format(prt_message, time_sp, temperature_out))
