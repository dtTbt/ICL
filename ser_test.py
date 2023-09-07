import serial
import time

# 创建串口对象
ser = serial.Serial('/dev/ttyAMA0', 9600)

# 待发送的字符串列表
messages = ["1", "2", "3", "4"]

while True:
    for message in messages:
        # 发送字符串并编码为字节
        ser.write(message.encode())
        print(f"已发送 {message}")
        time.sleep(4)
