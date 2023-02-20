import serial
import time
ser = serial.Serial(port="COM7",baudrate=9600, timeout=0.1)

while True:

    x = ser.readline()
    print(x[:-2].decode("utf-8"))
    ser.flush()
    time.sleep(0.1)