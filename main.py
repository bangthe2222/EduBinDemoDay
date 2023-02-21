import os
import kivy
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget

kivy.require('2.0.0')
print(kivy.__version__)

from kivy.clock import Clock
from kivy.app import App

from kivy.graphics import Rectangle
from kivy.core.image import Image
from kivy.uix.image import Image as KvImage
from kivy.uix.button import Button
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
import datetime
from kivy.uix.screenmanager import ScreenManager, Screen, WipeTransition
import cv2
from kivy.graphics.texture import Texture
import playsound 
import serial
import tensorflow as tf
import numpy as np
import time
date_Time = datetime.datetime.now()
date_Time_Str = str(date_Time)

Builder.load_string('''
<FancyButton>:
    background_normal: ''
    background_color: 44/255, 172/255, 61/255
    text_color: 1,0,0,1
    color: 122/255, 245/255, 51/255

''')

PATH_DETECT_IMAGE = ""
class FancyButton(Button):
    opacity = 0.87


class CameraWidget(Widget):
    def __init__(self, capture, **kwargs):
        super(CameraWidget, self).__init__(**kwargs)

        self.capture = capture
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        Clock.schedule_interval(self.update, 1.0 / 33.0)

    def update(self, dt):
        ret, frame = self.capture.read()
        frame = cv2.flip(frame, 0)

        if ret:
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(frame.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
            self.canvas.clear()
            with self.canvas:
                Rectangle(texture=texture, pos=self.pos, size=self.size)


class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super(CameraScreen, self).__init__(**kwargs)
        self.checkSensor = True
        # Tạo Widget layout
        layout = FloatLayout()

        # Tạo Widget hiển thị hình nền
        with layout.canvas:
            # Load the image and create a texture from it
            img = Image('2.jpg').texture
            # Create a Rectangle object with the texture as its source
            self.rect = Rectangle(texture=img, pos=layout.pos, size=layout.size)
        # Bind the texture and size properties of the Rectangle object
        # to the corresponding properties of the widget
        layout.bind(pos=self.update_rect, size=self.update_rect)

        # Tạo các widget và thêm chúng vào màn hình
        # tạo camera capture và widget camera
        self.cap = cv2.VideoCapture(1)
        camera_widget = CameraWidget(capture=self.cap)
        camera_widget.size_hint = (0.335, 0.5)
        camera_widget.pos_hint = {'top': 0.7, 'right': 0.497}
        layout.add_widget(camera_widget)

        # bắt đầu cập nhật khung hình camera
        Clock.schedule_interval(camera_widget.update, 1.0 / 30.0)
        self.ser = serial.Serial(port="COM7",baudrate=9600, timeout=0.1)
        # load model
        model_name = "UEH_vending_3class_20epoch_max.h5"
        self.model = tf.keras.models.load_model(model_name)
        Clock.schedule_interval(self.getSensor, 0.1)
        # Add FancyButton widget
        fancy_button = FancyButton(text='Chụp hình')
        fancy_button.size_hint = (None, None)
        fancy_button.size = (200, 50)
        fancy_button.pos_hint = {'center_x': 0.7, 'y': 0.3}
        fancy_button.bind(on_press=self.on_fancy_button_press)
        layout.add_widget(fancy_button)
        self.add_widget(layout)

        # config serial

        # yolov8 onnx

        # model_path = "./EduBinYolov8_10_2_2023.onnx"
        # class_names = ["bottle", "milk bottle", "metal"]
    
        # # Initialize YOLOv8 object detector
        # self.yolov8_detector = YOLOv8(model_path,  # path to onnx model 
        #                     class_names= class_names, # class names
        #                     conf_thres=0.8, # confidence threshold
        #                     iou_thres=0.5   # iou threshold 
        #                     )
    
    def getSensor(self, *args):
        self.ser.flush()
        self.ser.flushInput()
        self.ser.flushOutput()
        if self.checkSensor == True:
            x = self.ser.readline()
            self.ser.flush()
            print(x)
            data = x[:-2].decode("utf-8")
            if data == "0":
                self.checkSensor = False
                Clock.schedule_once(self.on_fancy_button_press, 1)
                

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def on_fancy_button_press(self, *args):
        # Do something when FancyButton is pressed
        # time.sleep(0.5)
        directory = "./images/"
        if not os.path.exists(directory):
            os.makedirs(directory)
            # Kết nối với camera

        # Đọc khung hình từ camera
        ret, frame = self.cap.read()

        # detect object
        # self.yolov8_detector.detect_objects(frame)

        # # Get Object ID
        # id_out = self.yolov8_detector.getIdObject()
        # combined_img = self.yolov8_detector.draw_detections(frame)
        # out_img = cv2.resize(combined_img, (640,480))

        # resize image
        image_src = cv2.resize(frame.copy(),(224,224))
        image = np.asarray([image_src])

        # pridict image
        predict = self.model.predict(image)
        id_out = np.argmax(predict[0])

        date_time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = directory +  f"picture_{date_time_str}.png"
        
        global PATH_DETECT_IMAGE
        PATH_DETECT_IMAGE = filename.replace("./images/", "./used/")
        # Lưu khung hình thành file ảnh
        cv2.imwrite(filename, frame)
        # Giải phóng kết nối với camera
   
        print(id_out)
        print(predict[0][id_out])
        if predict[0][id_out] > 0.45:
            ketQua = id_out
            if ketQua == 0:
                self.manager.current = 'kim_loai'
                Clock.schedule_once(self.play_sound_kim_loai,1)
                
            elif ketQua == 2:
                self.manager.current = 'nhua'
                # playsound.playsound("chai_nhua.mp3")
                Clock.schedule_once(self.play_sound_nhua,1)
            elif ketQua == 1:
                self.manager.current = 'hop_sua'
                # playsound.playsound("hop_sua.mp3")
                Clock.schedule_once(self.play_sound_hop_sua,1)
        else:
            self.manager.current = 'rac_khac'
                # playsound.playsound("hop_sua.mp3")
            # Clock.schedule_once(self.play_sound_hop_sua,1)  
        Clock.schedule_once(self.reset_camera, 7)
        
        
    def reset_camera(self, *args):
        # Quay lại màn hình camera
        self.manager.current = 'camera_screen'
        self.checkSensor = True

    def play_sound_kim_loai(self, path):
        playsound.playsound("lon_nhom.mp3")
    def play_sound_nhua(self, path):
        playsound.playsound("chai_nhua.mp3")
    def play_sound_hop_sua(self, path):
        playsound.playsound("hop_sua.mp3")

class HopSua(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        with self.canvas:
            # Load the image and create a texture from it
            img = Image('3.png').texture
            # Create a Rectangle object with the texture as its source
            self.rect = Rectangle(texture=img, pos=self.pos, size=self.size)
        # Bind the texture and size properties of the Rectangle object

        # to the corresponding properties of the widget
        self.bind(pos=self.update_rect, size=self.update_rect)
        Clock.schedule_interval(self.update_detect_img, 0.5)
        
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
    def update_detect_img(self, dt):
        global PATH_DETECT_IMAGE
        self.small_img = KvImage(source=PATH_DETECT_IMAGE, size_hint=(0.4, 0.4))
        self.small_img.pos_hint = {'center_x': 0.3, 'y': 0.15}
        self.add_widget(self.small_img)

class KimLoai(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        with self.canvas:
            # Load the image and create a texture from it
            img = Image('4.png').texture
            # Create a Rectangle object with the texture as its source
            self.rect = Rectangle(texture=img, pos=self.pos, size=self.size)
        # Bind the texture and size properties of the Rectangle object
        # to the corresponding properties of the widget
        self.bind(pos=self.update_rect, size=self.update_rect)
        Clock.schedule_interval(self.update_detect_img, 0.5)
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
    def update_detect_img(self, dt):
        global PATH_DETECT_IMAGE
        self.small_img = KvImage(source=PATH_DETECT_IMAGE, size_hint=(0.4, 0.4))
        self.small_img.pos_hint = {'center_x': 0.3, 'y': 0.15}
        self.add_widget(self.small_img)
class Nhua(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        with self.canvas:
            # Load the image and create a texture from it
            img = Image('5.png').texture
            # Create a Rectangle object with the texture as its source
            self.rect = Rectangle(texture=img, pos=self.pos, size=self.size)
        # Bind the texture and size properties of the Rectangle object
        # to the corresponding properties of the widget
        self.bind(pos=self.update_rect, size=self.update_rect)
        Clock.schedule_interval(self.update_detect_img, 0.5)
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
    def update_detect_img(self, dt):
        global PATH_DETECT_IMAGE
        self.small_img = KvImage(source=PATH_DETECT_IMAGE, size_hint=(0.4, 0.4))
        self.small_img.pos_hint = {'center_x': 0.3, 'y': 0.15}
        self.add_widget(self.small_img)

class Khac(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        with self.canvas:
            # Load the image and create a texture from it
            img = Image('1.png').texture
            # Create a Rectangle object with the texture as its source
            self.rect = Rectangle(texture=img, pos=self.pos, size=self.size)
        # Bind the texture and size properties of the Rectangle object
        # to the corresponding properties of the widget
        self.bind(pos=self.update_rect, size=self.update_rect)
        Clock.schedule_interval(self.update_detect_img, 0.5)
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
    def update_detect_img(self, dt):
        global PATH_DETECT_IMAGE
        self.small_img = KvImage(source=PATH_DETECT_IMAGE, size_hint=(0.4, 0.4))
        self.small_img.pos_hint = {'center_x': 0.3, 'y': 0.15}
        self.add_widget(self.small_img)
class MyScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        with self.canvas:
            # Load the image and create a texture from it
            img = Image('1.png').texture
            # Create a Rectangle object with the texture as its source
            self.rect = Rectangle(texture=img, pos=self.pos, size=self.size)
        # Bind the texture and size properties of the Rectangle object

        # to the corresponding properties of the widget
        self.bind(pos=self.update_rect, size=self.update_rect)

        fancy_button = FancyButton(text='Bắt đầu')
        fancy_button.size_hint = (None, None)  # Chỉ định kích thước cố định cho nút
        fancy_button.size = (200, 50)  # Thiết lập kích thước cho nút
        fancy_button.pos_hint = {'center_x': 0.4, 'y': 0.1}  # Đặt nút ở giữa dưới khung hình
        fancy_button.bind(on_press=self.on_press)
        self.add_widget(fancy_button)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def on_press(self, *args):
        # tạm dừng thực thi chương trình trong 1 giây
        Clock.schedule_once(self.change_screen, 1)

    def change_screen(self, *args):
        # kiểm tra xem self.manager có khác None hay không
        if self.manager is not None:
            self.manager.current = 'camera_screen'


class MyApp(App):
    def build(self):
        sm = ScreenManager(transition=WipeTransition())
        sm.add_widget(MyScreen(name='my_screen'))
        sm.add_widget(CameraScreen(name='camera_screen'))
        sm.add_widget(HopSua(name='hop_sua'))
        sm.add_widget(KimLoai(name='kim_loai'))
        sm.add_widget(Nhua(name='nhua'))
        sm.add_widget(Khac(name='rac_khac'))
        return sm


if __name__ == '__main__':
    MyApp().run()
