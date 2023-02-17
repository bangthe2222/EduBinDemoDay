import time
import cv2
import numpy as np
import onnxruntime
import os
from models.utils import xywh2xyxy, nms, draw_detections

class YOLOv8:
    """
        yolov8 onnx model detector
    """
    def __init__(self, path,class_names = ["bottle", "milo", "redbull"], conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        # Initialize model
        self.initialize_model(path)
        self.class_names = class_names

        # Create a list of colors for each class where each color is a tuple of 3 integer values
        rng = np.random.default_rng(3)
        self.colors = rng.uniform(0, 255, size=(len(class_names), 3))

    # def __call__(self, image):
    #     return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()


    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.2):

        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, self.colors, self.class_names, mask_alpha)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
    def getIdObject(self):
        id_out = None
        ID_list = [1,2,0]
        if len(self.class_ids):
            id_out = ID_list[self.class_ids[0]]
            return id_out
        else:
            return None

if __name__ == '__main__':
    model_path = "./EduBinYolov8_3_2_2023.onnx"
    class_names = ["bottle", "milo", "redbull"]
    # Initialize YOLOv8 object detector
    yolov8_detector = YOLOv8(model_path,  # path to onnx model 
                            class_names= class_names, # class names
                            conf_thres=0.5, 
                            iou_thres=0.5
                            )

    path = r"C:\Users\PC\Documents\HCMUT K21\EduBin\DEMO_DAY\CODE\Yolov8onnx\test\images\\"
    for file in os.listdir(path=path):

        img = cv2.imread(path + file)
        # Detect Objects
        yolov8_detector(img)

        # Draw detections
        combined_img = yolov8_detector.draw_detections(img)

        # Resize image
        out_img = cv2.resize(combined_img, ( 720,720))
        cv2.imshow("Output", out_img)
        cv2.waitKey(300)
