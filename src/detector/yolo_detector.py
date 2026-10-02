import torch
import numpy as np
from ultralytics import YOLO


class YOLOObjectDetector:
    """
    Object detector powered by Ultralytics YOLOv8 / YOLOv11.
    Optimized for high detection accuracy (DetA) on person class (COCO class 0).
    """
    def __init__(self, model_name="yolov8m.pt", conf_thresh=0.35, iou_thresh=0.45, classes=[0], device=None):
        self.model_name = model_name
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.classes = classes
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[YOLOObjectDetector] Loading {model_name} on device: {self.device}")
        self.model = YOLO(model_name)

    def detect(self, frame):
        """
        Detect persons in a single image/frame.
        Returns numpy array of shape (N, 6): [x1, y1, x2, y2, conf, class_id]
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=self.classes,
            device=self.device,
            verbose=False
        )
        
        if len(results) == 0 or results[0].boxes is None:
            return np.empty((0, 6), dtype=np.float32)
            
        boxes = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
        return boxes
