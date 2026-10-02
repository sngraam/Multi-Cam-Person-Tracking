import torch
import numpy as np
from ultralytics import YOLO


class MultiObjectTracker:
    """
    High-HOTA Multi-Object Tracker wrapping ByteTrack / BoT-SORT.
    Protects identity persistence across occlusions using Kalman filtering and IoU association.
    """
    def __init__(self, model_name="yolov8m.pt", tracker_type="bytetrack", conf_thresh=0.35, iou_thresh=0.45, device=None):
        self.model_name = model_name
        self.tracker_type = tracker_type if tracker_type in ["bytetrack", "botsort"] else "bytetrack"
        self.tracker_config = f"{self.tracker_type}.yaml"
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[MultiObjectTracker] Initializing YOLO + {self.tracker_type.upper()} tracker on {self.device}")
        self.model = YOLO(model_name)

    def track_frame(self, frame, persist=True):
        """
        Track objects in a single frame.
        Returns list of tracks: [ [x1, y1, x2, y2, track_id, conf], ... ]
        """
        results = self.model.track(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=[0],  # Person class
            persist=persist,
            tracker=self.tracker_config,
            device=self.device,
            verbose=False
        )
        
        tracks = []
        if len(results) == 0 or results[0].boxes is None:
            return tracks
            
        boxes = results[0].boxes
        if boxes.id is None:
            return tracks
            
        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        
        for bbox, tid, conf in zip(xyxy, track_ids, confs):
            x1, y1, x2, y2 = bbox
            tracks.append([float(x1), float(y1), float(x2), float(y2), int(tid), float(conf)])
            
        return tracks
