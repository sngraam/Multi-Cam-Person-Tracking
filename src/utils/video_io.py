import os
import cv2
import numpy as np


class VideoReader:
    """Robust video reader for single or multiple video files."""
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found at: {path}")
        
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video stream for {path}")
            
        self.fps = int(round(self.cap.get(cv2.CAP_PROP_FPS))) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_all_frames(self):
        """Read all frames from the video file into a list of BGR numpy arrays."""
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames

    def release(self):
        if self.cap is not None:
            self.cap.release()


class VideoWriter:
    """High compatibility video writer supporting mp4v, avc1, MJPG codecs for Gradio / web player compatibility."""
    def __init__(self, output_path, fps=30, frame_size=None):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None

    def _init_writer(self, width, height):
        ext = os.path.splitext(self.output_path)[1].lower()
        if ext == '.mp4':
            fourcc_list = ['mp4v', 'avc1', 'H264']
        else:
            fourcc_list = ['MJPG', 'XVID']

        for code in fourcc_list:
            try:
                fourcc = cv2.VideoWriter_fourcc(*code)
                writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
                if writer.isOpened():
                    self.writer = writer
                    return
            except Exception:
                continue

        # Fallback
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))

    def write(self, frame):
        if self.writer is None:
            h, w = frame.shape[:2]
            self._init_writer(w, h)
        self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def interpolate_trajectories(tracklets, max_gap=15):
    """
    Interpolates missing bounding box detections in a trajectory to improve HOTA/MOTA metrics.
    tracklets: dict mapping track_id -> list of [frame_idx, x1, y1, x2, y2]
    """
    interpolated_tracklets = {}
    
    for track_id, records in tracklets.items():
        if len(records) < 2:
            interpolated_tracklets[track_id] = records
            continue
            
        records = sorted(records, key=lambda r: r[0])
        filled_records = []
        
        for i in range(len(records) - 1):
            curr_rec = records[i]
            next_rec = records[i + 1]
            filled_records.append(curr_rec)
            
            frame_gap = next_rec[0] - curr_rec[0]
            if 1 < frame_gap <= max_gap:
                for step in range(1, frame_gap):
                    alpha = step / float(frame_gap)
                    interp_frame = curr_rec[0] + step
                    interp_x1 = int(curr_rec[1] + alpha * (next_rec[1] - curr_rec[1]))
                    interp_y1 = int(curr_rec[2] + alpha * (next_rec[2] - curr_rec[2]))
                    interp_x2 = int(curr_rec[3] + alpha * (next_rec[3] - curr_rec[3]))
                    interp_y2 = int(curr_rec[4] + alpha * (next_rec[4] - curr_rec[4]))
                    filled_records.append([interp_frame, interp_x1, interp_y1, interp_x2, interp_y2])
                    
        filled_records.append(records[-1])
        interpolated_tracklets[track_id] = filled_records

    return interpolated_tracklets
