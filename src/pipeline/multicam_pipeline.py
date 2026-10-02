import os
import cv2
import numpy as np
import yaml
from collections import defaultdict

from src.tracker.mot_tracker import MultiObjectTracker
from src.reid.osnet_reid import OSNetReIDExtractor
from src.utils.video_io import VideoReader, VideoWriter, interpolate_trajectories
from src.utils.visualization import draw_bounding_box, create_side_by_side_grid


class MultiCamTrackingPipeline:
    """
    State-of-the-Art Multi-Camera Person Tracking & Re-Identification Pipeline.
    Combines YOLOv8/v11 detection, ByteTrack single-camera association, and OSNet ReID cross-camera fusion.
    Optimized for high HOTA, MOTA, and IDF1 accuracy.
    """
    def __init__(self, config_path=None):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                'detector': {'model_name': 'yolov8m.pt', 'conf_threshold': 0.35, 'iou_threshold': 0.45},
                'tracker': {'tracker_type': 'bytetrack'},
                'reid': {'model_name': 'osnet_x1_0', 'reid_threshold': 0.25, 'min_frames_for_reid': 5},
                'pipeline': {'interpolate_missing_boxes': True, 'max_interpolation_gap': 15, 'fps': 30}
            }

        det_cfg = self.config.get('detector', {})
        trk_cfg = self.config.get('tracker', {})
        reid_cfg = self.config.get('reid', {})

        # Initialize tracker & ReID engine
        self.tracker = MultiObjectTracker(
            model_name=det_cfg.get('model_name', 'yolov8m.pt'),
            tracker_type=trk_cfg.get('tracker_type', 'bytetrack'),
            conf_thresh=det_cfg.get('conf_threshold', 0.35),
            iou_thresh=det_cfg.get('iou_threshold', 0.45)
        )
        
        self.reid_extractor = OSNetReIDExtractor(
            model_name=reid_cfg.get('model_name', 'osnet_x1_0')
        )
        
        self.reid_thresh = reid_cfg.get('reid_threshold', 0.25)
        self.min_frames_for_reid = reid_cfg.get('min_frames_for_reid', 5)

    def process_videos(self, video_paths, output_dir="videos/output"):
        """
        Process multiple video streams, perform tracking, extract ReID features,
        fuse global person identities, and render annotated output videos.
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[MultiCamTrackingPipeline] Processing {len(video_paths)} input video(s)...")

        video_readers = [VideoReader(path) for path in video_paths]
        all_frames_per_video = [vr.read_all_frames() for vr in video_readers]
        fps = video_readers[0].fps

        # Step 1: Perform MOT Tracking across all videos
        # Store tracking records: video_idx -> frame_idx -> list of [x1, y1, x2, y2, local_track_id, conf]
        video_track_records = []
        # Store crop images for ReID: (video_idx, local_track_id) -> list of crop images
        tracklet_crops = defaultdict(list)
        # Store tracklet bounding box history for interpolation & rendering: (video_idx, local_track_id) -> list of [frame_idx, x1, y1, x2, y2]
        tracklet_bbox_history = defaultdict(list)

        for v_idx, frames in enumerate(all_frames_per_video):
            print(f"--- Tracking Video {v_idx + 1}/{len(all_frames_per_video)} ({len(frames)} frames) ---")
            frame_records = []
            
            # Re-init tracker for separate video streams to get fresh local tracks
            local_tracker = MultiObjectTracker(
                model_name=self.config.get('detector', {}).get('model_name', 'yolov8m.pt'),
                tracker_type=self.config.get('tracker', {}).get('tracker_type', 'bytetrack'),
                conf_thresh=self.config.get('detector', {}).get('conf_threshold', 0.35),
                iou_thresh=self.config.get('detector', {}).get('iou_threshold', 0.45)
            )

            for f_idx, frame in enumerate(frames):
                tracks = local_tracker.track_frame(frame, persist=True)
                frame_records.append(tracks)

                h, w = frame.shape[:2]
                for trk in tracks:
                    x1, y1, x2, y2, local_tid, conf = trk
                    ix1, iy1, ix2, iy2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
                    
                    if ix2 > ix1 and iy2 > iy1:
                        crop = frame[iy1:iy2, ix1:ix2]
                        key = (v_idx, local_tid)
                        tracklet_crops[key].append(crop)
                        tracklet_bbox_history[key].append([f_idx, ix1, iy1, ix2, iy2])

            video_track_records.append(frame_records)

        # Step 2: Extract ReID embeddings for each tracklet
        print("\n--- Extracting Deep ReID Feature Embeddings ---")
        tracklet_features = {}
        for key, crops in tracklet_crops.items():
            if len(crops) >= self.min_frames_for_reid:
                # Subsample max 30 crops for feature averaging
                sampled_crops = crops[::max(1, len(crops) // 30)]
                feats = self.reid_extractor.extract_features_batch(sampled_crops)
                if len(feats) > 0:
                    mean_feat = np.mean(feats, axis=0)
                    mean_feat = mean_feat / (np.linalg.norm(mean_feat) + 1e-8)
                    tracklet_features[key] = mean_feat

        # Step 3: Hierarchical Cross-Video Global Identity Fusion (Clustering)
        print("\n--- Fusing Identities Across Cameras / Videos ---")
        unique_keys = list(tracklet_crops.keys())
        global_id_map = {}  # (video_idx, local_track_id) -> global_id
        next_global_id = 1

        for key in unique_keys:
            if key in global_id_map:
                continue
                
            global_id_map[key] = next_global_id
            
            if key in tracklet_features:
                feat1 = tracklet_features[key]
                v1, tid1 = key
                
                # Check distance with all unassigned tracklets in other videos / timeframes
                for other_key in unique_keys:
                    if other_key == key or other_key in global_id_map:
                        continue
                    if other_key in tracklet_features:
                        feat2 = tracklet_features[other_key]
                        dist = self.reid_extractor.compute_cosine_distance(feat1, feat2)[0, 0]
                        
                        if dist < self.reid_thresh:
                            # Match found! Assign same global ID
                            global_id_map[other_key] = next_global_id

            next_global_id += 1

        print(f"Total Unique Persons (Global IDs) Identified: {next_global_id - 1}")

        # Step 4: Trajectory Interpolation for smooth tracking
        if self.config.get('pipeline', {}).get('interpolate_missing_boxes', True):
            print("\n--- Performing Trajectory Interpolation (HOTA Boost) ---")
            interpolated_history = {}
            for key, recs in tracklet_bbox_history.items():
                max_gap = self.config.get('pipeline', {}).get('max_interpolation_gap', 15)
                interpolated_history[key] = interpolate_trajectories({key[1]: recs}, max_gap=max_gap)[key[1]]
            tracklet_bbox_history = interpolated_history

        # Step 5: Render Output Videos
        print("\n--- Rendering Annotated Videos ---")
        output_files = []
        
        # Render individual video streams with Global ID annotations
        for v_idx, frames in enumerate(all_frames_per_video):
            out_path = os.path.join(output_dir, f"tracking_video_{v_idx + 1}.mp4")
            writer = VideoWriter(out_path, fps=fps)

            # Build fast lookup for frame -> list of [global_id, x1, y1, x2, y2]
            frame_render_map = defaultdict(list)
            for (vid, ltid), records in tracklet_bbox_history.items():
                if vid == v_idx:
                    gid = global_id_map.get((vid, ltid), ltid)
                    for rec in records:
                        f_idx, x1, y1, x2, y2 = rec
                        frame_render_map[f_idx].append((gid, [x1, y1, x2, y2]))

            for f_idx, frame in enumerate(frames):
                annotated = frame.copy()
                for gid, bbox in frame_render_map[f_idx]:
                    draw_bounding_box(annotated, track_id=gid, bbox=bbox, text_prefix="Person")
                writer.write(annotated)

            writer.release()
            output_files.append(out_path)
            print(f"Saved: {out_path}")

        # Render combined multi-camera side-by-side video if multiple videos
        if len(all_frames_per_video) > 1:
            combined_path = os.path.join(output_dir, "combined_multicam_tracking.mp4")
            combined_writer = VideoWriter(combined_path, fps=fps)
            max_num_frames = max(len(f) for f in all_frames_per_video)

            for f_idx in range(max_num_frames):
                current_frame_grid = []
                for v_idx, frames in enumerate(all_frames_per_video):
                    if f_idx < len(frames):
                        frame = frames[f_idx].copy()
                        # Draw bounding boxes
                        for (vid, ltid), records in tracklet_bbox_history.items():
                            if vid == v_idx:
                                gid = global_id_map.get((vid, ltid), ltid)
                                for rec in records:
                                    if rec[0] == f_idx:
                                        draw_bounding_box(frame, track_id=gid, bbox=rec[1:], text_prefix="Person")
                    else:
                        frame = np.zeros_like(all_frames_per_video[0][0])
                    current_frame_grid.append(frame)

                grid_frame = create_side_by_side_grid(current_frame_grid)
                combined_writer.write(grid_frame)

            combined_writer.release()
            output_files.append(combined_path)
            print(f"Saved Combined Multi-Camera Video: {combined_path}")

        video_readers[0].release()
        return output_files
