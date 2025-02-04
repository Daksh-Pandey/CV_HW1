import cv2
import numpy as np
import pandas as pd
import pickle
import torch


import motmetrics as mm

import os

import time
from byte.byte_tracker import BYTETracker
from IOU_Tracker import IOUTracker





def get_image_frames(image_dir):
    """
    Reads frames (images) from a directory.

    Args:
        image_dir (str): Path to the directory containing images.

    Returns:
        list: A list of image frames (numpy arrays).
    """
    
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    if not image_files:
        print(f"No images found in {image_dir}.")
        return []
    
    frames = []
    for file in image_files:
        frame = cv2.imread(file)
        if frame is None:
            print(f"Error loading image: {file}")
        else:
            frames.append(frame)
    
    return frames


def load_mot_detections(det_path):
    """
    Load MOT format detections from a file.

    Args:
        det_path (str): Path to the detection file.
        det.txt format = [frame,id,x_left,y_top,w,'h',score]

    Returns:
        list: A list of detections, where each detection follows the format:
              [frame, xmin, ymin, xmax, ymax, score, class]
    """
    detections = []

    with open(det_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')

            # TODO: Extract values from `parts` and convert them to appropriate data types
            frame = None  # Replace None with extracted frame number
            x = None  # Replace None with extracted x coordinate
            y = None  # Replace None with extracted y coordinate
            w = None  # Replace None with extracted width
            h = None  # Replace None with extracted height
            score = None  # Replace None with extracted confidence score

            # TODO: Add a condition to skip invalid detections (e.g., if x < 0 or y<0 )
            
            cls = 0  # Default class (can be modified if needed)

            # TODO: Append the detection in the correct format
            # detections.append([...])

    return detections



def real_time_dataset(frames, detections, fps=30):
    time_per_frame = 1 / fps
    for frame_idx, frame in enumerate(frames):
        # Get detections for the current frame
        frame_detections = [d for d in detections if d[0] == frame_idx + 1]
        yield frame, frame_detections  # Yield current frame and its detections
        time.sleep(time_per_frame) 


#Usable with IOU tracker or ByteTracker
def run_tracker(tracker,frames, detections, fps=30):
    
    
    
    tracked_objects = []
    

    # Initialize video writer for output
    

    # Simulate real-time frame input
    frame_gen = real_time_dataset(frames, detections, fps)

    for frame_idx, (frame, frame_detections) in enumerate(frame_gen):
        if len(frame_detections) == 0:
            continue

        # Convert detections to numpy array.

        detection_array = None

        # Update tracker
        online_tracks = None

        for track in online_tracks:
            
            track_id = None
            
            x = None
            y = None
            w = None
            h = None
            cls = None
            score = None

            
        

            
            tracked_objects.append([frame_idx + 1, track_id, x, y, w, h, score, cls])

        
    
    return tracked_objects




def evaluate_tracking(gt_path, tracked_objects):
    gt_data = pd.read_csv(gt_path, header=None, names=["frame", "id", "x", "y", "w", "h", "conf", "class", "visibility"])
    print(len(gt_data))
    gt_data = gt_data[gt_data["conf"] != 0]
    print(len(gt_data))
    gt_data = gt_data[gt_data["y"] != -1]
    track_df = pd.DataFrame(tracked_objects, columns=["frame", "id", "x", "y", "w", "h","conf","class"])

    track_df.to_csv('output.txt',sep=',',index=False)
    

    acc = mm.MOTAccumulator(auto_id=True)
    for frame in sorted(gt_data["frame"].unique()):
        gt_frame = gt_data[gt_data["frame"] == frame]
        pred_frame = track_df[track_df["frame"] == frame]

        gt_ids = gt_frame["id"].values
        pred_ids = pred_frame["id"].values
        gt_boxes = gt_frame[["x", "y", "w", "h"]].values
        pred_boxes = pred_frame[["x", "y", "w", "h"]].values

        distances = mm.distances.iou_matrix(gt_boxes,pred_boxes,max_iou=0.5)

        acc.update(gt_ids, pred_ids, distances)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name="Overall")
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    mota_score = summary.loc["Overall", "mota"]
    
   
    return mota_score


if __name__ == "__main__":
    # Paths to the video and ground truth
    
    image_path = None
    gt_path = None
    det_path= None
    

    
    frames = get_image_frames(image_path)
    print(f"Loaded {len(frames)} frames from video.")

   
    detections = load_mot_detections(det_path)
    print(f"Detections generated: {len(detections)}")

    
    byte=None # use the default parameters for ByteTracker.
    iou_tracker=IOUTracker(iou_threshold=0.8) # Do not change the iou_threshold.
   
    #get the tracked objects by using run_tracker.
    tracked_objects = run_tracker(byte,frames, detections)
    
    print(f"Tracking results generated: {len(tracked_objects)}")
    #get the result using evaluating_track.
