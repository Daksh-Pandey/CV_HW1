import pytest
import pickle
from tracking import run_tracker, evaluate_tracking, load_mot_detections, get_image_frames



import pickle


import motmetrics as mm


from byte.byte_tracker import BYTETracker
from IOU_Tracker import IOUTracker



TEST_CASES_BYTE = [
   {
       "det_path": "train/MOT17-11-SDP/det/det.txt",
       "gt_path": "train/MOT17-11-SDP/gt/gt.txt", 
       "img_path": "train/MOT17-11-SDP/img1/",
       "expected_mota": 65.6
   },
   {
       "det_path": "train/MOT17-13-SDP/det/det.txt",
       "gt_path": "train/MOT17-13-SDP/gt/gt.txt",
       "img_path": "train/MOT17-13-SDP/img1",
       "expected_mota": 49.4
   }
   
]
TEST_CASES_IOU = [
   {
       "det_path": "train/MOT17-11-SDP/det/det.txt",
       "gt_path": "train/MOT17-11-SDP/gt/gt.txt", 
       "img_path": "train/MOT17-11-SDP/img1/",
       "expected_mota": 44.5
   },
   {
       "det_path": "train/MOT17-13-SDP/det/det.txt",
       "gt_path": "train/MOT17-13-SDP/gt/gt.txt",
       "img_path": "train/MOT17-13-SDP/img1",
       "expected_mota": 4.7
   }
  
   
]
@pytest.mark.parametrize("test_case", TEST_CASES_IOU)
def test_tracking_performance_IOU(test_case):
   with open("IouTracker.pkl", "rb") as f:
       tracker = pickle.load(f)
   
   frames = get_image_frames(test_case["img_path"])
   detections = load_mot_detections(test_case["det_path"])
   
   tracked_objects = run_tracker(tracker, frames, detections)
   pred_score = evaluate_tracking(test_case["gt_path"], tracked_objects) * 100
   
   assert (abs(pred_score - test_case["expected_mota"])<=1 or pred_score>=test_case['expected_mota'])

@pytest.mark.parametrize("test_case", TEST_CASES_BYTE)
def test_tracking_performance(test_case):
   with open("Byte.pkl", "rb") as f:
       byte = pickle.load(f)
   
   frames = get_image_frames(test_case["img_path"])
   detections = load_mot_detections(test_case["det_path"])
   
   tracked_objects = run_tracker(byte, frames, detections)
   pred_score = evaluate_tracking(test_case["gt_path"], tracked_objects) * 100
   
   assert (abs(pred_score - test_case["expected_mota"])<=1 or pred_score>=test_case['expected_mota'])
   
