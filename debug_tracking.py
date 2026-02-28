import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO("best.pt")
tracker = sv.ByteTrack()

frame = cv2.imread("uploaded_image.jpg")
bgr_frame = frame.copy()

results = model(bgr_frame, conf=0.3)[0]
detections = sv.Detections.from_ultralytics(results)
print("Detections from YOLO:", len(detections))

detections2 = tracker.update_with_detections(detections)
print("Detections after tracker:", len(detections2))
