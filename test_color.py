import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO('best.pt')
frame_bgr = cv2.imread('uploaded_image.jpg')
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

res_bgr = model(frame_bgr, conf=0.3)[0]
res_rgb = model(frame_rgb, conf=0.3)[0]

print("Detections on BGR:", len(res_bgr.boxes))
print("Detections on RGB:", len(res_rgb.boxes))
