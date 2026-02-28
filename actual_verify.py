import cv2
import app

# just manually override the mock
from depth_estimator import DepthEstimator
app.depth_estimator = DepthEstimator()

img = cv2.imread('uploaded_image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
annotated, pothole_info = app.callback(img_rgb, 0.3)
print("Detected potholes without mock:", len(pothole_info))
