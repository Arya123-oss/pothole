import cv2
import numpy as np
from PIL import Image

# Mock streamlit before importing app
import sys
from unittest.mock import MagicMock
sys.modules['streamlit'] = MagicMock()

import app

# test the callback using uploaded_image.jpg
img = cv2.imread('uploaded_image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

annotated, pothole_info = app.callback(img_rgb, 0.3)
print("Detected potholes after callback fix:", len(pothole_info))
