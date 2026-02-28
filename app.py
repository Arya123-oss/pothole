import streamlit as st
import cv2
import numpy as np
import os
import supervision as sv
from ultralytics import YOLO
import time
from PIL import Image
import json
from datetime import datetime
from streamlit_geolocation import streamlit_geolocation
from depth_estimator import DepthEstimator

st.set_page_config(layout="wide")

SUBMISSIONS_DIR = "submissions"
SUBMISSIONS_FILE = "submissions.json"

if not os.path.exists(SUBMISSIONS_DIR):
    os.makedirs(SUBMISSIONS_DIR)

if not os.path.exists(SUBMISSIONS_FILE):
    with open(SUBMISSIONS_FILE, "w") as f:
        json.dump([], f)

def load_submissions():
    with open(SUBMISSIONS_FILE, "r") as f:
        return json.load(f)

def save_submission(metadata):
    subs = load_submissions()
    subs.append(metadata)
    with open(SUBMISSIONS_FILE, "w") as f:
        json.dump(subs, f, indent=4)


# Initialize the YOLO model and tracker
model = YOLO("best.pt")
tracker = sv.ByteTrack()
mask_annotator = sv.MaskAnnotator()


@st.cache_resource
def load_depth_estimator():
    """Load MiDaS model once and cache it across Streamlit reruns."""
    return DepthEstimator()


depth_estimator = load_depth_estimator()


def callback(frame: np.ndarray, confidence: float) -> tuple:
    """
    Perform detection, tracking, depth estimation, and severity classification.

    Returns:
        (annotated_frame, pothole_info_list)
    """
    # YOLO detection (expects BGR numpy array)
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = model(bgr_frame, conf=confidence)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    # Annotate with masks
    annotated = mask_annotator.annotate(frame.copy(), detections=detections)

    # Depth + severity analysis
    pothole_info = depth_estimator.analyze_frame(frame, detections)

    # Draw severity labels and depth on annotated frame
    for info in pothole_info:
        x1, y1, x2, y2 = map(int, info["bbox"])
        severity = info["severity"]
        score = info["score"]
        color = info["color"]

        # Label text
        label = f"{severity} ({score:.0%})"

        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Position label above the bounding box
        label_y = max(y1 - 10, th + 5)
        cv2.rectangle(annotated, (x1, label_y - th - 5), (x1 + tw + 8, label_y + 5), color, -1)
        cv2.putText(annotated, label, (x1 + 4, label_y), font, font_scale, (255, 255, 255), thickness)

        # Draw bounding box with severity color
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

    return annotated, pothole_info


def video_input(data_src, confidence):
    vid_file = None
    if data_src == 'Sample data':
        vid_file = "sample/vid.mp4"
        if not os.path.exists(vid_file):
            st.warning("Sample video not found. Please select 'Upload your own data' instead.")
            vid_file = None
    else:
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mov', 'avi', 'mkv', 'webm'])
        if vid_bytes:
            vid_file = "uploaded_video." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        # Display video metadata
        fps = 0
        st1, st2, st3 = st.columns(3)

        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        info_placeholder = st.empty()
        prev_time = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                st.write("Can't read frame, stream ended? Exiting ....")
                break

            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            processed_frame, pothole_info = callback(frame, confidence)
            output.image(processed_frame, caption="Pothole Detection Result", use_column_width=True)

            # Show pothole info for current frame
            if pothole_info:
                info_text = " | ".join(
                    [f"Pothole {i+1}: {p['severity']} ({p['score']:.0%})"
                     for i, p in enumerate(pothole_info)]
                )
                info_placeholder.info(info_text)
            else:
                info_placeholder.empty()

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time

            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")

        cap.release()


def public_reporter_view():
    st.header("📸 Report a Pothole")
    st.write("Help keep our roads safe by reporting potholes in your area.")

    # Location
    st.subheader("1. Get your location")
    st.write("Please allow location access so PWD can find the pothole.")
    location = streamlit_geolocation()

    # Image upload
    st.subheader("2. Upload or Capture Image")
    data_src = st.radio("Select input source:", ['Camera', 'Upload Image'])

    img_bytes = None
    if data_src == 'Camera':
        img_bytes = st.camera_input("Take a picture")
    else:
        img_bytes = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])

    if st.button("Submit Report", type="primary"):
        if not img_bytes:
            st.error("Please provide an image.")
            return

        if not location or not location.get('latitude') or not location.get('longitude'):
            st.warning("Location not found. Submitting without GPS coordinates.")
            lat, lon = None, None
        else:
            lat = location['latitude']
            lon = location['longitude']

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{img_bytes.name}" if hasattr(img_bytes, 'name') else f"{timestamp}_capture.jpg"
        filepath = os.path.join(SUBMISSIONS_DIR, filename)

        image = Image.open(img_bytes)
        # Convert to RGB to ensure standard format when saving
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(filepath)

        # Save metadata
        metadata = {
            "id": timestamp,
            "filepath": filepath,
            "latitude": lat,
            "longitude": lon,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "Pending Analysis"
        }
        save_submission(metadata)

        st.success("✅ Thank you for reporting! Your submission has been saved and will be reviewed by PWD officials.")
        st.balloons()


def official_login_view():
    st.header("🔒 PWD Official Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "pwd123":
            st.session_state["logged_in"] = True
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")


def official_dashboard_view(confidence):
    st.header("🛠️ PWD Official Dashboard")
    st.write("Review and analyze public pothole reports.")

    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

    submissions = load_submissions()
    if not submissions:
        st.info("No pothole submissions yet.")
        return

    # Display submissions
    st.subheader(f"Total Reports: {len(submissions)}")

    for idx, sub in enumerate(reversed(submissions)):
        with st.expander(f"Report from {sub['timestamp']} - Status: {sub.get('status', 'Pending')}"):
            st.write(f"**Location:** {sub.get('latitude', 'Unknown')}, {sub.get('longitude', 'Unknown')}")

            if not os.path.exists(sub['filepath']):
                st.error("Image file not found.")
                continue

            # Load image for processing
            image = cv2.imread(sub['filepath'])
            if image is None:
                st.error("Could not read image file.")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image_rgb, caption="Original User Submission")

            # Run Analysis
            with col2:
                with st.spinner("Analyzing image..."):
                    processed_image, pothole_info = callback(image_rgb, confidence)
                    st.image(processed_image, caption="AI Detection + Severity Result")

            # Display stats
            if pothole_info:
                st.markdown("---")
                st.subheader("🕳️ Analysis Results")

                for i, info in enumerate(pothole_info):
                    severity = info["severity"]
                    score = info["score"]

                    if severity == "Low":
                        color_hex = "#28a745"
                    elif severity == "Medium":
                        color_hex = "#fd7e14"
                    else:
                        color_hex = "#dc3545"

                    x1, y1, x2, y2 = map(int, info["bbox"])
                    width_px = x2 - x1
                    height_px = y2 - y1
                    size_pct = info['size_ratio'] * 100

                    st.markdown(
                        f"**Pothole {i+1}** &nbsp; "
                        f"<span style='background-color:{color_hex}; color:white; padding:2px 10px; "
                        f"border-radius:4px; font-weight:bold;'>{severity}</span> &nbsp; "
                        f"Severity Score: **{score:.0%}** &nbsp; | &nbsp; "
                        f"Size: {width_px}×{height_px} px ({size_pct:.1f}% of image) &nbsp; | &nbsp; "
                        f"Depth Contrast: {info['depth_contrast']:.3f}",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No potholes detected in this submission.")


def video_demo_view(confidence):
    st.header("🎥 Real-time Video Demo")
    st.write("For testing pothole tracking on video streams.")
    input_source = st.radio("Select input source:", ['Sample data', 'Upload your own data'])
    video_input(input_source, confidence)


def main():
    st.title("🕳️ Pothole Detection & Severity Analysis System")

    # Initialize session state for navigation
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "current_view" not in st.session_state:
        st.session_state["current_view"] = "Public Reporter"

    st.sidebar.title("Navigation")
    view_options = ["Public Reporter", "PWD Official", "Video Demo"]
    selected_view = st.sidebar.radio("Go to:", view_options, index=view_options.index(st.session_state["current_view"]))
    st.session_state["current_view"] = selected_view

    st.sidebar.markdown("---")
    confidence = st.sidebar.slider('AI Confidence Threshold', min_value=0.1, max_value=1.0, value=0.3)

    st.sidebar.markdown("---")
    st.sidebar.info("The PWD Official login is: **admin** / **pwd123**")

    # Routing
    if selected_view == "Public Reporter":
        public_reporter_view()
    elif selected_view == "PWD Official":
        if st.session_state["logged_in"]:
            official_dashboard_view(confidence)
        else:
            official_login_view()
    elif selected_view == "Video Demo":
        video_demo_view(confidence)

if __name__ == "__main__":
    main()