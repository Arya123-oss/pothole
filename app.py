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
import streamlit.components.v1 as components
import pydeck as pdk
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd

# Initialize geolocator
geolocator = Nominatim(user_agent="pothole_detector_app")
reverse_geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)

@st.cache_data
def get_place_name(lat, lon):
    try:
        location = reverse_geocode((lat, lon), language='en')
        return location.address if location else "Unknown Location"
    except Exception as e:
        return "Unknown Location"


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

def update_submission_status(report_id, new_status):
    subs = load_submissions()
    for s in subs:
        if s["id"] == report_id:
            s["status"] = new_status
            break
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
    
    # NOTE: ByteTrack is only really useful for continuous video.
    # On a dashboard displaying independent, static images, the tracker 
    # will discard detections because they don't logically "track" from the previous image.
    # If this callback is used for a single image, we skip tracking.
    
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

        # Read standard image buffer using PIL to preserve original image properties
        image = Image.open(img_bytes)
        
        # Check image resolution and warn if too small
        width, height = image.size
        if width < 300 or height < 300:
            st.warning("⚠️ Warning: Based on your upload, the image resolution is very small. The AI model may not be able to effectively detect potholes in low-quality or thumbnail-sized images.")
            
        # Ensure it's in standard RGB format before saving to avoid RGBA/P issues
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save to file
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

    # Prepare data for tabulated view
    table_data = []
    for sub in reversed(submissions):
        lat = sub.get('latitude')
        lon = sub.get('longitude')
        place_name = get_place_name(lat, lon) if lat and lon else "Unknown Location"
        
        # We will use the timestamp as a unique ID to link the row
        table_data.append({
            "Report ID": sub['id'],
            "Report Time": sub['timestamp'],
            "Location": place_name.split(',')[0],  # Get just the prominent place name
            "Coordinates": f"{lat:.4f}, {lon:.4f}" if lat and lon else "N/A",
            "Status": sub.get('status', 'Pending Analysis')
        })
        
    # Render clickable table
    selected_report_id = st.session_state.get("selected_report_id", "All Reports")
    
    if table_data:
        # Table header row
        header = st.columns([2, 2, 2, 2, 2])
        header[0].markdown("**Report ID**")
        header[1].markdown("**Date/Time**")
        header[2].markdown("**Area**")
        header[3].markdown("**GPS Data**")
        header[4].markdown("**Status**")
        st.markdown("<hr style='margin:0; padding:0'>", unsafe_allow_html=True)
        
        # Table data rows — each row is a clickable button
        for d in table_data:
            cols = st.columns([2, 2, 2, 2, 2])
            with cols[0]:
                if st.button(f"🔍 {d['Report ID']}", key=f"row_{d['Report ID']}", use_container_width=True):
                    st.session_state["selected_report_id"] = d["Report ID"]
                    st.rerun()
            cols[1].write(d["Report Time"])
            cols[2].write(d["Location"])
            cols[3].write(d["Coordinates"])
            cols[4].write(d["Status"])
        
        # Show All button when filtering
        if selected_report_id != "All Reports":
            st.markdown("---")
            st.success(f"Currently viewing Report ID: **{selected_report_id}**")
            if st.button("🔄 Show All Reports"):
                st.session_state["selected_report_id"] = "All Reports"
                st.rerun()

    st.markdown("---")
    st.subheader("Detailed Reports & Artificial Intelligence Analysis")
    
    # Auto-scroll to selected report using JavaScript
    if selected_report_id != "All Reports":
        components.html(
            f"""
            <script>
                const el = window.parent.document.getElementById('report-{selected_report_id}');
                if (el) {{ el.scrollIntoView({{ behavior: 'smooth', block: 'start' }}); }}
            </script>
            """,
            height=0
        )

    for idx, sub in enumerate(reversed(submissions)):
        if selected_report_id != "All Reports" and sub['id'] != selected_report_id:
            continue
            
        lat = sub.get('latitude')
        lon = sub.get('longitude')
        
        # Determine the place name if coordinates exist
        if lat and lon:
            place_name = get_place_name(lat, lon)
            expander_title = f"[ID: {sub['id']}] Report from {sub['timestamp']} - {place_name.split(',')[0]} (Status: {sub.get('status', 'Pending')})"
        else:
            place_name = "Unknown Location"
            expander_title = f"[ID: {sub['id']}] Report from {sub['timestamp']} - Unknown Location (Status: {sub.get('status', 'Pending')})"

        # Anchor for auto-scroll
        st.markdown(f"<div id='report-{sub['id']}'></div>", unsafe_allow_html=True)
        
        with st.expander(expander_title, expanded=(selected_report_id != "All Reports")):
            
            col_info, col_status = st.columns([3, 1])
            with col_info:
                st.write(f"**Place Name:** {place_name}")
                st.write(f"**GPS Coordinates:** {lat if lat else 'Unknown'}, {lon if lon else 'Unknown'}")
            
            with col_status:
                current_status = sub.get('status', 'Pending Analysis')
                # Normalize legacy status values
                valid_statuses = ["Pending Analysis", "In Progress", "Resolved"]
                if current_status not in valid_statuses:
                    current_status = "Pending Analysis"
                    
                new_status = st.selectbox(
                    "Update Status",
                    options=valid_statuses,
                    index=valid_statuses.index(current_status),
                    key=f"status_{sub['id']}"
                )
                
                # Allow immediate change for Pending <-> In Progress
                if new_status != current_status and new_status != "Resolved":
                    update_submission_status(sub['id'], new_status)
                    st.rerun()
            
            # If official wants to mark as Resolved, require repair verification
            if new_status == "Resolved" and current_status != "Resolved":
                st.markdown("---")
                st.subheader("🔧 Repair Verification Required")
                st.write("To mark this complaint as **Resolved**, you must upload a photo of the repaired road from the **same location**.")
                
                repair_col1, repair_col2 = st.columns(2)
                
                with repair_col1:
                    st.write("**Step 1: Get your current location**")
                    repair_location = streamlit_geolocation()
                
                with repair_col2:
                    st.write("**Step 2: Upload repaired road image**")
                    repair_image = st.file_uploader("Upload repair photo", type=['png', 'jpeg', 'jpg'], key=f"repair_{sub['id']}")
                
                if st.button("✅ Verify & Mark as Resolved", key=f"verify_{sub['id']}", type="primary"):
                    # Check 1: Location verification
                    if not repair_location or not repair_location.get('latitude') or not repair_location.get('longitude'):
                        st.error("❌ Location not detected. Please allow location access and try again.")
                    elif not repair_image:
                        st.error("❌ Please upload a photo of the repaired road.")
                    else:
                        repair_lat = repair_location['latitude']
                        repair_lon = repair_location['longitude']
                        orig_lat = sub.get('latitude')
                        orig_lon = sub.get('longitude')
                        
                        # Calculate distance between coordinates (simple Haversine approximation)
                        import math
                        def haversine(lat1, lon1, lat2, lon2):
                            R = 6371000  # Earth radius in meters
                            phi1, phi2 = math.radians(lat1), math.radians(lat2)
                            dphi = math.radians(lat2 - lat1)
                            dlambda = math.radians(lon2 - lon1)
                            a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
                            return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))
                        
                        if orig_lat and orig_lon:
                            distance = haversine(orig_lat, orig_lon, repair_lat, repair_lon)
                            st.info(f"📍 Distance from original report location: **{distance:.0f} meters**")
                            
                            if distance > 500:
                                st.error(f"❌ Location mismatch! You are **{distance:.0f}m** away from the original pothole location. You must be within 500 meters to verify the repair.")
                            else:
                                # Load repair image
                                repair_pil = Image.open(repair_image)
                                if repair_pil.mode != 'RGB':
                                    repair_pil = repair_pil.convert('RGB')
                                repair_np = np.array(repair_pil)
                                repair_bgr = cv2.cvtColor(repair_np, cv2.COLOR_RGB2BGR)
                                
                                # Check 2: Image similarity — verify repair image matches the same scene
                                with st.spinner("🔍 Comparing original and repair images..."):
                                    orig_img = cv2.imread(sub['filepath'])
                                    if orig_img is not None:
                                        # Resize both to same dimensions for fair comparison
                                        compare_size = (256, 256)
                                        orig_resized = cv2.resize(orig_img, compare_size)
                                        repair_resized = cv2.resize(repair_bgr, compare_size)
                                        
                                        # Convert to HSV for better color-based comparison
                                        orig_hsv = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2HSV)
                                        repair_hsv = cv2.cvtColor(repair_resized, cv2.COLOR_BGR2HSV)
                                        
                                        # Calculate histograms
                                        hist_orig = cv2.calcHist([orig_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                                        hist_repair = cv2.calcHist([repair_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                                        
                                        cv2.normalize(hist_orig, hist_orig)
                                        cv2.normalize(hist_repair, hist_repair)
                                        
                                        similarity = cv2.compareHist(hist_orig, hist_repair, cv2.HISTCMP_CORREL)
                                        st.info(f"🖼️ Image scene similarity: **{similarity:.2%}**")
                                
                                if orig_img is not None and similarity < 0.3:
                                    st.error("❌ The uploaded repair image does not appear to match the original complaint location. The scenes look completely different. Please upload a photo from the same road/area.")
                                    comp_col1, comp_col2 = st.columns(2)
                                    with comp_col1:
                                        st.image(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB), caption="Original Complaint Image")
                                    with comp_col2:
                                        st.image(repair_np, caption="Your Uploaded Repair Image")
                                else:
                                    # Check 3: YOLO verification — no potholes should be detected
                                    with st.spinner("🔍 AI is checking the repaired road for remaining potholes..."):
                                        repair_results = model(repair_bgr, conf=confidence)[0]
                                        num_potholes = len(repair_results.boxes)
                                    
                                    if num_potholes > 0:
                                        st.error(f"❌ Verification failed! The AI still detects **{num_potholes} pothole(s)** in the uploaded repair image. The road does not appear to be fully repaired.")
                                        st.image(repair_np, caption="Uploaded Repair Image (Potholes Still Detected)")
                                    else:
                                        st.success("✅ Verification passed! Images match, no potholes detected, and location confirmed. Marking as Resolved.")
                                        st.image(repair_np, caption="Verified Repair Image — No Potholes Detected")
                                        
                                        # Save repair image
                                        repair_filename = f"repair_{sub['id']}.jpg"
                                        repair_filepath = os.path.join(SUBMISSIONS_DIR, repair_filename)
                                        repair_pil.save(repair_filepath)
                                        
                                        update_submission_status(sub['id'], "Resolved")
                                        st.balloons()
                                        st.rerun()
                        else:
                            st.warning("⚠️ Original complaint has no GPS coordinates. Skipping location check.")
                            # Still check for potholes
                            with st.spinner("🔍 AI is checking the repaired road..."):
                                repair_pil = Image.open(repair_image)
                                if repair_pil.mode != 'RGB':
                                    repair_pil = repair_pil.convert('RGB')
                                repair_np = np.array(repair_pil)
                                repair_bgr = cv2.cvtColor(repair_np, cv2.COLOR_RGB2BGR)
                                
                                repair_results = model(repair_bgr, conf=confidence)[0]
                                num_potholes = len(repair_results.boxes)
                            
                            if num_potholes > 0:
                                st.error(f"❌ Verification failed! AI still detects **{num_potholes} pothole(s)**.")
                            else:
                                st.success("✅ No potholes detected. Marking as Resolved.")
                                update_submission_status(sub['id'], "Resolved")
                                st.balloons()
                                st.rerun()

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
            
            # Show repair image for resolved reports
            if current_status == "Resolved":
                repair_path = os.path.join(SUBMISSIONS_DIR, f"repair_{sub['id']}.jpg")
                if os.path.exists(repair_path):
                    with col2:
                        repair_img = Image.open(repair_path)
                        st.image(repair_img, caption="🟢 After Repair (Verified)")
                    st.success("✅ This pothole has been repaired and verified.")

            # Run Analysis
            with col2:
                with st.spinner("Analyzing image..."):
                    # The callback expects an image in RGB format, so passing image_rgb is correct.
                    # HOWEVER! The callback currently has: bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    processed_image, pothole_info = callback(image_rgb, confidence)
                    st.image(processed_image, caption="AI Detection + Severity Result")

            # Display stats and map
            if pothole_info:
                st.markdown("---")
                st.subheader("🕳️ Analysis Results")
                
                # Determine highest severity for map marker color
                max_severity_score = max(info["score"] for info in pothole_info)
                if max_severity_score < 0.30:
                    overall_severity, marker_color = "Low", [40, 167, 69]       # Green
                elif max_severity_score < 0.55:
                    overall_severity, marker_color = "Medium", [253, 126, 20]   # Orange
                else:
                    overall_severity, marker_color = "High", [220, 53, 69]      # Red

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
                
                # Render the Map with Color coding
                if lat and lon:
                    st.markdown("---")
                    st.subheader(f"🗺️ Map Location (Overall Severity: {overall_severity})")
                    map_data = [{"lat": lat, "lon": lon, "color": marker_color}]
                    
                    st.pydeck_chart(pdk.Deck(
                        map_style=None,
                        initial_view_state=pdk.ViewState(
                            latitude=lat,
                            longitude=lon,
                            zoom=17,
                            pitch=45,
                        ),
                        layers=[
                            pdk.Layer(
                                'ScatterplotLayer',
                                data=map_data,
                                get_position='[lon, lat]',
                                get_fill_color='color',
                                get_radius=50,
                                radius_min_pixels=15,
                                radius_max_pixels=35,
                                pickable=True,
                            ),
                        ],
                    ))
            else:
                height, width = image_rgb.shape[:2]
                if width < 300 or height < 300:
                    st.warning("No potholes detected. ⚠️ Note: This image is extremely small or low resolution. The AI cannot reliably detect features on thumbnail-sized or highly pixelated images.")
                else:
                    st.info("""
                    **No potholes detected in this submission.** 
                    
                    *Possible reasons for this result:*
                    - The AI confidence threshold is set too high (currently {:.0%}). Try adjusting the slider on the left.
                    - The image quality is poor, blurry, or completely lacks depth contrast.
                    - The damage on the road is just a dry patch/discoloration and not an actual deep pothole.
                    - The pothole is obscured by shadows, water, or debris.
                    """.format(confidence))


def complaint_status_view():
    st.header("📊 Complaint Status Tracker")
    st.write("Track the status of all reported potholes in your area.")

    submissions = load_submissions()
    if not submissions:
        st.info("No complaints have been filed yet.")
        return

    # Prepare table data
    status_data = []
    for sub in reversed(submissions):
        lat = sub.get('latitude')
        lon = sub.get('longitude')
        place_name = get_place_name(lat, lon) if lat and lon else "Unknown Location"
        status = sub.get('status', 'Pending Analysis')

        # Assign emoji based on status
        if status == "Resolved":
            status_icon = "✅ Resolved"
        elif status == "In Progress":
            status_icon = "🔧 In Progress"
        else:
            status_icon = "⏳ Pending Analysis"

        status_data.append({
            "Report ID": sub['id'],
            "Date": sub['timestamp'],
            "Location": place_name.split(',')[0],
            "Status": status_icon
        })

    df = pd.DataFrame(status_data)

    # Summary counts
    total = len(status_data)
    pending = sum(1 for d in status_data if "Pending" in d["Status"])
    in_progress = sum(1 for d in status_data if "In Progress" in d["Status"])
    resolved = sum(1 for d in status_data if "Resolved" in d["Status"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Complaints", total)
    col2.metric("⏳ Pending", pending)
    col3.metric("🔧 In Progress", in_progress)
    col4.metric("✅ Resolved", resolved)

    st.markdown("---")

    # Filter
    filter_option = st.selectbox("Filter by status:", ["All", "⏳ Pending Analysis", "🔧 In Progress", "✅ Resolved"])
    if filter_option != "All":
        df = df[df["Status"] == filter_option]

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Report ID": "Complaint ID",
            "Date": st.column_config.DatetimeColumn("Filed On", format="YYYY-MM-DD HH:mm"),
            "Location": "Area",
            "Status": "Current Status"
        }
    )
    
    # Show before/after images for resolved complaints
    resolved_subs = [sub for sub in reversed(submissions) if sub.get('status') == 'Resolved']
    if resolved_subs:
        st.markdown("---")
        st.subheader("✅ Resolved Complaints — Before & After")
        
        for sub in resolved_subs:
            repair_path = os.path.join(SUBMISSIONS_DIR, f"repair_{sub['id']}.jpg")
            original_path = sub.get('filepath', '')
            
            lat = sub.get('latitude')
            lon = sub.get('longitude')
            place_name = get_place_name(lat, lon) if lat and lon else "Unknown Location"
            
            with st.expander(f"✅ {sub['id']} — {place_name.split(',')[0]}", expanded=True):
                if os.path.exists(original_path) and os.path.exists(repair_path):
                    before_col, after_col = st.columns(2)
                    with before_col:
                        orig_img = Image.open(original_path)
                        st.image(orig_img, caption="🔴 Before (Pothole Reported)")
                    with after_col:
                        repair_img = Image.open(repair_path)
                        st.image(repair_img, caption="🟢 After (Repaired & Verified)")
                elif os.path.exists(original_path):
                    st.image(Image.open(original_path), caption="Original Report (Repair image not available)")
                else:
                    st.info("Images not available for this complaint.")


def main():
    st.title("🕳️ Pothole Detection & Severity Analysis System")

    # Initialize session state for navigation
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "current_view" not in st.session_state:
        st.session_state["current_view"] = "Public Reporter"

    st.sidebar.title("Navigation")
    view_options = ["Public Reporter", "PWD Official", "Complaint Status"]
    selected_view = st.sidebar.radio("Go to:", view_options, index=view_options.index(st.session_state["current_view"]))
    st.session_state["current_view"] = selected_view

    st.sidebar.markdown("---")
    confidence = st.sidebar.slider('AI Confidence Threshold', min_value=0.1, max_value=1.0, value=0.3)

    st.sidebar.markdown("---")

    # Routing
    if selected_view == "Public Reporter":
        public_reporter_view()
    elif selected_view == "PWD Official":
        if st.session_state["logged_in"]:
            official_dashboard_view(confidence)
        else:
            official_login_view()
    elif selected_view == "Complaint Status":
        complaint_status_view()

if __name__ == "__main__":
    main()