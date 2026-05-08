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
from escalation_engine import run_escalation_check, get_overdue_summary, get_email_log, calculate_deadline
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

def reclassify_severity_from_score(score: float) -> str:
    """
    Derive the correct severity label from a numeric score using the
    same thresholds as DepthEstimator.classify_severity().

    Thresholds:
        score < 0.30  → Low
        score < 0.55  → Medium
        score >= 0.55 → High
    """
    if score < 0.30:
        return "Low"
    elif score < 0.55:
        return "Medium"
    else:
        return "High"


def load_submissions():
    """
    Load submissions from disk and auto-correct any stale severity labels.

    If a submission already has a stored ``severity_score`` but its
    ``severity`` label was saved under a different threshold regime, the
    label is recomputed from the score and the corrected value is written
    back to disk so the JSON stays authoritative.
    """
    with open(SUBMISSIONS_FILE, "r") as f:
        subs = json.load(f)

    dirty = False
    for sub in subs:
        score = sub.get("severity_score")
        if score is not None:                          # has been analysed
            correct_label = reclassify_severity_from_score(score)
            if sub.get("severity") != correct_label:  # label is stale
                sub["severity"] = correct_label
                dirty = True

    if dirty:                                          # write corrections back once
        with open(SUBMISSIONS_FILE, "w") as f:
            json.dump(subs, f, indent=4)

    return subs


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
    for pothole_num, info in enumerate(pothole_info, start=1):
        x1, y1, x2, y2 = map(int, info["bbox"])
        severity = info["severity"]
        score = info["score"]
        color = info["color"]

        # Label text — include pothole number so it matches the list below
        label = f"#{pothole_num} {severity} ({score:.0%})"

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
            output.image(processed_frame, caption="Pothole Detection Result", use_container_width=True)

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

    # ── AUTO ESCALATION CHECK ──────────────────────────────────────
    # Runs every time the dashboard loads — sends real emails for newly overdue complaints
    with st.spinner("🔍 Checking for overdue complaints and sending escalation emails..."):
        escalation_actions = run_escalation_check()

    # Flash banner for newly sent emails this session
    if escalation_actions:
        newly_sent = [a for a in escalation_actions if a["email_sent"]]
        newly_failed = [a for a in escalation_actions if not a["email_sent"]]
        if newly_sent:
            st.success(
                f"📧 **{len(newly_sent)} escalation email(s) just sent** to higher officials "
                f"for overdue complaints."
            )
        if newly_failed:
            st.error(
                f"❌ **{len(newly_failed)} email(s) failed to send.** "
                f"Check SMTP credentials in escalation_engine.py."
            )

    # ── ESCALATION EMAIL LOG ───────────────────────────────────────
    email_log = get_email_log()

    st.markdown("---")
    st.subheader("📧 Escalation Emails Sent to Higher Officials")

    if not email_log:
        st.info(
            "No escalation emails have been sent yet. "
            "Emails are dispatched automatically when a repair deadline is missed."
        )
    else:
        # Sort newest first
        sorted_log = sorted(email_log, key=lambda e: e.get("timestamp", ""), reverse=True)

        for entry in sorted_log:
            sent_ok      = entry.get("email_sent", False)
            severity     = entry.get("severity", "N/A")
            sev_color    = "#dc3545" if severity == "High" else "#fd7e14" if severity == "Medium" else "#28a745"
            days_over    = entry.get("days_overdue", 0)
            complaint_id = entry.get("complaint_id", "N/A")
            sent_time    = entry.get("timestamp", "N/A")
            to_auth      = entry.get("to_authority", "N/A")
            to_email     = entry.get("to_email", "N/A")
            deadline     = entry.get("deadline", "N/A")
            email_status = entry.get("email_status", "N/A")

            sub_match    = next((s for s in load_submissions() if s["id"] == complaint_id), {})
            submitted_on = sub_match.get("timestamp", "N/A")
            latitude     = sub_match.get("latitude")
            longitude    = sub_match.get("longitude")
            gps_str      = f"{latitude:.5f}, {longitude:.5f}" if latitude and longitude else "N/A"

            status_label = "✅ Email Sent" if sent_ok else "❌ Email Failed"

            with st.expander(
                f"{status_label}  ·  To: {to_auth} ({to_email})  ·  {sent_time}",
                expanded=False,
            ):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.caption("Complaint ID")
                    if sub_match and st.button(
                        f"🔍 {complaint_id}",
                        key=f"esclog_img_{entry.get('id', complaint_id)}",
                        help="Click to view the user-submitted image",
                        use_container_width=True,
                    ):
                        show_image_dialog(sub_match)
                c2.metric("Submitted On", submitted_on[:10] if submitted_on != "N/A" else "N/A")
                c3.metric("Repair Deadline", deadline[:10] if deadline and deadline != "N/A" else "N/A")
                c4.metric("Days Overdue", f"{days_over} days")

                st.markdown(
                    f"**Severity:** &nbsp;"
                    f"<span style='background:{sev_color}; color:white; padding:2px 12px; "
                    f"border-radius:4px; font-size:14px; font-weight:bold;'>{severity}</span>"
                    f"&nbsp;&nbsp;&nbsp; **GPS:** `{gps_str}`"
                    f"&nbsp;&nbsp;&nbsp; **Email Status:** {email_status}",
                    unsafe_allow_html=True,
                )
                if latitude and longitude:
                    st.markdown(
                        f"[📍 View on Google Maps](https://www.google.com/maps?q={latitude},{longitude})"
                    )

    st.markdown("---")


    submissions = load_submissions()
    if not submissions:
        st.info("No pothole submissions yet.")
        return

    # Display submissions
    st.subheader(f"Total Reports: {len(submissions)}")

    # ── SORT CONTROL ───────────────────────────────────────────────
    sort_col, _ = st.columns([2, 4])
    with sort_col:
        sort_option = st.selectbox(
            "Sort by:",
            options=["🕐 Recent First", "🔴 Severity: High → Low", "🟢 Severity: Low → High"],
            index=0,
            key="pwd_sort_option"
        )

    # Severity rank map for sorting (higher number = more severe)
    severity_rank = {"High": 3, "Medium": 2, "Low": 1, None: 0, "": 0}

    # Prepare data for tabulated view
    table_data = []
    for sub in reversed(submissions):   # reversed = recent first baseline
        lat = sub.get('latitude')
        lon = sub.get('longitude')
        place_name = get_place_name(lat, lon) if lat and lon else "Unknown Location"
        table_data.append({
            "Report ID": sub['id'],
            "Report Time": sub['timestamp'],
            "Location": place_name.split(',')[0],
            "Coordinates": f"{lat:.4f}, {lon:.4f}" if lat and lon else "N/A",
            "Status": sub.get('status', 'Pending Analysis'),
            "Severity": sub.get('severity', ''),          # stored after first analysis
            "_sub": sub,                                   # keep reference for detail loop
        })

    # Apply sort
    if sort_option == "🔴 Severity: High → Low":
        table_data.sort(key=lambda r: severity_rank.get(r["Severity"], 0), reverse=True)
    elif sort_option == "🟢 Severity: Low → High":
        table_data.sort(key=lambda r: severity_rank.get(r["Severity"], 0), reverse=False)
    # else: keep "Recent First" (already reversed)

    # Build ordered submissions list to match the chosen sort
    sorted_submissions = [r["_sub"] for r in table_data]

    # Render clickable table
    selected_report_id = st.session_state.get("selected_report_id", "All Reports")

    if table_data:
        # Severity badge helper
        def severity_badge(sev):
            if sev == "High":
                return "🔴 High"
            elif sev == "Medium":
                return "🟠 Medium"
            elif sev == "Low":
                return "🟢 Low"
            return "➖ N/A"

        # Table header row
        header = st.columns([2, 2, 2, 1, 2, 2])
        header[0].markdown("**Report ID**")
        header[1].markdown("**Date/Time**")
        header[2].markdown("**Area**")
        header[3].markdown("**Severity**")
        header[4].markdown("**GPS Data**")
        header[5].markdown("**Status**")
        st.markdown("<hr style='margin:0; padding:0'>", unsafe_allow_html=True)

        # Table data rows — each row is a clickable button
        for d in table_data:
            cols = st.columns([2, 2, 2, 1, 2, 2])
            with cols[0]:
                if st.button(f"🔍 {d['Report ID']}", key=f"row_{d['Report ID']}", use_container_width=True):
                    # Open full report in a modal dialog — no scrolling
                    show_pwd_report_dialog(d["_sub"], confidence)
            cols[1].write(d["Report Time"])
            cols[2].write(d["Location"])
            cols[3].write(severity_badge(d["Severity"]))
            cols[4].write(d["Coordinates"])
            cols[5].write(d["Status"])

    st.markdown("---")



@st.dialog("📋 Report Details & AI Analysis", width="large")
def show_pwd_report_dialog(sub, confidence):
    """Modal popup showing the full PWD report: images, AI analysis, severity, map."""
    import math

    sub_id  = sub['id']
    lat     = sub.get('latitude')
    lon     = sub.get('longitude')
    place_name = get_place_name(lat, lon) if lat and lon else "Unknown Location"
    current_status = sub.get('status', 'Pending Analysis')
    valid_statuses = ["Pending Analysis", "In Progress", "Resolved"]
    if current_status not in valid_statuses:
        current_status = "Pending Analysis"

    # ── Header info ────────────────────────────────────────────────
    st.markdown(f"**🆔 Report ID:** `{sub_id}`  &nbsp; | &nbsp; **📅 Submitted:** {sub.get('timestamp','N/A')}")
    st.markdown(f"**📍 Location:** {place_name}")
    if lat and lon:
        st.markdown(f"**📡 GPS:** {lat:.5f}, {lon:.5f} — [View on Map](https://www.google.com/maps?q={lat},{lon})")

    # ── Status updater ─────────────────────────────────────────────
    new_status = st.selectbox(
        "Update Status",
        options=valid_statuses,
        index=valid_statuses.index(current_status),
        key=f"dlg_status_{sub_id}"
    )
    if new_status != current_status and new_status != "Resolved":
        update_submission_status(sub_id, new_status)
        st.success(f"Status updated to **{new_status}**. Refresh to see changes.")

    # ── Repair verification — shown IMMEDIATELY when Resolved is selected ──
    if new_status == "Resolved" and current_status != "Resolved":
        st.markdown("---")
        st.subheader("🔧 Repair Verification — 3 Steps Required")
        st.info(
            "To mark this complaint as **Resolved**, you must:\n"
            "1. Share your **current GPS location** (must be within 500 m of the original pothole)\n"
            "2. Upload a **photo of the repaired road**\n"
            "3. Click **Verify & Mark as Resolved** — AI will confirm no potholes remain"
        )

        orig_lat = sub.get('latitude')
        orig_lon = sub.get('longitude')

        # Step 1 — Location
        st.markdown("#### 📍 Step 1: Share Your Current Location")
        repair_location = streamlit_geolocation()

        repair_lat, repair_lon = None, None
        location_ok = False
        repair_pil = None

        if repair_location and repair_location.get('latitude') and repair_location.get('longitude'):
            repair_lat = repair_location['latitude']
            repair_lon = repair_location['longitude']
            if orig_lat and orig_lon:
                def haversine(lat1, lon1, lat2, lon2):
                    R = 6371000
                    p1, p2 = math.radians(lat1), math.radians(lat2)
                    dp = math.radians(lat2 - lat1)
                    dl = math.radians(lon2 - lon1)
                    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
                    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))
                distance = haversine(orig_lat, orig_lon, repair_lat, repair_lon)
                if distance <= 500:
                    st.success(f"✅ Location confirmed — you are **{distance:.0f} m** from the reported pothole.")
                    location_ok = True
                else:
                    st.error(f"❌ You are **{distance:.0f} m** away. Must be within **500 m** to verify the repair.")
            else:
                st.warning("⚠️ Original complaint has no GPS. Location check skipped.")
                location_ok = True
        else:
            st.caption("⏳ Waiting for location… Allow browser location access if prompted.")

        # Step 2 — Upload image
        st.markdown("#### 📸 Step 2: Upload Repaired Road Photo")
        repair_image = st.file_uploader(
            "Upload a clear photo of the repaired road",
            type=['png', 'jpeg', 'jpg'],
            key=f"dlg_repair_{sub_id}"
        )
        if repair_image:
            repair_pil = Image.open(repair_image)
            if repair_pil.mode != 'RGB':
                repair_pil = repair_pil.convert('RGB')
            st.image(repair_pil, caption="📷 Repair photo preview", use_container_width=True)

        # Step 3 — Verify button
        st.markdown("#### ✅ Step 3: Verify & Submit")
        if st.button("✅ Verify & Mark as Resolved", key=f"dlg_verify_{sub_id}", type="primary"):
            errors = []
            if not repair_location or not repair_location.get('latitude'):
                errors.append("❌ Location not detected — please allow location access and wait for it to load.")
            elif not location_ok:
                errors.append("❌ Your location is too far from the original pothole. Move closer and try again.")
            if not repair_image:
                errors.append("❌ Please upload a photo of the repaired road.")
            if errors:
                for e in errors:
                    st.error(e)
            else:
                repair_np  = np.array(repair_pil)
                repair_bgr = cv2.cvtColor(repair_np, cv2.COLOR_RGB2BGR)
                with st.spinner("🤖 AI is checking the repaired road for remaining potholes..."):
                    repair_results = model(repair_bgr, conf=confidence)[0]
                    num_potholes   = len(repair_results.boxes)
                if num_potholes > 0:
                    st.error(
                        f"❌ Verification failed — AI still detects **{num_potholes} pothole(s)**. "
                        "Please ensure the road is fully repaired and upload a clear photo."
                    )
                    st.image(repair_np, caption="Repair Photo (Potholes Still Detected)", use_container_width=True)
                else:
                    repair_pil.save(os.path.join(SUBMISSIONS_DIR, f"repair_{sub_id}.jpg"))
                    update_submission_status(sub_id, "Resolved")
                    st.success("✅ All checks passed! Location confirmed, no potholes detected. Complaint marked as **Resolved**.")
                    st.balloons()
        return   # Don't show images/map until verification is done

    st.markdown("---")

    # ── Check image exists ─────────────────────────────────────────
    if not os.path.exists(sub.get('filepath', '')):
        st.error("Image file not found.")
        return

    image = cv2.imread(sub['filepath'])
    if image is None:
        st.error("Could not read image file.")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="📸 Original User Submission", use_container_width=True)

    # Show repair image for resolved reports
    if current_status == "Resolved":
        repair_path = os.path.join(SUBMISSIONS_DIR, f"repair_{sub_id}.jpg")
        if os.path.exists(repair_path):
            with col2:
                st.image(Image.open(repair_path), caption="🟢 After Repair (Verified)", use_container_width=True)
        st.success("✅ This pothole has been repaired and verified.")
        return

    # ── AI Analysis ────────────────────────────────────────────────
    with col2:
        with st.spinner("🤖 Running AI analysis..."):
            processed_image, pothole_info = callback(image_rgb, confidence)
        st.image(processed_image, caption="🔍 AI Detection + Severity Result", use_container_width=True)

    # ── Pothole details ────────────────────────────────────────────
    if pothole_info:
        st.markdown("---")
        st.subheader("🕳️ Analysis Results")

        max_severity_score = max(info["score"] for info in pothole_info)
        if max_severity_score < 0.30:
            overall_severity, marker_color = "Low",    [40, 167, 69]
        elif max_severity_score < 0.55:
            overall_severity, marker_color = "Medium", [253, 126, 20]
        else:
            overall_severity, marker_color = "High",   [220, 53, 69]

        if not sub.get("severity"):
            subs_all = load_submissions()
            for s in subs_all:
                if s["id"] == sub_id:
                    s["severity"]       = overall_severity
                    s["severity_score"] = max_severity_score
                    s["deadline"]       = calculate_deadline(sub["timestamp"], overall_severity)
                    s["analyzed_at"]    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    break
            with open(SUBMISSIONS_FILE, "w") as f:
                json.dump(subs_all, f, indent=4)

        for i, info in enumerate(pothole_info):
            sev       = info["severity"]
            score     = info["score"]
            color_hex = "#28a745" if sev == "Low" else "#fd7e14" if sev == "Medium" else "#dc3545"
            x1, y1, x2, y2 = map(int, info["bbox"])
            w_px  = x2 - x1
            h_px  = y2 - y1
            s_pct = info['size_ratio'] * 100
            st.markdown(
                f"**Pothole {i+1}** &nbsp; "
                f"<span style='background:{color_hex}; color:white; padding:2px 10px; "
                f"border-radius:4px; font-weight:bold;'>{sev}</span> &nbsp; "
                f"Score: **{score:.0%}** &nbsp;|&nbsp; "
                f"Size: {w_px}×{h_px} px ({s_pct:.1f}%) &nbsp;|&nbsp; "
                f"Depth Contrast: {info['depth_contrast']:.3f}",
                unsafe_allow_html=True,
            )

    else:
        h, w = image_rgb.shape[:2]
        if w < 300 or h < 300:
            st.warning("No potholes detected — image resolution is too low for AI analysis.")
        else:
            st.info("**No potholes detected** in this submission.")

    # ── Map always shown when GPS is available ──────────────────────
    if lat and lon:
        # Use live severity color if available, otherwise fall back to stored severity
        if not pothole_info:
            stored_sev = sub.get("severity", "Low")
            overall_severity = stored_sev
            marker_color = (
                [220, 53, 69]  if stored_sev == "High"   else
                [253, 126, 20] if stored_sev == "Medium" else
                [40, 167, 69]
            )
        st.markdown("---")
        st.subheader(f"🗺️ Map Location (Overall Severity: {overall_severity})")
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=17, pitch=45),
            layers=[pdk.Layer(
                'ScatterplotLayer',
                data=[{"lat": lat, "lon": lon, "color": marker_color}],
                get_position='[lon, lat]',
                get_fill_color='color',
                get_radius=50,
                radius_min_pixels=15,
                radius_max_pixels=35,
                pickable=True,
            )],
        ))






@st.dialog("📸 Submitted Image", width="large")
def show_image_dialog(sub):
    """Modal dialog showing only the user-submitted image for a complaint."""
    filepath = sub.get("filepath", "")
    lat = sub.get("latitude")
    lon = sub.get("longitude")
    place_name = get_place_name(lat, lon) if lat and lon else "Unknown Location"
    status = sub.get("status", "Pending Analysis")

    st.markdown(f"**🆔 Report ID:** `{sub['id']}`")
    st.markdown(f"**📍 Location:** {place_name.split(',')[0]}")
    st.markdown(f"**🕐 Filed On:** {sub.get('timestamp', 'N/A')}")
    st.markdown(f"**📡 GPS:** {f'{lat:.5f}, {lon:.5f}' if lat and lon else 'N/A'}")

    if status == "Resolved":
        st.success("✅ Status: Resolved")
    elif status == "In Progress":
        st.warning("🔧 Status: In Progress")
    else:
        st.info("⏳ Status: Pending Analysis")

    st.markdown("---")
    if filepath and os.path.exists(filepath):
        orig_img = Image.open(filepath)
        st.image(orig_img, caption="📸 User Submitted Image", use_container_width=True)
    else:
        st.warning("⚠️ Original image file not found.")


@st.dialog("🚨 Escalated Report Details", width="large")
def show_escalated_report_dialog(sub, days_overdue):
    """Modal showing submitted image, date, location and overdue info for an escalated complaint."""
    sub_id   = sub.get("id", "N/A")
    lat      = sub.get("latitude")
    lon      = sub.get("longitude")
    filepath = sub.get("filepath", "")
    place_name = get_place_name(lat, lon) if lat and lon else "Unknown Location"
    severity = sub.get("severity", "N/A")
    sev_color = "#dc3545" if severity == "High" else "#fd7e14" if severity == "Medium" else "#28a745"
    deadline  = sub.get("deadline", "N/A")

    # Header info
    st.markdown(f"**🆔 Report ID:** `{sub_id}`")
    st.markdown(f"**📍 Location:** {place_name}")
    if lat and lon:
        st.markdown(f"**📡 GPS:** `{lat:.5f}, {lon:.5f}` — [View on Map](https://www.google.com/maps?q={lat},{lon})")
    st.markdown(f"**🕐 Reported On:** {sub.get('timestamp', 'N/A')}")

    # Key metrics
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f"**Severity**\n\n"
        f"<span style='background:{sev_color}; color:white; padding:3px 14px; "
        f"border-radius:4px; font-size:15px; font-weight:bold;'>{severity}</span>",
        unsafe_allow_html=True,
    )
    c2.metric("Repair Deadline", deadline[:10] if deadline and deadline != "N/A" else "N/A")
    c3.metric("Days Overdue", f"{days_overdue} days", delta=f"+{days_overdue}", delta_color="inverse")

    st.error(f"⚠️ This complaint is **{days_overdue} day(s) overdue** and has been escalated to higher officials.")
    st.markdown("---")

    # Image
    if filepath and os.path.exists(filepath):
        st.image(Image.open(filepath), caption="📸 User Submitted Image", use_container_width=True)
    else:
        st.warning("⚠️ Original image file not found.")


def complaint_status_view():
    st.header("📊 Complaint Status Tracker")
    st.write("Track the status of all reported potholes in your area.")

    submissions = load_submissions()
    if not submissions:
        st.info("No complaints have been filed yet.")
        return

    # ── ESCALATED REPORTS SECTION ──────────────────────────────────
    email_log = get_email_log()
    if email_log:
        # Build a unique set: complaint_id → max days_overdue from the log
        escalated_map = {}
        for entry in email_log:
            cid  = entry.get("complaint_id")
            days = entry.get("days_overdue", 0)
            if cid and (cid not in escalated_map or days > escalated_map[cid]):
                escalated_map[cid] = days

        if escalated_map:
            sub_lookup = {s["id"]: s for s in submissions}
            st.markdown("---")
            st.subheader("🚨 Reports Escalated to Higher Officials")
            st.caption(
                "The following complaints missed their repair deadline and have been "
                "escalated to senior authorities. Click a Report ID for details."
            )

            # Column header
            h = st.columns([2, 2, 2, 1, 1])
            h[0].markdown("**Report ID**")
            h[1].markdown("**Reported On**")
            h[2].markdown("**Area**")
            h[3].markdown("**Severity**")
            h[4].markdown("**Days Overdue**")
            st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)

            for cid, days_over in sorted(escalated_map.items(), key=lambda x: x[1], reverse=True):
                sub = sub_lookup.get(cid)
                if not sub:
                    continue
                lat = sub.get("latitude")
                lon = sub.get("longitude")
                area = get_place_name(lat, lon).split(",")[0] if lat and lon else "Unknown"
                severity = sub.get("severity", "N/A")
                sev_icon = "🔴" if severity == "High" else "🟠" if severity == "Medium" else "🟢"

                cols = st.columns([2, 2, 2, 1, 1])
                with cols[0]:
                    if st.button(
                        f"🔍 {cid}",
                        key=f"esc_{cid}",
                        use_container_width=True,
                        help="Click to view submitted image and details",
                    ):
                        show_escalated_report_dialog(sub, days_over)
                cols[1].write(sub.get("timestamp", "N/A"))
                cols[2].write(area)
                cols[3].write(f"{sev_icon} {severity}")
                cols[4].write(f"**{days_over}d**")

            st.markdown("---")

    # Prepare table data
    status_data = []
    for sub in reversed(submissions):
        lat = sub.get('latitude')
        lon = sub.get('longitude')
        place_name = get_place_name(lat, lon) if lat and lon else "Unknown Location"
        status = sub.get('status', 'Pending Analysis')

        if status == "Resolved":
            status_icon = "✅ Resolved"
        elif status == "In Progress":
            status_icon = "🔧 In Progress"
        else:
            status_icon = "⏳ Pending Analysis"

        status_data.append({
            "id":       sub['id'],
            "Date":     sub['timestamp'],
            "Location": place_name.split(',')[0],
            "Status":   status_icon,
            "_sub":     sub,
        })

    # ── Summary metrics ────────────────────────────────────────────
    total       = len(status_data)
    pending     = sum(1 for d in status_data if "Pending"     in d["Status"])
    in_progress = sum(1 for d in status_data if "In Progress" in d["Status"])
    resolved    = sum(1 for d in status_data if "Resolved"    in d["Status"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Complaints", total)
    col2.metric("⏳ Pending",      pending)
    col3.metric("🔧 In Progress",   in_progress)
    col4.metric("✅ Resolved",      resolved)

    st.markdown("---")

    # ── Filter ─────────────────────────────────────────────────────
    filter_option = st.selectbox(
        "Filter by status:",
        ["All", "⏳ Pending Analysis", "🔧 In Progress", "✅ Resolved"],
        key="status_filter",
    )
    filtered_data = [
        d for d in status_data
        if filter_option == "All" or d["Status"] == filter_option
    ]

    # ── Single unified clickable table ─────────────────────────────
    hdr = st.columns([3, 2, 2, 2])
    hdr[0].markdown("**Complaint ID**")
    hdr[1].markdown("**Filed On**")
    hdr[2].markdown("**Area**")
    hdr[3].markdown("**Status**")
    st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)

    sub_map = {sub["id"]: sub for sub in submissions}
    for row in filtered_data:
        cols = st.columns([3, 2, 2, 2])
        with cols[0]:
            if st.button(f"🔍 {row['id']}", key=f"cs_{row['id']}", use_container_width=True, help="Click to view submitted image"):
                show_image_dialog(sub_map[row["id"]])
        cols[1].write(row["Date"])
        cols[2].write(row["Location"])
        cols[3].write(row["Status"])


    # ── RESOLVED BEFORE/AFTER SECTION ─────────────────────────────
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

    confidence = 0.3

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
