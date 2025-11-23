# app_streamlit_webrtc_register_fixed.py
import os
import io
import re
from datetime import datetime
import threading
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# streamlit-webrtc imports
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
except Exception:
    webrtc_streamer = None
    VideoProcessorBase = object
    RTCConfiguration = None
    WebRtcMode = None

# av is required by VideoProcessorBase output
try:
    import av
except Exception:
    av = None

# Optional libs
try:
    import cv2
except Exception:
    cv2 = None

try:
    import dlib
except Exception:
    dlib = None

# Folium / map
try:
    from streamlit_folium import st_folium
    import folium
except Exception:
    st_folium = None
    folium = None

import requests

# ---------------- Config ----------------
BASE_DIR = os.getcwd()
CSV_FILE = os.path.join(BASE_DIR, "registered_staff_full.csv")
EXCEL_FILE = os.path.splitext(CSV_FILE)[0] + ".xlsx"
FACES_DIR = os.path.join(BASE_DIR, "captured_faces")
os.makedirs(FACES_DIR, exist_ok=True)

EMBED_COLS = [f"e{i}" for i in range(128)]
CSV_COLS = [
    "Staff Name", "Project Name", "State", "District", "Designation",
    "Gender", "Email ID", "Contact No", "Employee Since (DD-MM-YYYY)",
    "Latitude", "Longitude", "Timestamp", "Image Path"
] + EMBED_COLS

if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
    pd.DataFrame(columns=CSV_COLS).to_csv(CSV_FILE, index=False)

# dlib model paths (place files here if you want embeddings)
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
RECOGNITION_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"

has_dlib_models = (dlib is not None and
                   os.path.exists(PREDICTOR_PATH) and
                   os.path.exists(RECOGNITION_MODEL_PATH))
if has_dlib_models:
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(PREDICTOR_PATH)
    facerec = dlib.face_recognition_model_v1(RECOGNITION_MODEL_PATH)
else:
    detector = sp = facerec = None

# State->districts (kept short; you can extend)
STATE_DISTRICTS = {
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Kurnool"],
    "Karnataka": ["Bengaluru", "Mysuru", "Mangaluru"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"],
    "Delhi": ["New Delhi"],
    "Uttar Pradesh": ["Lucknow", "Kanpur"],
    "West Bengal": ["Kolkata"],
    "Bihar": ["Patna"],
    "Jharkhand": ["Ranchi"]
}
GENDERS = ["Male", "Female", "Other"]

# ---------------- utils ----------------
def is_valid_email(email: str) -> bool:
    import re
    return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', (email or "").strip()))

def get_geo_coords_google(api_key=None):
    """Try Google geolocation API if key supplied; else IP-based fallback."""
    if api_key:
        try:
            url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={api_key}"
            r = requests.post(url, json={"considerIp": True}, timeout=6)
            if r.ok:
                j = r.json()
                loc = j.get("location")
                if loc:
                    return float(loc.get("lat")), float(loc.get("lng"))
        except Exception:
            pass
    # IP fallback
    try:
        r = requests.get("https://ipinfo.io/json", timeout=6)
        if r.ok:
            j = r.json()
            loc = j.get("loc")
            if loc:
                lat, lon = loc.split(",")
                return float(lat), float(lon)
    except Exception:
        pass
    # Ranchi fallback
    return 23.3441, 85.3096

def compute_embedding_from_bgr(frame_bgr):
    """Compute 128-d embedding using dlib if available. Accepts BGR OpenCV frame."""
    if detector is None or facerec is None or cv2 is None:
        return np.zeros(128, dtype=float)
    # convert to RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    try:
        dets = detector(rgb)
        if len(dets) == 0:
            return None
        shape = sp(rgb, dets[0])
        emb = np.array(facerec.compute_face_descriptor(rgb, shape), dtype=float)
        return emb
    except Exception:
        return None

# ---------------- WebRTC VideoProcessor ----------------
class FaceVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None  # store last frame (BGR numpy array)
        self.lock = threading.Lock()

    def recv(self, frame):
        # ensure av is present
        if av is None:
            return frame
        img = frame.to_ndarray(format="bgr24")
        # store frame
        with self.lock:
            self.frame = img.copy()
        # overlay face boxes (draw rectangles)
        if detector is not None and cv2 is not None:
            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                dets = detector(rgb)
                for d in dets:
                    x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except Exception:
                pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# RTC configuration (STUN to help browser connect)
RTC_CONFIGURATION = None
try:
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
except Exception:
    RTC_CONFIGURATION = None

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Face Registration (WebRTC)", layout="wide")
st.title("Face Registration — Streamlit WebRTC (converted from Tkinter)")

col_left, col_right = st.columns([1, 1])

# initialize session flags for compatibility
if "webrtc_active" not in st.session_state:
    st.session_state["webrtc_active"] = False
if "webrtc_context" not in st.session_state:
    st.session_state["webrtc_context"] = None

# left side: live camera + map + start/stop/capture controls (to mimic Tkinter)
with col_left:
    st.subheader("Live Camera")
    st.markdown("Use **Start Camera** to open live preview, **Stop Camera** to close it. Click **Capture** to save current frame.")
    # Control buttons mimic Tkinter Start/Stop/Capture
    tcol1, tcol2, tcol3 = st.columns([1,1,1])
    with tcol1:
        if st.button("Start Camera"):
            # mount the streamer component by setting flag
            st.session_state["webrtc_active"] = True
    with tcol2:
        if st.button("Stop Camera"):
            # unmount streamer by clearing flag and context
            st.session_state["webrtc_active"] = False
            # try to cleanup previous context
            try:
                st.session_state["webrtc_context"] = None
            except Exception:
                pass

    # When active: create the webrtc streamer component (older versions will automatically show controls)
    webrtc_ctx = None
    if st.session_state["webrtc_active"]:
        if webrtc_streamer is None:
            st.error("streamlit-webrtc not installed. Install with: pip install streamlit-webrtc")
        else:
            # create / re-create component
            webrtc_ctx = webrtc_streamer(
                key="webrtc-face-register",
                mode=WebRtcMode.SENDRECV if WebRtcMode is not None else None,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=FaceVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            st.session_state["webrtc_context"] = webrtc_ctx
    else:
        st.info("Camera is stopped. Click Start Camera to begin preview.")
    # Capture & Save (uses whatever frame is available)
    with tcol3:
        if st.button("Capture & Save (from live)"):
            proc = None
            # prefer direct context from session_state
            proc = st.session_state.get("webrtc_context")
            if proc is None:
                st.error("Camera is not running. Start camera first.")
            else:
                # get video_processor
                vp = None
                try:
                    vp = proc.video_processor
                except Exception:
                    vp = None
                frame = None
                if vp is not None:
                    try:
                        with vp.lock:
                            frame = vp.frame.copy() if vp.frame is not None else None
                    except Exception:
                        frame = None
                if frame is None:
                    st.error("No frame available to capture. Make sure camera is started and visible.")
                else:
                    st.session_state["_captured_frame_bgr"] = frame
                    st.success("Captured live frame — fill form and click Save on right side (or press Capture here again).")

    st.markdown("---")

    # Map (click to choose location)
    st.subheader("Map / Location")
    lat_default, lon_default = get_geo_coords_google()
    if folium is not None and st_folium is not None:
        m = folium.Map(location=[lat_default, lon_default], zoom_start=10)
        marker = folium.Marker(location=[lat_default, lon_default], draggable=True)
        marker.add_to(m)
        fmap = st_folium(m, width=700, height=350)
        # get last clicked coordinates (streamlit-folium returns a dict)
        lat_selected = lat_default
        lon_selected = lon_default
        if fmap:
            if "last_clicked" in fmap and fmap["last_clicked"]:
                lat_selected = fmap["last_clicked"]["lat"]
                lon_selected = fmap["last_clicked"]["lng"]
        st.write(f"Selected location: {lat_selected:.6f}, {lon_selected:.6f}")
    else:
        st.warning("streamlit_folium or folium not installed; using IP/default coordinates.")
        lat_selected, lon_selected = lat_default, lon_default
        st.write(f"Location: {lat_selected:.6f}, {lon_selected:.6f}")

# right side: form with same fields and Save/Export behavior
with col_right:
    st.subheader("Register New Face (Form)")

    with st.form("reg_form"):
        project_name = st.text_input("Project Name")
        staff_name = st.text_input("Staff Name")
        designation = st.text_input("Designation")
        email = st.text_input("Email ID")
        contact_no = st.text_input("Contact No (10 digits)")
        gender = st.selectbox("Gender", GENDERS)
        state = st.selectbox("State", list(STATE_DISTRICTS.keys()))
        districts = STATE_DISTRICTS.get(state, [])
        district = st.selectbox("District", districts if districts else [""])
        employee_since = st.date_input("Employee Since")
        # option: user may have captured frame already via live Capture button, or upload snapshot
        upload_option = st.radio("Image source", ("Live capture (Start Camera → Capture)", "Upload image file"))
        uploaded_file = None
        if upload_option.startswith("Upload"):
            uploaded_file = st.file_uploader("Upload face image (jpg/png)", type=["jpg", "jpeg", "png"])

        save_btn = st.form_submit_button("Save (Register)")

    # show dlib/model state
    if dlib is None:
        st.warning("dlib not installed — embeddings will be zeros. To enable embeddings, install dlib and place model files.")
    elif not has_dlib_models:
        st.warning(f"dlib installed but model files not found: {PREDICTOR_PATH}, {RECOGNITION_MODEL_PATH} — embeddings will be zeros.")
    else:
        st.success("dlib + models present — embeddings will be computed.")

# process saving
if save_btn:
    # validations
    missing = [k for k,v in {
        "Staff Name": staff_name,
        "Project Name": project_name,
        "State": state,
        "District": district,
        "Designation": designation,
        "Gender": gender,
        "Email ID": email,
        "Contact No": contact_no,
        "Employee Since (Date)": employee_since
    }.items() if not v]
    if missing:
        st.error(f"Please fill: {', '.join(missing)}")
    elif not (contact_no.isdigit() and len(contact_no)==10):
        st.error("Contact No must be exactly 10 digits.")
    elif not is_valid_email(email):
        st.error("Invalid email address.")
    else:
        # prepare image source
        bgr_frame = None
        if upload_option.startswith("Upload") and uploaded_file is not None:
            try:
                pil = Image.open(uploaded_file).convert("RGB")
                # convert to BGR numpy
                arr = np.array(pil)[:, :, ::-1].copy()
                bgr_frame = arr
            except Exception as e:
                st.error(f"Failed to read uploaded image: {e}")
        else:
            # prefer live-captured frame saved in session by the Capture & Save button
            if "_captured_frame_bgr" in st.session_state:
                bgr_frame = st.session_state["_captured_frame_bgr"]
            else:
                # fallback: try grabbing current processor frame if camera running
                proc = st.session_state.get("webrtc_context")
                vp = None
                if proc is not None:
                    try:
                        vp = proc.video_processor
                    except Exception:
                        vp = None
                if vp is not None:
                    try:
                        with vp.lock:
                            if vp.frame is not None:
                                bgr_frame = vp.frame.copy()
                    except Exception:
                        bgr_frame = None
        if bgr_frame is None:
            st.error("No image available. Start camera & press Capture, or upload an image.")
        else:
            # compute embedding
            emb = None
            if detector is not None and cv2 is not None:
                emb = compute_embedding_from_bgr(bgr_frame)
            if emb is None:
                # embedding failed or face not detected; use zeros
                emb = np.zeros(128, dtype=float)
                st.warning("Face not detected or embedding failed — saving row with zero embeddings.")

            # save image file as jpeg
            pil_save = Image.fromarray(bgr_frame[:, :, ::-1])  # BGR->RGB
            safe_name = re.sub(r"[^\w\d_-]", "_", staff_name)
            safe_proj = re.sub(r"[^\w\d_-]", "_", project_name)
            fname = f"{safe_name}_{safe_proj}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            img_path = os.path.join(FACES_DIR, fname)
            try:
                pil_save.save(img_path, format="JPEG", quality=90)
            except Exception as e:
                st.error(f"Failed to save image: {e}")
                img_path = ""

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # lat_selected/lon_selected come from left side; ensure they exist
            # (if user clicked map using streamlit_folium we stored them earlier)
            try:
                lat_final = float(lat_selected)
                lon_final = float(lon_selected)
            except Exception:
                lat_final, lon_final = get_geo_coords_google()

            # build row
            row = {
                "Staff Name": staff_name,
                "Project Name": project_name,
                "State": state,
                "District": district,
                "Designation": designation,
                "Gender": gender,
                "Email ID": email,
                "Contact No": contact_no,
                "Employee Since (DD-MM-YYYY)": employee_since.strftime('%d-%m-%Y'),
                "Latitude": float(lat_final),
                "Longitude": float(lon_final),
                "Timestamp": timestamp,
                "Image Path": img_path
            }
            for i in range(128):
                row[f"e{i}"] = float(emb[i])

            try:
                df = pd.read_csv(CSV_FILE)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                df.to_csv(CSV_FILE, index=False)
                df.to_excel(EXCEL_FILE, index=False, engine="openpyxl")
                st.success(f"Face registered for {staff_name} — saved to {CSV_FILE}")
                with st.expander("Preview saved row"):
                    st.write(pd.DataFrame([row]))
                # clear captured frame from session after save
                if "_captured_frame_bgr" in st.session_state:
                    del st.session_state["_captured_frame_bgr"]
            except Exception as e:
                st.error(f"Failed to save data: {e}")

# Export & Download area (like Tkinter Export button)
st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    if st.button("Export to Excel"):
        try:
            df = pd.read_csv(CSV_FILE)
            df.to_excel(EXCEL_FILE, index=False, engine="openpyxl")
            st.success(f"Exported to {EXCEL_FILE}")
        except Exception as e:
            st.error(f"Export failed: {e}")
with col_b:
    try:
        with open(CSV_FILE, "rb") as f:
            st.download_button("Download CSV", data=f, file_name=os.path.basename(CSV_FILE), mime="text/csv")
    except Exception:
        st.write("CSV not found yet.")

st.caption("This Streamlit WebRTC UI preserves the left-camera+map + right-form layout and restores Start, Stop, Capture, Save, Export behaviors from the original Tkinter app.")
