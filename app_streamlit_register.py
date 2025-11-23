import os
import io
import re
from datetime import datetime
import threading
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# ==============================
#        AUTHENTICATION
# ==============================
USERNAME = "New_face"
PASSWORD = "12345"

def show_login_screen():
    st.set_page_config(page_title="Login", layout="centered")

    # Centered layout using empty columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align:center;'>Login</h2>", unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == USERNAME and password == PASSWORD:
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("Invalid username or password")

# Show login first
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    show_login_screen()
    st.stop()

# ==============================
#   ORIGINAL APP STARTS HERE
# ==============================

# streamlit-webrtc imports
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
except Exception:
    webrtc_streamer = None
    VideoProcessorBase = object
    RTC_CONFIGURATION = None
    WebRtcMode = None

try:
    import av
except Exception:
    av = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import dlib
except Exception:
    dlib = None

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

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
RECOGNITION_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"

has_dlib_models = (
    dlib is not None and
    os.path.exists(PREDICTOR_PATH) and
    os.path.exists(RECOGNITION_MODEL_PATH)
)

if has_dlib_models:
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(PREDICTOR_PATH)
    facerec = dlib.face_recognition_model_v1(RECOGNITION_MODEL_PATH)
else:
    detector = sp = facerec = None

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

def is_valid_email(email):
    return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', (email or "").strip()))

def get_geo_coords():
    try:
        r = requests.get("https://ipinfo.io/json", timeout=6)
        if r.ok:
            loc = r.json().get("loc")
            if loc:
                lat, lon = loc.split(",")
                return float(lat), float(lon)
    except:
        pass
    return 23.3441, 85.3096  # Ranchi fallback

def compute_embedding_from_bgr(frame_bgr):
    if detector is None or facerec is None or cv2 is None:
        return np.zeros(128, dtype=float)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    try:
        dets = detector(rgb)
        if len(dets) == 0:
            return None
        shape = sp(rgb, dets[0])
        emb = np.array(facerec.compute_face_descriptor(rgb, shape), dtype=float)
        return emb
    except:
        return None

class FaceVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()

    def recv(self, frame):
        if av is None:
            return frame
        img = frame.to_ndarray(format="bgr24")

        with self.lock:
            self.frame = img.copy()

        if detector is not None and cv2 is not None:
            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                dets = detector(rgb)
                for d in dets:
                    cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0,255,0), 2)
            except:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

try:
    RTC_CONFIGURATION = RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})
except:
    RTC_CONFIGURATION = None

# ======================================================
#         ORIGINAL LAYOUT (UNCHANGED)
# ======================================================
st.set_page_config(page_title="Face Registration", layout="wide")
st.title("Face Registration")

col_left, col_right = st.columns([1, 1])

if "webrtc_active" not in st.session_state:
    st.session_state["webrtc_active"] = False
if "webrtc_context" not in st.session_state:
    st.session_state["webrtc_context"] = None

# LEFT COLUMN (Camera + Map)
with col_left:
    st.subheader("Live Camera")
    st.markdown("Start → Stop → Capture same as Tkinter")

    b1, b2, b3 = st.columns([1,1,1])

    with b1:
        if st.button("Start Camera"):
            st.session_state["webrtc_active"] = True

    with b2:
        if st.button("Stop Camera"):
            st.session_state["webrtc_active"] = False
            st.session_state["webrtc_context"] = None

    webrtc_ctx = None
    if st.session_state["webrtc_active"]:
        if webrtc_streamer is None:
            st.error("streamlit-webrtc not installed")
        else:
            webrtc_ctx = webrtc_streamer(
                key="regcam",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=FaceVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
            st.session_state["webrtc_context"] = webrtc_ctx
    else:
        st.info("Camera stopped.")

    with b3:
        if st.button("Capture"):
            ctx = st.session_state["webrtc_context"]
            if ctx:
                vp = ctx.video_processor
                if vp:
                    with vp.lock:
                        if vp.frame is not None:
                            st.session_state["_captured_frame_bgr"] = vp.frame.copy()
                            st.success("Captured.")
                        else:
                            st.error("No frame yet.")
            else:
                st.error("Start camera first.")

    st.subheader("Map / Location")
    lat_default, lon_default = get_geo_coords()

    if folium and st_folium:
        m = folium.Map(location=[lat_default, lon_default], zoom_start=10)
        marker = folium.Marker(
            location=[lat_default, lon_default],
            draggable=True
        )
        marker.add_to(m)
        fmap = st_folium(m, width=700, height=350)

        lat_selected = lat_default
        lon_selected = lon_default

        if fmap and "last_clicked" in fmap and fmap["last_clicked"]:
            lat_selected = fmap["last_clicked"]["lat"]
            lon_selected = fmap["last_clicked"]["lng"]

        st.write(f"Selected: {lat_selected:.6f}, {lon_selected:.6f}")
    else:
        st.write(f"Location: {lat_default}, {lon_default}")

# RIGHT COLUMN (Form)
with col_right:
    st.subheader("Register New Face")
    with st.form("regform"):
        project = st.text_input("Project Name")
        name = st.text_input("Staff Name")
        designation = st.text_input("Designation")
        email = st.text_input("Email")
        phone = st.text_input("Contact No (10 digits)")
        gender = st.selectbox("Gender", GENDERS)
        state = st.selectbox("State", list(STATE_DISTRICTS.keys()))
        district = st.selectbox("District", STATE_DISTRICTS.get(state, [""]))
        employee_since = st.date_input("Employee Since")
        source = st.radio("Image Source", ["Live Capture", "Upload File"])
        file = None
        if source == "Upload File":
            file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
        save = st.form_submit_button("Save")

# SAVE PROCESS
if save:
    missing = [k for k,v in {
        "Project": project, "Name": name,
        "Designation": designation, "State": state,
        "District": district, "Email": email,
        "Phone": phone
    }.items() if not v]

    if missing:
        st.error("Missing: " + ", ".join(missing))
    elif not phone.isdigit() or len(phone)!=10:
        st.error("Invalid phone")
    elif not is_valid_email(email):
        st.error("Invalid email")
    else:
        bgr = None

        if source == "Upload File" and file:
            pil = Image.open(file).convert("RGB")
            bgr = np.array(pil)[:, :, ::-1]

        elif source == "Live Capture":
            if "_captured_frame_bgr" in st.session_state:
                bgr = st.session_state["_captured_frame_bgr"]

        if bgr is None:
            st.error("No image available")
        else:
            emb = compute_embedding_from_bgr(bgr)
            if emb is None:
                emb = np.zeros(128)
                st.warning("Embedding failed.")

            pil_save = Image.fromarray(bgr[:, :, ::-1])
            fname = f"{name}_{project}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            path = os.path.join(FACES_DIR, fname)
            pil_save.save(path)

            row = {
                "Staff Name": name,
                "Project Name": project,
                "State": state,
                "District": district,
                "Designation": designation,
                "Gender": gender,
                "Email ID": email,
                "Contact No": phone,
                "Employee Since (DD-MM-YYYY)": employee_since.strftime('%d-%m-%Y'),
                "Latitude": float(lat_selected),
                "Longitude": float(lon_selected),
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Image Path": path
            }
            for i in range(128):
                row[f"e{i}"] = float(emb[i])

            df = pd.read_csv(CSV_FILE)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(CSV_FILE, index=False)
            df.to_excel(EXCEL_FILE, index=False)

            st.success("Saved!")

# EXPORT
st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    if st.button("Export Excel"):
        df = pd.read_csv(CSV_FILE)
        df.to_excel(EXCEL_FILE, index=False)
        st.success("Exported!")

with c2:
    with open(CSV_FILE, "rb") as f:
        st.download_button("Download CSV", f, file_name="registered_staff_full.csv")

st.caption("Layout preserved exactly as original Tkinter → Streamlit WebRTC.")
