# app_streamlit_logout.py
# Streamlit app for Employee Logout Tracking (same layout/design as attendance app)

import os
import io
import time
from datetime import datetime

import streamlit as st
import cv2
import dlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import requests
import plotly.express as px

# ---------------- Google Maps API key ----------------
GOOGLE_API_KEY = "AIzaSyB1m4dSIy5-zuqCi1TcRy_a7Hepisd4Zck"  # ← replace with your real key


# ----------------- Streamlit State Initialization -----------------
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

if "cap" not in st.session_state:
    st.session_state.cap = None

if "seen_names" not in st.session_state:
    st.session_state.seen_names = {}

if "logout_records" not in st.session_state:
    st.session_state.logout_records = pd.DataFrame(columns=[
        "Name", "Project Name", "Designation", "Gender", "Email ID",
        "Contact No", "Employee Since", "Latitude", "Longitude",
        "City", "State", "Timestamp", "ImagePath", "Status"
    ])

# ---------------- optional geopy ----------------
try:
    from geopy.geocoders import Nominatim
except Exception:
    Nominatim = None

# ---------------- Page config & title ----------------
st.set_page_config(page_title="Employee Logout Tracker", layout="wide")
st.title("📸 Company Logout Tracker (Live Face Recognition + Geo Tagging)")

# ---------------- Paths & folders ----------------
REGISTERED_CSV = "registered_staff_full.csv"
CAPTURE_DIR = "captured_images"
EXPORT_DIR = "logout_exports"

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# ---------------- Validate registered CSV ----------------
if not os.path.exists(REGISTERED_CSV):
    st.error(f"Registered CSV not found: {REGISTERED_CSV}. Run get_faces_from_camera_tkinter.py first.")
    st.stop()

df_registered = pd.read_csv(REGISTERED_CSV)

required_cols = [
    "Staff Name", "Project Name", "State", "District", "Designation",
    "Gender", "Email ID", "Contact No", "Employee Since (DD-MM-YYYY)"
]
for col in required_cols:
    if col not in df_registered.columns:
        st.error(f"Registered CSV missing expected column: {col}")
        st.stop()

embedding_cols = [c for c in df_registered.columns if c.startswith("e")]
if len(embedding_cols) == 0:
    st.warning("No embedding columns found in registered CSV — recognition won't work without embeddings.")

known_names = df_registered["Staff Name"].tolist()
known_embeddings = df_registered[embedding_cols].to_numpy(dtype=np.float64) if embedding_cols else np.empty((0, 128))

# ---------------- Load Dlib models ----------------
try:
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
except Exception:
    st.error("dlib models not found or could not be loaded. Ensure the files exist in working directory.")
    st.stop()

# ---------------- Google Geo helpers ----------------
def get_current_location():
    """Get accurate current latitude and longitude using Google Geolocation API."""
    try:
        url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={GOOGLE_API_KEY}"
        r = requests.post(url, timeout=5)
        if r.ok:
            j = r.json()
            if "location" in j:
                lat = j["location"]["lat"]
                lon = j["location"]["lng"]
                return lat, lon
    except Exception:
        pass
    return 20.5937, 78.9629  # fallback: India center


def reverse_geocode(lat, lon):
    """Get city and state using Google Geocoding API."""
    try:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={GOOGLE_API_KEY}"
        r = requests.get(url, timeout=5)
        if r.ok:
            j = r.json()
            if j.get("results"):
                addr = j["results"][0]["address_components"]
                city = ""
                state = ""
                for comp in addr:
                    if "locality" in comp["types"]:
                        city = comp["long_name"]
                    if "administrative_area_level_1" in comp["types"]:
                        state = comp["long_name"]
                return city, state
    except Exception:
        pass
    return "", ""


# ---------------- Recognition helper ----------------
def recognize_face_from_frame(frame_rgb):
    if known_embeddings.size == 0:
        return None, None
    dets = detector(frame_rgb)
    if len(dets) == 0:
        return None, None
    sizes = [(d.right()-d.left())*(d.bottom()-d.top()) for d in dets]
    best_idx = int(np.argmax(sizes))
    d = dets[best_idx]
    shape = sp(frame_rgb, d)
    emb = np.array(facerec.compute_face_descriptor(frame_rgb, shape), dtype=np.float64)
    if emb.size == 0:
        return None, d
    dists = np.linalg.norm(known_embeddings - emb, axis=1)
    if dists.size == 0:
        return None, d
    min_idx = int(np.argmin(dists))
    if dists[min_idx] < 0.6:
        return known_names[min_idx], d
    return None, d

# ---------------- Save image with overlays ----------------
# ---------------- Save image with Google roadmap mini-map ----------------
def save_frame_with_minimap_and_geotag(frame_rgb, out_path, lat, lon, name_text=None, rect=None):
    import io
    from PIL import Image, ImageDraw, ImageFont
    import requests

    pil = Image.fromarray(np.uint8(frame_rgb)).convert("RGB")
    draw = ImageDraw.Draw(pil, "RGBA")
    w, h = pil.size
    timestamp = datetime.now().strftime("%d-%b-%Y %H:%M:%S")

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

    # Reverse geocode using Google API
    city, state = reverse_geocode(lat, lon)
    location_text = f"{city}, {state}" if city or state else "Unknown Location"
    geo_text = f"Lat: {lat:.5f}, Lon: {lon:.5f}"

    # --- Draw text info ---
    info_lines = [f"Time: {timestamp}", location_text, geo_text]
    text_y = h - 20
    padding = 10
    for line in reversed(info_lines):
        tw, th = draw.textbbox((0, 0), line, font=font)[2:]
        text_y -= th + 5
        draw.rectangle([10, text_y - 5, 10 + tw + padding, text_y + th + 5], fill=(0, 0, 0, 160))
        draw.text((15, text_y), line, font=font, fill=(255, 255, 255, 255))

    # --- Google roadmap static map ---
    map_w, map_h = 200, 120
    map_x, map_y = w - map_w - 10, h - map_h - 10
    try:
        map_url = (
            f"https://maps.googleapis.com/maps/api/staticmap"
            f"?center={lat},{lon}&zoom=15&size={map_w}x{map_h}"
            f"&markers=color:red|{lat},{lon}&maptype=roadmap&key={GOOGLE_API_KEY}"
        )
        r = requests.get(map_url, timeout=5)
        if r.ok:
            map_img = Image.open(io.BytesIO(r.content)).convert("RGBA")
            pil.paste(map_img, (map_x, map_y), map_img)
    except Exception:
        pass

    # --- Optional: draw name near face box ---
    if name_text and rect is not None:
        try:
            x1, y1 = rect.left(), rect.top()
            label = str(name_text)
            lw, lh = draw.textbbox((0, 0), label, font=font)[2:]
            lx0 = max(6, x1)
            ly0 = max(6, y1 - lh - 10)
            draw.rectangle([lx0 - 4, ly0 - 4, lx0 + lw + 6, ly0 + lh + 4], fill=(0, 0, 0, 160))
            draw.text((lx0 + 2, ly0), label, font=font, fill=(255, 255, 255, 255))
        except Exception:
            pass

    pil.save(out_path, "JPEG", quality=90)


# ---------------- Persist Excel ----------------
def persist_logout_excel():
    try:
        df = st.session_state.logout_records
        if df.empty:
            return None
        dt = datetime.now().strftime("%Y-%m-%d")
        xlsx_path = os.path.join(EXPORT_DIR, f"logout_{dt}.xlsx")
        df.to_excel(xlsx_path, index=False)
        return xlsx_path
    except Exception as e:
        st.warning(f"Error saving Excel: {e}")
        return None

# ---------------- Camera open ----------------
def open_camera(index=0):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW, cv2.CAP_ANY]
    for b in backends:
        try:
            cap = cv2.VideoCapture(index, b)
            if cap and cap.isOpened():
                return cap
        except Exception:
            pass
    cap = cv2.VideoCapture(index)
    return cap if cap.isOpened() else None

# ---------------- UI Layout ----------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Live Camera Feed")
    frame_placeholder = st.empty()

with col_right:
    st.subheader("Live Map (Google Roadmap)")
    lat0, lon0 = get_current_location()
    map_url = (
        f"https://www.google.com/maps/embed/v1/view"
        f"?key={GOOGLE_API_KEY}&center={lat0},{lon0}&zoom=14&maptype=roadmap"
    )
    st.markdown(
        f'<iframe src="{map_url}" width="100%" height="400" style="border:0;" allowfullscreen="" loading="lazy"></iframe>',
        unsafe_allow_html=True
    )


# ---------------- Controls ----------------
st.markdown("### Controls")
cc1, cc2, cc3, cc4 = st.columns(4)

with cc1:
    if st.button("Start Camera"):
        if not st.session_state.camera_on:
            cap = open_camera(index=0)
            if cap is None:
                st.error("Unable to open camera.")
            else:
                st.session_state.cap = cap
                st.session_state.camera_on = True
                st.success("Camera started.")
                st.rerun()

with cc2:
    if st.button("Stop Camera"):
        if st.session_state.camera_on:
            if st.session_state.cap:
                st.session_state.cap.release()
            st.session_state.camera_on = False
            st.info("Camera stopped.")
            st.rerun()

with cc3:
    if st.button("Export to Excel"):
        path = persist_logout_excel()
        if path:
            with open(path, "rb") as f:
                st.download_button("Download Logout Excel", f, file_name=os.path.basename(path))
            st.success(f"Saved Excel: {os.path.basename(path)}")
        else:
            st.warning("No logout entries yet.")

with cc4:
    st.write("Tip: press Stop Camera before closing the app.")

# ---------------- Main camera loop ----------------
if st.session_state.camera_on:
    cap = st.session_state.cap
    if cap is None or not cap.isOpened():
        st.error("Camera not available.")
        st.session_state.camera_on = False
        st.stop()

    ret, frame = cap.read()
    if not ret:
        st.warning("Camera initializing...")
        time.sleep(0.5)
        st.rerun()

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    dets = detector(frame_rgb)
    for d in dets:
        cv2.rectangle(frame_rgb, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), 2)

    recognized, rect = recognize_face_from_frame(frame_rgb)
    now_ts = time.time()

    if recognized:
        already_marked = recognized in st.session_state.logout_records["Name"].values
        if already_marked:
            st.info(f"Logout already marked for {recognized}.")
        else:
            last_seen = st.session_state.seen_names.get(recognized, 0)
            if now_ts - last_seen > 8:
                st.session_state.seen_names[recognized] = now_ts

                lat, lon = get_current_location()
                city, state = reverse_geocode(lat, lon)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                safe_ts = timestamp.replace(":", "-").replace(" ", "_")

                # --- Save per-person folder ---
                person_dir = os.path.join(CAPTURE_DIR, recognized)
                os.makedirs(person_dir, exist_ok=True)
                img_name = f"{recognized}_{safe_ts}_logout.jpg"
                img_path = os.path.join(person_dir, img_name)

                try:
                    save_frame_with_minimap_and_geotag(frame_rgb, img_path, lat, lon, name_text=recognized, rect=rect)
                except Exception:
                    Image.fromarray(frame_rgb.astype(np.uint8)).save(img_path)

                row = df_registered[df_registered["Staff Name"] == recognized]
                if not row.empty:
                    row = row.iloc[0]
                    entry = {
                        "Name": row["Staff Name"],
                        "Project Name": row["Project Name"],
                        "Designation": row["Designation"],
                        "Gender": row["Gender"],
                        "Email ID": row["Email ID"],
                        "Contact No": row["Contact No"],
                        "Employee Since": row["Employee Since (DD-MM-YYYY)"],
                        "Latitude": lat,
                        "Longitude": lon,
                        "City": city,
                        "State": state,
                        "Timestamp": timestamp,
                        "ImagePath": img_path,
                        "Status": "Logout"
                    }
                    new_df = pd.DataFrame([entry])
                    st.session_state.logout_records = pd.concat(
                        [st.session_state.logout_records, new_df],
                        ignore_index=True
                    )
                    persist_logout_excel()
                    st.success(f"Logout marked for {recognized}")
                    if os.path.exists(img_path):
                        st.image(Image.open(img_path), caption=f"Saved image for {recognized}", width=300)

    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    time.sleep(0.1)
    st.rerun()

# ---------------- Live table ----------------
st.subheader("🧾 Live Logout Records")
st.dataframe(st.session_state.logout_records, width="stretch")

today_xlsx = os.path.join(EXPORT_DIR, f"logout_{datetime.now().strftime('%Y-%m-%d')}.xlsx")
if os.path.exists(today_xlsx):
    with open(today_xlsx, "rb") as f:
        st.download_button("Download Today's Logout Excel", f, file_name=os.path.basename(today_xlsx))

st.markdown("""
**Notes:**  
- Layout and design identical to login app.  
- Saves logout images in name-wise folders.  
- Exports data as `logout_<date>.xlsx`.  
""")
