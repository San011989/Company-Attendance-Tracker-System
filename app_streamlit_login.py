# FULL CODE START
# (Your full provided code is inserted exactly, unchanged)

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
import streamlit.components.v1 as components

GOOGLE_API_KEY = "AIzaSyB1m4dSIy5-zuqCi1TcRy_a7Hepisd4Zck"

def inject_js_for_gps():
    components.html(
        """
        <script>
        navigator.geolocation.getCurrentPosition(
            function(position) {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                const url = new URL(window.location.href);
                url.searchParams.set('lat', lat);
                url.searchParams.set('lon', lon);
                window.history.replaceState(null, "", url);
            },
            function(error) {},
            { enableHighAccuracy: true, timeout: 10000 }
        );
        </script>
        """,
        height=0,
    )

inject_js_for_gps()

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False
if "cap" not in st.session_state:
    st.session_state.cap = None
if "seen_names" not in st.session_state:
    st.session_state.seen_names = {}
if "attendance" not in st.session_state:
    st.session_state.attendance = pd.DataFrame(columns=[
        "Name", "Project Name", "Designation", "Gender", "Email ID",
        "Contact No", "Employee Since", "Latitude", "Longitude",
        "City", "State", "Timestamp", "ImagePath", "Status"
    ])

try:
    from geopy.geocoders import Nominatim
except Exception:
    Nominatim = None

st.set_page_config(page_title="Company Attendance Tracker", layout="wide")
st.title("📸 Company Attendance Tracker (Live Face Recognition + Geo Tagging)")

REGISTERED_CSV = "registered_staff_full.csv"
CAPTURE_DIR = "captured_images"
EXPORT_DIR = "attendance_exports"

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

if not os.path.exists(REGISTERED_CSV):
    st.error(f"Registered CSV not found: {REGISTERED_CSV}.")
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
    st.warning("No embedding columns found.")

known_names = df_registered["Staff Name"].tolist()
known_embeddings = df_registered[embedding_cols].to_numpy(dtype=np.float64) if embedding_cols else np.empty((0, 128))

try:
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
except Exception:
    st.error("dlib models missing.")
    st.stop()

def google_geolocate():
    if not GOOGLE_API_KEY:
        return None
    try:
        url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={GOOGLE_API_KEY}"
        r = requests.post(url, json={"considerIp": True}, timeout=8)
        if r.ok:
            j = r.json()
            loc = j.get("location")
            if loc:
                return float(loc.get("lat")), float(loc.get("lng"))
    except Exception:
        pass
    return None

def ip_geolocate():
    try:
        r = requests.get("https://ipinfo.io/json", timeout=6)
        if r.ok:
            j = r.json()
            loc = j.get("loc")
            if loc:
                lat_str, lon_str = loc.split(",")
                return float(lat_str), float(lon_str)
    except Exception:
        pass
    return 25.5941, 85.1376

def get_current_location():
    try:
        qp = st.query_params()
        lat_str = qp.get("lat", [None])[0]
        lon_str = qp.get("lon", [None])[0]
        if lat_str and lon_str:
            return float(lat_str), float(lon_str)
    except Exception:
        pass

    g = google_geolocate()
    if g:
        return g

    return ip_geolocate()

def google_reverse_geocode(lat, lon):
    if not GOOGLE_API_KEY:
        return "", ""
    try:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={GOOGLE_API_KEY}"
        r = requests.get(url, timeout=8)
        if r.ok:
            j = r.json()
            if j.get("results"):
                components = j["results"][0].get("address_components", [])
                city = ""
                state = ""
                for comp in components:
                    types = comp.get("types", [])
                    if "locality" in types and not city:
                        city = comp.get("long_name", "")
                    if "administrative_area_level_1" in types and not state:
                        state = comp.get("long_name", "")
                if not city:
                    formatted = j["results"][0].get("formatted_address", "")
                    return formatted, state
                return city, state
    except Exception:
        pass

    if Nominatim:
        try:
            geolocator = Nominatim(user_agent="attendance_app_reverse_geocode")
            loc = geolocator.reverse(f"{lat}, {lon}", language="en", timeout=5)
            if loc and "address" in loc.raw:
                addr = loc.raw["address"]
                city = addr.get("city") or addr.get("town") or addr.get("village") or ""
                state = addr.get("state") or ""
                return city, state
        except Exception:
            pass

    return "", ""

def fetch_osm_minimap(lat, lon, size=150, zoom=16):
    try:
        url = f"https://staticmap.openstreetmap.de/staticmap.php?center={lat},{lon}&zoom={zoom}&size={size}x{size}&markers={lat},{lon},red-pushpin"
        r = requests.get(url, timeout=8)
        if r.ok:
            return Image.open(io.BytesIO(r.content)).convert("RGBA")
    except Exception:
        pass
    return None

def save_attendance_image_with_overlays(
    frame_bgr,
    out_path,
    lat,
    lon,
    place_text="",
    name_text=None,
    rect=None,
    minimap_size=150
):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb).convert("RGBA")
    w, h = pil_img.size
    draw = ImageDraw.Draw(pil_img)

    minimap = fetch_osm_minimap(lat, lon, minimap_size)
    if minimap is None:
        minimap = Image.new("RGBA", (minimap_size, minimap_size), (60, 60, 60, 255))
        ImageDraw.Draw(minimap).text((10, 60), "No Map", fill=(255, 255, 255, 255))

    pad = 15
    x_map = w - minimap_size - pad
    y_map = h - minimap_size - pad
    pil_img.paste(minimap, (x_map, y_map), minimap)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    coords_text = f"Lat: {lat:.5f}, Lng: {lon:.5f}"
    place_line = f"{place_text}" if place_text else "Unknown Location"

    text_lines = [timestamp, coords_text, place_line]
    line_spacing = 6
    text_block_w = max([draw.textbbox((0, 0), t, font=font)[2] for t in text_lines]) + 20
    text_block_h = sum([draw.textbbox((0, 0), t, font=font)[3] for t in text_lines]) + (
        len(text_lines) - 1
    ) * line_spacing + 20

    rect_x1 = pad
    rect_y1 = h - text_block_h - pad
    rect_x2 = rect_x1 + text_block_w
    rect_y2 = h - pad
    draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill=(0, 0, 0, 180))

    ty = rect_y1 + 10
    for t in text_lines:
        draw.text((rect_x1 + 10, ty), t, fill=(255, 255, 255, 255), font=font)
        ty += draw.textbbox((0, 0), t, font=font)[3] + line_spacing

    if name_text and rect is not None:
        try:
            font_name = ImageFont.truetype("arial.ttf", 22)
        except:
            font_name = ImageFont.load_default()
        label = str(name_text)
        tw, th = draw.textbbox((0, 0), label, font=font_name)[2:]
        x1, y1 = rect.left(), rect.top()
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 6, y1], fill=(0, 0, 0, 180))
        draw.text((x1 + 3, y1 - th - 3), label, fill=(255, 255, 255, 255), font=font_name)

    pil_img.convert("RGB").save(out_path, format="JPEG", quality=95)

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
    dists = np.linalg.norm(known_embeddings - emb, axis=1)
    min_idx = int(np.argmin(dists))
    if dists[min_idx] < 0.6:
        return known_names[min_idx], d
    return None, d

def persist_attendance_excel():
    try:
        df = st.session_state.attendance
        if df.empty:
            return None
        dt = datetime.now().strftime("%Y-%m-%d")
        xlsx_path = os.path.join(EXPORT_DIR, f"attendance_{dt}.xlsx")
        df.to_excel(xlsx_path, index=False)
        return xlsx_path
    except:
        return None

def open_camera(index=0):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW, cv2.CAP_ANY]
    for b in backends:
        try:
            cap = cv2.VideoCapture(index, b)
            if cap and cap.isOpened():
                return cap
        except:
            pass
    try:
        cap = cv2.VideoCapture(index)
        if cap and cap.isOpened():
            return cap
    except:
        pass
    return None

col_left, col_right = st.columns([1, 1])
with col_left:
    st.subheader("Live Camera Feed")
    frame_placeholder = st.empty()
with col_right:
    st.subheader("Live Map (Live view)")
    lat0, lon0 = get_current_location()
    map_placeholder = st.empty()

    def render_map(lat, lon):
        span = 0.005
        left = lon - span
        right = lon + span
        top = lat + span
        bottom = lat - span

        osm_embed_url = (
            f"https://www.openstreetmap.org/export/embed.html?bbox={left}%2C{bottom}%2C{right}%2C{top}"
            f"&layer=mapnik&marker={lat}%2C{lon}"
        )
        map_html = f"""
        <iframe id="liveMap" src="{osm_embed_url}" width="100%" height="400" style="border:0;"></iframe>
        """
        map_placeholder.markdown(map_html, unsafe_allow_html=True)

    render_map(lat0, lon0)

st.markdown("### Controls")
cc1, cc2, cc3, cc4 = st.columns(4)
with cc1:
    if st.button("Start Camera"):
        if not st.session_state.camera_on:
            cap = open_camera(0)
            if cap is None:
                st.error("Cannot open webcam.")
            else:
                st.session_state.cap = cap
                st.session_state.camera_on = True
                st.success("Camera started.")
                st.rerun()

with cc2:
    if st.button("Stop Camera"):
        if st.session_state.camera_on:
            st.session_state.camera_on = False
            if st.session_state.cap:
                try:
                    st.session_state.cap.release()
                except:
                    pass
                st.session_state.cap = None
            st.info("Camera stopped.")
            st.rerun()

with cc3:
    if st.button("Export to Excel"):
        path = persist_attendance_excel()
        if path:
            with open(path, "rb") as f:
                st.download_button("Download Excel", f, file_name=os.path.basename(path))
        else:
            st.warning("No attendance data.")

with cc4:
    st.write("Tip: Stop camera before closing.")

if st.session_state.camera_on:
    cap = st.session_state.cap
    if cap is None or not cap.isOpened():
        st.error("Camera not opened.")
        st.session_state.camera_on = False
        st.stop()

    frame_ok = False
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            frame_ok = True
            break
        time.sleep(0.1)

    if not frame_ok:
        frame_placeholder.warning("Camera initializing…")
        time.sleep(0.5)
        st.rerun()
    else:
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        dets = detector(frame_rgb)
        for d in dets:
            x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

        recognized, rect = recognize_face_from_frame(frame_rgb)
        now_ts = time.time()

        if recognized:
            already_marked = recognized in st.session_state.attendance["Name"].values
            if not already_marked:
                last_seen = st.session_state.seen_names.get(recognized, 0)
                bbox_width = rect.right() - rect.left() if rect else 0
                if now_ts - last_seen > 8 and bbox_width >= 60:
                    st.session_state.seen_names[recognized] = now_ts

                    lat, lon = get_current_location()
                    city, state = google_reverse_geocode(lat, lon)
                    place_text = ", ".join(p for p in [city, state] if p)

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    safe_ts = timestamp.replace(":", "-").replace(" ", "_")
                    person_dir = os.path.join(CAPTURE_DIR, recognized)
                    os.makedirs(person_dir, exist_ok=True)
                    img_name = f"{recognized}_{safe_ts}.jpg"
                    img_path = os.path.join(person_dir, img_name)

                    try:
                        save_attendance_image_with_overlays(frame, img_path, lat, lon, place_text, recognized, rect)
                        saved_image = True
                    except:
                        saved_image = False

                    row = df_registered[df_registered["Staff Name"] == recognized]
                    if not row.empty:
                        row_data = row.iloc[0]
                        entry = {
                            "Name": row_data["Staff Name"],
                            "Project Name": row_data.get("Project Name", ""),
                            "Designation": row_data.get("Designation", ""),
                            "Gender": row_data.get("Gender", ""),
                            "Email ID": row_data.get("Email ID", ""),
                            "Contact No": row_data.get("Contact No", ""),
                            "Employee Since": row_data.get("Employee Since (DD-MM-YYYY)", ""),
                            "Latitude": lat,
                            "Longitude": lon,
                            "City": city,
                            "State": state,
                            "Timestamp": timestamp,
                            "ImagePath": img_path if saved_image else "",
                            "Status": "Present"
                        }
                    else:
                        entry = {
                            "Name": recognized,
                            "Project Name": "",
                            "Designation": "",
                            "Gender": "",
                            "Email ID": "",
                            "Contact No": "",
                            "Employee Since": "",
                            "Latitude": lat,
                            "Longitude": lon,
                            "City": city,
                            "State": state,
                            "Timestamp": timestamp,
                            "ImagePath": img_path if saved_image else "",
                            "Status": "Present"
                        }

                    new_df = pd.DataFrame([entry])
                    st.session_state.attendance = pd.concat([st.session_state.attendance, new_df], ignore_index=True)

                    persist_attendance_excel()

                    st.success(f"Attendance marked for {recognized}")
                    if saved_image and os.path.exists(img_path):
                        st.image(Image.open(img_path), width=700)

        frame_placeholder.image(frame_rgb, channels="RGB")

        ret, _ = cap.read()
        if not ret:
            st.warning("Camera lost.")
            st.session_state.camera_on = False
        else:
            time.sleep(0.1)
            st.rerun()

st.subheader("🧾 Live Attendance Records")
st.dataframe(st.session_state.attendance, width="stretch")

today_xlsx = os.path.join(EXPORT_DIR, f"attendance_{datetime.now().strftime('%Y-%m-%d')}.xlsx")
if os.path.exists(today_xlsx):
    with open(today_xlsx, "rb") as f:
        st.download_button("Download Today's Excel", f, file_name=os.path.basename(today_xlsx))

# FULL CODE END
