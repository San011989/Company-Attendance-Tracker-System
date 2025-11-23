# app_streamlit.py
# Final: capture image, overlay Google minimap + timestamp + coords + place (on-image),
# save to captured_images/<StaffName>/attendance_YYYY-MM-DD_HH-MM-SS.jpg,
# update in-memory attendance table, show preview + dashboard in Streamlit.

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

# ---------------- CONFIG ----------------
# Put your real key here (you asked to hard-code it)
GOOGLE_API_KEY = "AIzaSyB1m4dSIy5-zuqCi1TcRy_a7Hepisd4Zck"  # <-- REPLACE with your real Google API key

# ---------------- Inject browser GPS JS ----------------
def inject_js_for_gps():
    components.html(
        """
        <script>
        // attempt to get browser GPS and store in URL params
        navigator.geolocation.getCurrentPosition(
            function(position) {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                const url = new URL(window.location.href);
                url.searchParams.set('lat', lat);
                url.searchParams.set('lon', lon);
                window.history.replaceState(null, "", url);
                console.log("GPS coords put into URL params:", lat, lon);
            },
            function(error) {
                console.warn("Browser GPS error:", error.message);
            },
            { enableHighAccuracy: true, timeout: 10000 }
        );
        </script>
        """,
        height=0,
    )

inject_js_for_gps()

# ---------------- Streamlit state ----------------
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

# optional geopy for fallback reverse geocoding
try:
    from geopy.geocoders import Nominatim
except Exception:
    Nominatim = None

# ---------------- Page config ----------------
st.set_page_config(page_title="Company Attendance Tracker", layout="wide")
st.title("📸 Company Attendance Tracker (Live Face Recognition + Geo Tagging)")

# ---------------- Paths ----------------
REGISTERED_CSV = "registered_staff_full.csv"
CAPTURE_DIR = "captured_images"
EXPORT_DIR = "attendance_exports"

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# ---------------- Validate registered CSV ----------------
if not os.path.exists(REGISTERED_CSV):
    st.error(f"Registered CSV not found: {REGISTERED_CSV}. Place it in the script directory.")
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

# embedding columns (e0..e127)
embedding_cols = [c for c in df_registered.columns if c.startswith("e")]
if len(embedding_cols) == 0:
    st.warning("No embedding columns found — recognition won't work without embeddings.")

known_names = df_registered["Staff Name"].tolist()
known_embeddings = df_registered[embedding_cols].to_numpy(dtype=np.float64) if embedding_cols else np.empty((0, 128))

# ---------------- Load dlib models ----------------
try:
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
except Exception as e:
    st.error("dlib models not found or could not be loaded. Place them in the working directory.")
    st.stop()

# ---------------- Geo helpers ----------------
def google_geolocate():
    """Google Geolocation API to get lat/lng."""
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
    """IP-based geolocation fallback (ipinfo.io)."""
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
    return 25.5941, 85.1376  # fallback center

def get_current_location():
    """Prefer browser params -> Google Geolocation -> IP fallback."""
    try:
        qp = st.query_params()
        lat_str = qp.get("lat", [None])[0]
        lon_str = qp.get("lon", [None])[0]
        if lat_str and lon_str:
            try:
                return float(lat_str), float(lon_str)
            except Exception:
                pass
    except Exception:
        pass

    if GOOGLE_API_KEY:
        g = google_geolocate()
        if g:
            return g

    return ip_geolocate()

def google_reverse_geocode(lat, lon):
    """Reverse geocode using Google Geocoding API to get a place name."""
    if not GOOGLE_API_KEY:
        return "", ""
    try:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={GOOGLE_API_KEY}"
        r = requests.get(url, timeout=8)
        if r.ok:
            j = r.json()
            if j.get("results"):
                # prefer locality / administrative_area_level_1
                components = j["results"][0].get("address_components", [])
                city = ""
                state = ""
                for comp in components:
                    types = comp.get("types", [])
                    if "locality" in types and not city:
                        city = comp.get("long_name", "")
                    if "administrative_area_level_1" in types and not state:
                        state = comp.get("long_name", "")
                # fallback to formatted address if city empty
                if not city:
                    formatted = j["results"][0].get("formatted_address", "")
                    return formatted, state
                return city, state
    except Exception:
        pass

    # fallback Nominatim
    try:
        if Nominatim:
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

def fetch_google_static_map(lat, lon, size=150, zoom=16):
    """Fetch Google Static Map (small minimap). Returns PIL Image or None."""
    if not GOOGLE_API_KEY:
        return None
    try:
        marker = f"color:red%7C{lat},{lon}"
        url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size={size}x{size}&maptype=roadmap&markers={marker}&key={GOOGLE_API_KEY}"
        r = requests.get(url, timeout=8)
        if r.ok:
            return Image.open(io.BytesIO(r.content)).convert("RGBA")
    except Exception:
        pass
    return None

def fetch_osm_minimap(lat, lon, size=150, zoom=16):
    """OSM static minimap fallback."""
    try:
        url = f"https://staticmap.openstreetmap.de/staticmap.php?center={lat},{lon}&zoom={zoom}&size={size}x{size}&markers={lat},{lon},red-pushpin"
        r = requests.get(url, timeout=8)
        if r.ok:
            return Image.open(io.BytesIO(r.content)).convert("RGBA")
    except Exception:
        pass
    return None

# ---------------- Image composition: embed minimap and text overlay ----------------
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
    """
    Save captured frame with:
      • Google minimap (bottom-right, inside image)
      • Timestamp + coordinates + place text (bottom-left, inside image)
      • No image shrinking
    """
    import cv2, io, requests
    from datetime import datetime
    from PIL import Image, ImageDraw, ImageFont

    # Convert to PIL (RGB)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb).convert("RGBA")
    w, h = pil_img.size
    draw = ImageDraw.Draw(pil_img)

    # === 1. Fetch Google Static Map ===
    minimap = None
    if GOOGLE_API_KEY:
        try:
            url = (
                f"https://maps.googleapis.com/maps/api/staticmap"
                f"?center={lat},{lon}&zoom=16&size={minimap_size}x{minimap_size}"
                f"&markers=color:red%7C{lat},{lon}&key={GOOGLE_API_KEY}"
            )
            resp = requests.get(url)
            if resp.status_code == 200:
                minimap = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        except Exception:
            pass
    if minimap is None:
        minimap = Image.new("RGBA", (minimap_size, minimap_size), (60, 60, 60, 255))
        ImageDraw.Draw(minimap).text((10, 60), "No Map", fill=(255, 255, 255, 255))

    # === 2. Paste minimap inside bottom-right corner ===
    pad = 15
    x_map = w - minimap_size - pad
    y_map = h - minimap_size - pad
    pil_img.paste(minimap, (x_map, y_map), minimap)

    # === 3. Prepare overlay text ===
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

    # === 4. Draw semi-transparent rectangle for text (bottom-left) ===
    rect_x1 = pad
    rect_y1 = h - text_block_h - pad
    rect_x2 = rect_x1 + text_block_w
    rect_y2 = h - pad
    draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill=(0, 0, 0, 180))

    # === 5. Draw text lines inside rectangle ===
    ty = rect_y1 + 10
    for t in text_lines:
        draw.text((rect_x1 + 10, ty), t, fill=(255, 255, 255, 255), font=font)
        ty += draw.textbbox((0, 0), t, font=font)[3] + line_spacing

    # === 6. Optional name label ===
    if name_text and rect is not None:
        x1, y1 = rect.left(), rect.top()
        try:
            font_name = ImageFont.truetype("arial.ttf", 22)
        except:
            font_name = ImageFont.load_default()
        label = str(name_text)
        tw, th = draw.textbbox((0, 0), label, font=font_name)[2:]
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 6, y1], fill=(0, 0, 0, 180))
        draw.text((x1 + 3, y1 - th - 3), label, fill=(255, 255, 255, 255), font=font_name)

    # === 7. Save final image ===
    pil_img.convert("RGB").save(out_path, format="JPEG", quality=95)


# ---------------- Recognition helper ----------------
def recognize_face_from_frame(frame_rgb):
    """
    Return (recognized_name, rect) or (None, None)
    """
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

# ---------------- Persist attendance Excel ----------------
def persist_attendance_excel():
    try:
        df = st.session_state.attendance
        if df.empty:
            return None
        dt = datetime.now().strftime("%Y-%m-%d")
        xlsx_path = os.path.join(EXPORT_DIR, f"attendance_{dt}.xlsx")
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
            if cap is not None and cap.isOpened():
                return cap
            else:
                try:
                    cap.release()
                except Exception:
                    pass
        except Exception:
            pass
    try:
        cap = cv2.VideoCapture(index)
        if cap is not None and cap.isOpened():
            return cap
    except Exception:
        pass
    return None

# ---------------- UI Layout ----------------
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
        <script>
        setInterval(() => {{
            navigator.geolocation.getCurrentPosition(
                (pos) => {{
                    const lat = pos.coords.latitude;
                    const lon = pos.coords.longitude;
                    const span = 0.005;
                    const left = lon - span;
                    const right = lon + span;
                    const top = lat + span;
                    const bottom = lat - span;
                    const url = "https://www.openstreetmap.org/export/embed.html?bbox=" +
                                encodeURIComponent(left) + "%2C" + encodeURIComponent(bottom) + "%2C" +
                                encodeURIComponent(right) + "%2C" + encodeURIComponent(top) +
                                "&layer=mapnik&marker=" + encodeURIComponent(lat) + "%2C" + encodeURIComponent(lon);
                    document.getElementById('liveMap').src = url;
                }},
                (err) => {{ /* ignore */ }},
                {{ enableHighAccuracy: true, timeout: 8000 }}
            );
        }}, 10000);
        </script>
        """
        map_placeholder.markdown(map_html, unsafe_allow_html=True)

    render_map(lat0, lon0)

# ---------------- Controls ----------------
st.markdown("### Controls")
cc1, cc2, cc3, cc4 = st.columns(4)
with cc1:
    if st.button("Start Camera"):
        if not st.session_state.camera_on:
            cap = open_camera(index=0)
            if cap is None:
                st.error("Unable to open camera. Check privacy settings or other apps using it.")
            else:
                st.session_state.cap = cap
                st.session_state.camera_on = True
                st.success("Camera started.")
                st.rerun()
with cc2:
    if st.button("Stop Camera"):
        if st.session_state.camera_on:
            st.session_state.camera_on = False
            if st.session_state.cap is not None:
                try:
                    st.session_state.cap.release()
                except Exception:
                    pass
                st.session_state.cap = None
            st.info("Camera stopped and released.")
            st.rerun()
with cc3:
    if st.button("Export to Excel"):
        path = persist_attendance_excel()
        if path:
            with open(path, "rb") as f:
                st.download_button("Download Excel", f, file_name=os.path.basename(path))
            st.success(f"Saved Excel: {os.path.basename(path)}")
        else:
            st.warning("No attendance rows to export.")
with cc4:
    st.write("Tip: press Stop Camera before closing the app.")

# ---------------- Main camera loop ----------------
if st.session_state.camera_on:
    cap = st.session_state.cap
    if cap is None or not getattr(cap, "isOpened", lambda: False)():
        st.error("Camera object missing or not opened. Stopping camera.")
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        st.session_state.cap = None
        st.session_state.camera_on = False
        st.stop()

    # warm-up
    frame_ok = False
    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_ok = True
            break
        time.sleep(0.1)

    if not frame_ok:
        frame_placeholder.warning("⚠️ Camera initializing... please wait 2–3 seconds.")
        time.sleep(0.5)
        st.rerun()
    else:
        frame = cv2.flip(frame, 1)
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            frame_rgb = frame

        # Draw rectangles on detections for UI
        dets = detector(frame_rgb)
        for d in dets:
            x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Recognition
        recognized, rect = recognize_face_from_frame(frame_rgb)
        now_ts = time.time()

        if recognized:
            already_marked = recognized in st.session_state.attendance["Name"].values
            if already_marked:
                st.info(f"Attendance already marked for {recognized}.")
            else:
                last_seen = st.session_state.seen_names.get(recognized, 0)
                bbox_width = 0
                try:
                    bbox_width = rect.right() - rect.left()
                except Exception:
                    bbox_width = 0

                if now_ts - last_seen > 8 and bbox_width >= 60:
                    st.session_state.seen_names[recognized] = now_ts

                    # Obtain location and place
                    lat, lon = get_current_location()
                    city, state = google_reverse_geocode(lat, lon)
                    place_text = ", ".join(p for p in [city, state] if p).strip()

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    safe_ts = timestamp.replace(":", "-").replace(" ", "_")
                    person_dir = os.path.join(CAPTURE_DIR, recognized)
                    os.makedirs(person_dir, exist_ok=True)
                    img_name = f"{recognized}_{safe_ts}.jpg"
                    img_path = os.path.join(person_dir, img_name)

                    # Save attendance image with overlays (minimap embedded bottom-right, text at bottom)
                    saved_image = False
                    try:
                        save_attendance_image_with_overlays(frame, img_path, lat, lon, place_text, name_text=recognized, rect=rect, minimap_size=150)
                        saved_image = True
                    except Exception as e:
                        st.warning(f"Failed to create/save attendance image: {e}")
                        # fallback save raw frame
                        try:
                            face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            Image.fromarray(face_rgb).save(img_path)
                            saved_image = True
                        except Exception:
                            saved_image = False

                    # Build entry and append
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

                    # Append to session DataFrame
                    new_df = pd.DataFrame([entry])
                    dfs = [df for df in [st.session_state.attendance, new_df] if not df.empty]
                    if dfs:
                        st.session_state.attendance = pd.concat(dfs, ignore_index=True)
                    else:
                        st.session_state.attendance = pd.DataFrame(columns=st.session_state.attendance.columns)

                    # Persist Excel
                    persist_attendance_excel()

                    # UI feedback and display saved image
                    st.success(f"Attendance marked for {recognized}")
                    if saved_image and os.path.exists(img_path):
                        try:
                            im = Image.open(img_path)
                            st.image(im, caption=f"Saved attendance image for {recognized}", use_column_width=False, width=700)
                        except Exception:
                            pass

                        # Show live dashboard info under the image
                        st.markdown("**Attendance Details**")
                        st.write(f"**Name:** {entry['Name']}")
                        st.write(f"**Timestamp:** {entry['Timestamp']}")
                        st.write(f"**Coordinates:** {entry['Latitude']:.5f}, {entry['Longitude']:.5f}")
                        st.write(f"**Location:** {entry['City']}, {entry['State']}")
                        st.write(f"**Saved Path:** `{entry['ImagePath']}`")
                    else:
                        st.warning("Image was not saved; attendance recorded in session only.")

        # show live frame
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # next frame / rerun
        ret, next_frame = cap.read()
        if not ret:
            st.warning("Camera feed lost.")
            st.session_state.camera_on = False
        else:
            time.sleep(0.1)
            st.rerun()

# ---------------- Live attendance table ----------------
st.subheader("🧾 Live Attendance Records")
st.dataframe(st.session_state.attendance, width="stretch")

# today's excel download
today_xlsx = os.path.join(EXPORT_DIR, f"attendance_{datetime.now().strftime('%Y-%m-%d')}.xlsx")
if os.path.exists(today_xlsx):
    with open(today_xlsx, "rb") as f:
        st.download_button("Download Today's Excel", f, file_name=os.path.basename(today_xlsx))

# ---------------- Footer ----------------
st.markdown("""
**Notes:**  
- Saved image contains the captured photo with a small Google (or OSM) minimap overlay in the bottom-right, and the timestamp, coordinates and place printed on the bottom (white text on semi-transparent black).  
- Ensure `GOOGLE_API_KEY` has Geolocation, Geocoding, and Static Maps enabled if you want the most accurate results.  
""")
