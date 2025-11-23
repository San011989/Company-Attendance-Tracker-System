"""
Final polished Face Registration GUI
- Left half: Camera (top) + interactive map (below, medium height)
- Right half: Single-column inputs, increased spacing
- Buttons under inputs (styled, larger font, hover effect)
- Auto geo-load from IP with retries, fallback to India center
- Camera index 0
- Saves image + CSV + Excel including 128-d embedding
"""

import os
import re
import cv2
import dlib
import threading
import time
import numpy as np
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import requests
from tkintermapview import TkinterMapView




# ---------------- Google Maps API key ----------------
GOOGLE_API_KEY = "AIzaSyB1m4dSIy5-zuqCi1TcRy_a7Hepisd4Zck"  # ← Replace with your real key


# optional ui/map/calendar libs
try:
    from tkintermapview import TkinterMapView
except Exception:
    TkinterMapView = None

try:
    from tkcalendar import DateEntry
except Exception:
    DateEntry = None

# ------------------- Config -------------------
CSV_FILE = "registered_staff_full.csv"
EXCEL_FILE = os.path.splitext(CSV_FILE)[0] + ".xlsx"
FACES_DIR = "captured_faces"
os.makedirs(FACES_DIR, exist_ok=True)

EMBED_COLS = [f"e{i}" for i in range(128)]
CSV_COLS = [
    "Staff Name", "Project Name", "State", "District", "Designation",
    "Gender", "Email ID", "Contact No", "Employee Since (DD-MM-YYYY)",
    "Latitude", "Longitude", "Timestamp", "Image Path"
] + EMBED_COLS
if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
    pd.DataFrame(columns=CSV_COLS).to_csv(CSV_FILE, index=False)

# dlib models (must exist)
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
RECOGNITION_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"
if not os.path.exists(PREDICTOR_PATH) or not os.path.exists(RECOGNITION_MODEL_PATH):
    raise FileNotFoundError(
        f"Missing dlib models. Place '{PREDICTOR_PATH}' and '{RECOGNITION_MODEL_PATH}' in the script folder."
    )

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(PREDICTOR_PATH)
facerec = dlib.face_recognition_model_v1(RECOGNITION_MODEL_PATH)

# State -> districts (sample)
STATE_DISTRICTS = {
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Kurnool"],
    "Karnataka": ["Bengaluru", "Mysuru", "Mangaluru", "Hubballi", "Belagavi"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Thane"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"],
    "Delhi": ["New Delhi", "Central Delhi", "North Delhi", "South Delhi"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi"],
    "West Bengal": ["Kolkata", "Howrah"],
    "Bihar": ["Patna","Nalanda","Bhojpur","Rohtas","Buxar","Muzaffarpur","Vaishali","Sitamarhi","Sheohar","East Champaran","West Champaran","Saran","Siwan","Gopalganj","Darbhanga","Madhubani","Samastipur","Saharsa","Supaul","Madhepura","Purnea","Araria","Katihar","Kishanganj","Bhagalpur","Banka","Munger","Lakhisarai","Sheikhpura","Jamui","Khagaria","Gaya","Nawada","Aurangabad","Jehanabad","Arwal"],
    "Jharkhand": ["Ranchi","Khunti","Lohardaga","Gumla","Simdega","Latehar","Palamu","Garhwa","Chatra","Hazaribagh","Ramgarh","Koderma","Giridih","Bokaro","Dhanbad","Jamtara","Deoghar","Dumka","Pakur","Sahebganj","Godda","West Singhbhum","East Singhbhum","Seraikela-Kharsawan"],

}
GENDERS = ["Male", "Female", "Other"]

# ------------------- Utilities -------------------
#def get_geo_coords_with_retry(retries: int = 2, timeout: int = 6):
 #   """Use Google Geolocation API with fallback to ipapi.co if unavailable."""
  #  geo_url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={GOOGLE_API_KEY}"
   # for attempt in range(retries + 1):
    #    try:
     #       r = requests.post(geo_url, timeout=timeout)
      #      if r.ok:
       #         j = r.json()
        #        if "location" in j:
         #           lat = j["location"]["lat"]
          #          lon = j["location"]["lng"]
           #         return lat, lon
        #except Exception:
         #   time.sleep(0.5 + attempt * 0.5)

    # fallback: ipapi.co if Google fails
    #try:
     #   r = requests.get("https://ipapi.co/json/", timeout=timeout)
      #  if r.status_code == 200:
       #     j = r.json()
        #    return j.get("latitude", 20.5937), j.get("longitude", 78.9629)
    #except Exception:
     #   pass

    #return 20.5937, 78.9629  # fallback to India center

import requests

def get_geo_coords_with_retry_google(api_key, retries=3, timeout=15):
    """
    Try Google Geolocation API first; if it fails, fall back to IP-based or default Ranchi coords.
    """
    google_url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={api_key}"
    payload = {"considerIp": True}

    # --- Try Google API ---
    for attempt in range(retries):
        try:
            res = requests.post(google_url, json=payload, timeout=timeout)
            if res.status_code == 200:
                data = res.json()
                lat = data["location"]["lat"]
                lon = data["location"]["lng"]
                print("[Geo] Google API location found:", lat, lon)
                return lat, lon
            else:
                print(f"[Geo Retry {attempt+1}] HTTP {res.status_code}: {res.text}")
        except Exception as e:
            print(f"[Geo Retry {attempt+1}] Error: {e}")

    # --- Fallback 1: IP-based location (ipinfo.io) ---
    try:
        ipinfo = requests.get("https://ipinfo.io/json", timeout=8).json()
        lat, lon = map(float, ipinfo["loc"].split(","))
        print(f"[Geo Fallback] Using IP-based location: {lat}, {lon}")
        return lat, lon
    except Exception as e:
        print("[Geo Fallback] IP-based lookup failed:", e)

    # --- Fallback 2: Default Ranchi coordinates ---
    print("[Geo Default] Using static Ranchi coordinates")
    return 23.3441, 85.3096

def is_valid_email(email):
    return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', (email or "").strip()))

# ------------------- Simple Login Window -------------------
class LoginWindow:
    def __init__(self, master, on_success):
        self.master = master
        self.on_success = on_success
        master.title("Login")
        master.geometry("300x180")
        master.resizable(False, False)

        tk.Label(master, text="User Login", font=("Helvetica", 14, "bold")).pack(pady=10)

        tk.Label(master, text="Username:").pack()
        self.user_entry = ttk.Entry(master, width=25)
        self.user_entry.pack(pady=2)

        tk.Label(master, text="Password:").pack()
        self.pass_entry = ttk.Entry(master, width=25, show="*")
        self.pass_entry.pack(pady=2)

        self.login_btn = tk.Button(master, text="Login", bg="#2980B9", fg="white",
                                   font=("Helvetica", 11, "bold"), command=self.check_login)
        self.login_btn.pack(pady=10)

        # simple hardcoded login
        self.correct_user = "New_face"
        self.correct_pass = "12345"

    def check_login(self):
        if self.user_entry.get() == self.correct_user and self.pass_entry.get() == self.correct_pass:
            self.master.destroy()
            self.on_success()
        else:
            messagebox.showerror("Login Failed", "Invalid username or password!")


# ------------------- App -------------------
class FaceRegisterApp:
    def __init__(self):
        # require tkintermapview
        if TkinterMapView is None:
            messagebox.showerror(
                "Missing dependency",
                "tkintermapview is required. Install it in your face_env:\n\npip install git+https://github.com/TomSchimansky/TkinterMapView.git"
            )
            raise SystemExit("tkintermapview missing")

        self.root = tk.Tk()
        self.root.title("Face Registration System")
        # fixed large-screen layout
        self.root.geometry("1200x700")
        self.root.configure(bg="#F5F6FA")

        # camera state
        self.cap = None
        self.camera_running = False
        self.camera_index = 0  # user confirmed 0

        # geo state
        self.latitude = 20.5937
        self.longitude = 78.9629
        self.map_marker = None

        # widget store
        self.widgets = {}

        # layout split 50/50 horizontally
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # ---------- LEFT HALF (camera + map stacked) ----------
        left_frame = ttk.Frame(self.root)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        left_frame.grid_rowconfigure(0, weight=2)  # camera larger
        left_frame.grid_rowconfigure(1, weight=1)  # map medium
        left_frame.grid_columnconfigure(0, weight=1)

        # Camera frame
        cam_frame = ttk.Frame(left_frame, relief="sunken")
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=(4,2))
        cam_frame.grid_rowconfigure(0, weight=1)
        cam_frame.grid_columnconfigure(0, weight=1)
        self.video_label = tk.Label(cam_frame, bg="black")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Map frame
        map_frame = ttk.Frame(left_frame, relief="sunken")
        map_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=(2,4))
        map_frame.grid_rowconfigure(0, weight=1)
        map_frame.grid_columnconfigure(0, weight=1)
        self.map_widget = TkinterMapView(map_frame, corner_radius=0)
        self.map_widget.set_tile_server(
            f"https://mt0.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}&key={GOOGLE_API_KEY}")

        self.map_widget.grid(row=0, column=0, sticky="nsew")
        # set default then async update
        self.map_widget.set_position(self.latitude, self.longitude)
        self.map_widget.set_zoom(6)

        # ---------- RIGHT HALF (single column inputs) ----------
        right_frame = tk.Frame(self.root, bg="#FFFFFF", bd=1, relief="solid")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        right_frame.grid_rowconfigure(0, weight=0)
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        title = tk.Label(right_frame, text="Register New Face", font=("Helvetica", 16, "bold"), bg="#FFFFFF")
        title.grid(row=0, column=0, sticky="ew", pady=(8, 12), padx=10)

        form = tk.Frame(right_frame, bg="#FFFFFF")
        form.grid(row=1, column=0, sticky="n", padx=8)
        ENTRY_W = 36
        CB_W = 34
        pad_label = 6
        pad_entry = 10
        font_label = ("Helvetica", 11, "bold")
        font_input = ("Helvetica", 11)

        # helper to add label & entry with extra spacing
        def add_label_entry(parent, label_text, key):
            lbl = tk.Label(parent, text=label_text + ":", font=font_label, bg="#FFFFFF", anchor="w")
            lbl.pack(fill="x", padx=6, pady=(pad_label, 0))
            ent = ttk.Entry(parent, width=ENTRY_W, font=font_input)
            ent.pack(fill="x", padx=6, pady=(0, pad_entry))
            self.widgets[key] = ent

        def add_label_combobox(parent, label_text, key, values):
            lbl = tk.Label(parent, text=label_text + ":", font=font_label, bg="#FFFFFF", anchor="w")
            lbl.pack(fill="x", padx=6, pady=(pad_label, 0))
            cb = ttk.Combobox(parent, values=values, width=CB_W, state="readonly", font=font_input)
            cb.pack(fill="x", padx=6, pady=(0, pad_entry))
            self.widgets[key] = cb
            return cb

        # Fields (single column)
        add_label_entry(form, "Project Name", "Project Name")
        add_label_entry(form, "Staff Name", "Staff Name")
        add_label_entry(form, "Designation", "Designation")
        add_label_entry(form, "Email ID", "Email ID")
        add_label_entry(form, "Contact No (10 digits)", "Contact No")

        add_label_combobox(form, "Gender", "Gender", GENDERS)

        cb_state = add_label_combobox(form, "State", "State", list(STATE_DISTRICTS.keys()))
        cb_state.bind("<<ComboboxSelected>>", self._on_state_selected)

        add_label_combobox(form, "District", "District", [])

        # Employee Since (date picker if available)
        lbl = tk.Label(form, text="Employee Since (Date):", font=font_label, bg="#FFFFFF", anchor="w")
        lbl.pack(fill="x", padx=6, pady=(pad_label, 0))
        if DateEntry is not None:
            date_w = DateEntry(form, date_pattern="dd-mm-yyyy", width=20)
            date_w.pack(fill="x", padx=6, pady=(0, pad_entry))
            self.widgets["Employee Since (Date)"] = date_w
        else:
            ent_date = ttk.Entry(form, width=ENTRY_W, font=font_input)
            ent_date.pack(fill="x", padx=6, pady=(0, pad_entry))
            self.widgets["Employee Since (Date)"] = ent_date

        # location & timestamp display
        self.loc_display = tk.Label(form, text="Location: Fetching...    Timestamp: --", bg="#FFFFFF", font=("Helvetica", 10, "italic"), anchor="w")
        self.loc_display.pack(fill="x", padx=6, pady=(8, 12))

        # ---------- Styled buttons (same design as original) ----------
        # simple color theme (kept consistent with earlier design)
        BTN_BG = "#2980B9"
        BTN_FG = "white"
        BTN_WIDTH = 16
        BTN_HEIGHT = 2
        BTN_FONT = ("Helvetica", 12, "bold")
        BTN_ACTIVE = "#1F618D"

        btn_container = tk.Frame(form, bg="#FFFFFF")
        btn_container.pack(fill="x", pady=(6, 6))

        def make_button(text, cmd, col=0):
            b = tk.Button(btn_container, text=text, command=cmd,
                          bg=BTN_BG, fg=BTN_FG, activebackground=BTN_ACTIVE,
                          activeforeground=BTN_FG, font=BTN_FONT, bd=0,
                          width=BTN_WIDTH, height=BTN_HEIGHT, relief="flat")
            # hover effect
            def on_enter(e):
                b['bg'] = BTN_ACTIVE
            def on_leave(e):
                b['bg'] = BTN_BG
            b.bind("<Enter>", on_enter)
            b.bind("<Leave>", on_leave)
            return b

        # Buttons: Start, Capture & Save, Stop, Export, Exit (Exit separate below)
        btn_start = make_button("Start Camera", self.start_camera)
        btn_capture = make_button("Capture & Save", self.capture_and_save)
        btn_stop = make_button("Stop Camera", self.stop_camera)
        btn_export = make_button("Export to Excel", self.export_to_excel)

        # place buttons in a row
        btn_start.grid(row=0, column=0, padx=6)
        btn_capture.grid(row=0, column=1, padx=6)
        btn_stop.grid(row=0, column=2, padx=6)
        btn_export.grid(row=0, column=3, padx=6)

        # Exit button (smaller, under the row)
        exit_btn = tk.Button(form, text="Exit", command=self.on_exit,
                             bg="#C0392B", fg="white", activebackground="#922B21",
                             font=BTN_FONT, bd=0, width=BTN_WIDTH, height=BTN_HEIGHT, relief="flat")
        exit_btn.pack(pady=(12,4))

        # convenience capture button left below map (keeps previous UX)
        left_capture = tk.Button(left_frame, text="Capture & Save", command=self.capture_and_save,
                                 bg=BTN_BG, fg=BTN_FG, activebackground=BTN_ACTIVE, font=BTN_FONT, bd=0, width=20, height=1)
        left_capture.grid(row=2, column=0, sticky="ew", padx=8, pady=(6,4))

        # async geo + map initialization
        threading.Thread(target=self._init_geo_and_map, daemon=True).start()

        # close protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

    # ----------------- State selection -----------------
    def _on_state_selected(self, event):
        state = self.widgets["State"].get()
        districts = STATE_DISTRICTS.get(state, [])
        cb = self.widgets.get("District")
        if cb is not None:
            cb['values'] = districts
            if districts:
                cb.current(0)

    # ----------------- Geo & map init -----------------
   # def _init_geo_and_map(self):
        #lat, lon = get_geo_coords_with_retry(retries=2, timeout=6)
    #    lat, lon = get_geo_coords_with_retry_google(GOOGLE_API_KEY, retries=2, timeout=6)

#        self.latitude, self.longitude = lat, lon
        # update label on main thread
 #       self.root.after(0, self._update_loc_display)
        # update map marker & center on main thread
  #      def update_map():
   #         try:
    #            if self.map_widget:
     #               self.map_widget.set_position(self.latitude, self.longitude)
      #              self.map_widget.set_zoom(12)
       #             if self.map_marker:
        #                try:
         #                   self.map_marker.delete()
          #              except Exception:
           #                 pass
            #        self.map_marker = self.map_widget.set_marker(self.latitude, self.longitude, text="Current Location")
            #except Exception:
             #   pass
        #self.root.after(0, update_map)

    def _init_geo_and_map(self):
        # get initial guess (Google Geolocation or fallback)
        lat, lon = get_geo_coords_with_retry_google(GOOGLE_API_KEY, retries=2, timeout=15)
        self.latitude, self.longitude = lat, lon

        def setup_map():
            try:
                if not self.map_widget:
                    return

                # center and zoom
                self.map_widget.set_position(lat, lon)
                self.map_widget.set_zoom(12)

                # remove existing marker
                if self.map_marker:
                    try:
                        self.map_marker.delete()
                    except Exception:
                        pass

                # create initial marker (no callback attached here)
                self.map_marker = self.map_widget.set_marker(
                    lat, lon, text="Click map to correct location"
                )

                # handler for map clicks: tkintermapview passes either (lat, lon) tuple
                # or an object; we defensively unpack both cases.
                def handle_map_click(coords):
                    try:
                        # coords often comes as a tuple (lat, lon)
                        click_lat, click_lon = coords
                    except Exception:
                        # fallback: object with latitude/longitude attributes
                        try:
                            click_lat = coords.latitude
                            click_lon = coords.longitude
                        except Exception:
                            return
                    # update via class helper
                    self._move_marker(click_lat, click_lon)

                # register the click handler
                self.map_widget.add_left_click_map_command(handle_map_click)

            except Exception as e:
                print("Map init error:", e)

        # schedule setup on main thread
        self.root.after(0, setup_map)
        self.root.after(0, self._update_loc_display)

    def _move_marker(self, lat, lon):
        """Move map marker to clicked coordinates and update display."""
        try:
            self.latitude = float(lat)
            self.longitude = float(lon)
        except Exception:
            return

        # update UI label
        self._update_loc_display()

        # move or create marker on the map
        try:
            if self.map_marker:
                # tkintermapview marker has set_position method
                self.map_marker.set_position(self.latitude, self.longitude)
            else:
                self.map_marker = self.map_widget.set_marker(self.latitude, self.longitude, text="Manual Location")
            # center map on new position
            self.map_widget.set_position(self.latitude, self.longitude)
        except Exception as e:
            print("Marker move error:", e)

    def _update_loc_display(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.loc_display.config(text=f"Location: {self.latitude:.6f}, {self.longitude:.6f}    Timestamp: {ts}")

    # ----------------- Camera controls -----------------
    def start_camera(self):
        if self.camera_running:
            return
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        except Exception:
            self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Unable to access camera (index 0).")
            return
        self.camera_running = True
        self._video_loop()

    def _video_loop(self):
        if not (self.camera_running and self.cap):
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            # draw rectangles
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dets = detector(rgb)
                for d in dets:
                    x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except Exception:
                pass
            # scale to fit label
            h, w = frame.shape[:2]
            lbl_w = self.video_label.winfo_width() or 560
            lbl_h = self.video_label.winfo_height() or 380
            if lbl_w <= 1 or lbl_h <= 1:
                # widget not yet realized; schedule after short delay
                self.root.after(50, self._video_loop)
                return
            scale = min(lbl_w / w, lbl_h / h)
            if scale < 1:
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        self.root.after(30, self._video_loop)

    def stop_camera(self):
        self.camera_running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.video_label.config(image="", text="Camera stopped.", bg="#DDDDDD")

    # ----------------- Capture & Save -----------------
    def capture_and_save(self):
        # collect inputs
        data = {}
        for key, widget in self.widgets.items():
            try:
                if isinstance(widget, ttk.Combobox):
                    data[key] = widget.get().strip()
                elif DateEntry is not None and isinstance(widget, DateEntry):
                    data[key] = widget.get()
                else:
                    data[key] = widget.get().strip()
            except Exception:
                data[key] = ""

        # required fields
        required = ["Staff Name", "Project Name", "State", "District", "Designation", "Gender", "Email ID", "Contact No", "Employee Since (Date)"]
        for r in required:
            if not data.get(r):
                messagebox.showwarning("Missing Field", f"Please fill/select '{r}'.")
                return

        # validations
        phone = data.get("Contact No", "")
        if not (phone.isdigit() and len(phone) == 10):
            messagebox.showerror("Invalid Contact", "Contact No must be exactly 10 digits.")
            return
        if not is_valid_email(data.get("Email ID", "")):
            messagebox.showerror("Invalid Email", "Please enter a valid Email ID.")
            return

        # camera
        if not (self.camera_running and self.cap):
            messagebox.showerror("Camera", "Start camera before capturing.")
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Capture Error", "Failed to capture frame.")
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = detector(rgb)
        if len(dets) == 0:
            messagebox.showwarning("No Face", "No face detected in the captured frame.")
            return

        # compute embedding
        shape = sp(rgb, dets[0])
        embedding = np.array(facerec.compute_face_descriptor(rgb, shape))

        # geo quick attempt
        #lat, lon = get_geo_coords_with_retry(retries=1, timeout=4)
        lat, lon = get_geo_coords_with_retry_google(GOOGLE_API_KEY, retries=1, timeout=4)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # save image
        safe_name = re.sub(r"[^\w\d_-]", "_", data["Staff Name"])
        safe_proj = re.sub(r"[^\w\d_-]", "_", data["Project Name"])
        fname = f"{safe_name}_{safe_proj}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        img_path = os.path.join(FACES_DIR, fname)
        cv2.imwrite(img_path, frame)

        # CSV row
        row = {
            "Staff Name": data["Staff Name"],
            "Project Name": data["Project Name"],
            "State": data["State"],
            "District": data["District"],
            "Designation": data["Designation"],
            "Gender": data["Gender"],
            "Email ID": data["Email ID"],
            "Contact No": data["Contact No"],
            "Employee Since (DD-MM-YYYY)": data["Employee Since (Date)"],
            "Latitude": float(lat),
            "Longitude": float(lon),
            "Timestamp": timestamp,
            "Image Path": img_path
        }
        for i in range(128):
            row[f"e{i}"] = float(embedding[i])

        # append to CSV & Excel
        try:
            df = pd.read_csv(CSV_FILE)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(CSV_FILE, index=False)
            df.to_excel(EXCEL_FILE, index=False, engine="openpyxl")
        except Exception as e:
            messagebox.showerror("CSV/Excel Error", f"Failed to write data: {e}")
            return

        # update display & map on main thread
        self.latitude = float(lat)
        self.longitude = float(lon)
        self.root.after(0, self._update_loc_display)
        def update_map_marker():
            try:
                if self.map_widget:
                    if self.map_marker:
                        try:
                            self.map_marker.delete()
                        except Exception:
                            pass
                    self.map_marker = self.map_widget.set_marker(self.latitude, self.longitude, text="Registered Here")
                    self.map_widget.set_position(self.latitude, self.longitude)
            except Exception:
                pass
        self.root.after(0, update_map_marker)

        messagebox.showinfo("Success", f"Face registered for {data['Staff Name']}")

    # ----------------- Export -----------------
    def export_to_excel(self):
        try:
            df = pd.read_csv(CSV_FILE)
            df.to_excel(EXCEL_FILE, index=False, engine="openpyxl")
            messagebox.showinfo("Exported", f"Excel exported: {EXCEL_FILE}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    # ----------------- Exit & cleanup -----------------
    def on_exit(self):
        self.stop_camera()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

# ------------------- Run -------------------
if __name__ == "__main__":

    def start_app():
        app = FaceRegisterApp()
        app.run()

    # show login window first
    login_root = tk.Tk()
    LoginWindow(login_root, start_app)
    login_root.mainloop()

