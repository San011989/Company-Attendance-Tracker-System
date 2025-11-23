import os
import glob
import sqlite3
import pandas as pd
from datetime import datetime

# ---------------- Paths ----------------
BASE_DIR = r"C:\Users\HP\Company_Tracker_System"
REGISTERED_CSV = os.path.join(BASE_DIR, "registered_staff_full.csv")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance_exports")
LOGOUT_DIR = os.path.join(BASE_DIR, "logout_exports")
DB_PATH = os.path.join(BASE_DIR, "database", "attendance.db")

os.makedirs(os.path.join(BASE_DIR, "database"), exist_ok=True)

# ---------------- Database connection ----------------
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# ---------------- Create tables ----------------
cur.executescript("""
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS attendance;
DROP TABLE IF EXISTS logout;
DROP TABLE IF EXISTS work_summary;

CREATE TABLE employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    project_name TEXT,
    designation TEXT,
    gender TEXT,
    email TEXT,
    contact TEXT,
    employee_since TEXT,
    state TEXT,
    district TEXT
);

CREATE TABLE attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    project_name TEXT,
    designation TEXT,
    gender TEXT,
    email TEXT,
    contact TEXT,
    employee_since TEXT,
    latitude REAL,
    longitude REAL,
    city TEXT,
    state TEXT,
    timestamp TEXT,
    image_path TEXT,
    status TEXT
);

CREATE TABLE logout (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    project_name TEXT,
    designation TEXT,
    gender TEXT,
    email TEXT,
    contact TEXT,
    employee_since TEXT,
    latitude REAL,
    longitude REAL,
    city TEXT,
    state TEXT,
    timestamp TEXT,
    image_path TEXT,
    status TEXT
);

CREATE TABLE work_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    date TEXT,
    checkin_time TEXT,
    checkout_time TEXT,
    total_hours REAL,
    status_flag TEXT
);
""")

conn.commit()

# ---------------- Employees Table ----------------
if os.path.exists(REGISTERED_CSV):
    df_emp = pd.read_csv(REGISTERED_CSV)
    emp_cols = ["Staff Name", "Project Name", "Designation", "Gender", "Email ID", "Contact No",
                "Employee Since (DD-MM-YYYY)", "State", "District"]
    df_emp = df_emp[[c for c in emp_cols if c in df_emp.columns]]
    df_emp.columns = ["name", "project_name", "designation", "gender", "email", "contact",
                      "employee_since", "state", "district"]
    df_emp.to_sql("employees", conn, if_exists="append", index=False)
    print(f"✅ Loaded {len(df_emp)} employees")

# ---------------- Helper function to normalize columns ----------------
def normalize_columns(df):
    """Standardize column names for attendance/logout files"""
    df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]
    rename_map = {
        "Name": "name",
        "Project_Name": "project_name",
        "Designation": "designation",
        "Gender": "gender",
        "Email_ID": "email",
        "Email": "email",
        "EmailId": "email",
        "Contact_No": "contact",
        "Contact": "contact",
        "Employee_Since": "employee_since",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "City": "city",
        "State": "state",
        "Timestamp": "timestamp",
        "ImagePath": "image_path",
        "Status": "status"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df

# ---------------- Attendance Table ----------------
attendance_files = glob.glob(os.path.join(ATTENDANCE_DIR, "attendance_*.xlsx"))
all_att = []

for file in attendance_files:
    try:
        df = pd.read_excel(file)
        df = normalize_columns(df)
        all_att.append(df)
        print(f"Loaded: {os.path.basename(file)} ({len(df)} rows)")
    except Exception as e:
        print(f"⚠️ Could not read {file}: {e}")

if all_att:
    df_all_att = pd.concat(all_att, ignore_index=True)
    keep_cols = [
        "name", "project_name", "designation", "gender", "email", "contact", "employee_since",
        "latitude", "longitude", "city", "state", "timestamp", "image_path", "status"
    ]
    df_all_att = df_all_att[[c for c in keep_cols if c in df_all_att.columns]]
    df_all_att.to_sql("attendance", conn, if_exists="append", index=False)
    print(f"✅ Total attendance rows inserted: {len(df_all_att)}")
else:
    print("⚠️ No attendance Excel files found.")

# ---------------- Logout Table (optional) ----------------
logout_files = glob.glob(os.path.join(LOGOUT_DIR, "logout_*.xlsx"))
all_out = []

for file in logout_files:
    try:
        df = pd.read_excel(file)
        df = normalize_columns(df)
        all_out.append(df)
        print(f"Loaded logout: {os.path.basename(file)} ({len(df)} rows)")
    except Exception as e:
        print(f"⚠️ Could not read {file}: {e}")

if all_out:
    df_all_out = pd.concat(all_out, ignore_index=True)
    keep_cols = [
        "name", "project_name", "designation", "gender", "email", "contact", "employee_since",
        "latitude", "longitude", "city", "state", "timestamp", "image_path", "status"
    ]
    df_all_out = df_all_out[[c for c in keep_cols if c in df_all_out.columns]]
    df_all_out.to_sql("logout", conn, if_exists="append", index=False)
    print(f"✅ Total logout rows inserted: {len(df_all_out)}")
else:
    print("ℹ️ No logout Excel files found (this is fine for now).")

# ---------------- Work Summary (8-hour rule) ----------------
query = """
SELECT a.name, 
       date(a.timestamp) AS date,
       MIN(a.timestamp) AS checkin_time,
       MAX(l.timestamp) AS checkout_time
FROM attendance a
LEFT JOIN logout l ON a.name = l.name AND date(a.timestamp) = date(l.timestamp)
GROUP BY a.name, date(a.timestamp)
"""
try:
    df_summary = pd.read_sql_query(query, conn)
    df_summary["checkin_time"] = pd.to_datetime(df_summary["checkin_time"])
    df_summary["checkout_time"] = pd.to_datetime(df_summary["checkout_time"])
    df_summary["total_hours"] = (df_summary["checkout_time"] - df_summary["checkin_time"]).dt.total_seconds() / 3600
    df_summary["status_flag"] = df_summary["total_hours"].apply(
        lambda h: "Green" if h >= 8 else "Red"
    )
    df_summary.to_sql("work_summary", conn, if_exists="append", index=False)
    print(f"✅ Work summary computed for {len(df_summary)} records")
except Exception as e:
    print(f"⚠️ Work summary skipped (need logout data): {e}")

conn.commit()
conn.close()

print("\n✅ Database refresh complete: database/attendance.db")
