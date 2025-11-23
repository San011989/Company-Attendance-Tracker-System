# supervisor_dashboard.py
import os
import sqlite3
from io import BytesIO
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.utils import simpleSplit

# ---------------- CONFIG ----------------
BASE_DIR = r"C:\Users\HP\Company_Tracker_System"
DB_PATH = os.path.join(BASE_DIR, "database", "attendance.db")
PDF_EXPORT_DIR = os.path.join(BASE_DIR, "reports")
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")

os.makedirs(PDF_EXPORT_DIR, exist_ok=True)

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Supervisor Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(to bottom right, #f7fbff, #ffffff); }
    .card { background: white; padding: 16px; border-radius: 12px; box-shadow: 0 6px 18px rgba(30,60,120,0.06); }
    .header { color:#1f77b4; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- AUTH (simple supervisor login) ----------------
if "supervisor_logged_in" not in st.session_state:
    st.session_state.supervisor_logged_in = False

def supervisor_login_ui():
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=160)
        st.markdown("<h2 class='header'>Supervisor Login</h2>", unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("🔐 Login"):
            if username == "supervisor" and password == "12345":
                st.session_state.supervisor_logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")
        st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.supervisor_logged_in:
    supervisor_login_ui()
    st.stop()

# ---------------- HELPERS ----------------
def read_table(query):
    """Safely read SQL into DataFrame using context manager (avoids closed DB errors)."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"Database read error: {e}")
        return pd.DataFrame()

def fetch_distincts():
    """Fetch distinct project, state, and district/city values from attendance table only."""
    project_names, states, districts_all = [], [], []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT project_name FROM attendance WHERE project_name IS NOT NULL AND project_name != ''")
            project_names = sorted({(r[0] or '').strip().title() for r in cur.fetchall()})

            cur.execute("SELECT DISTINCT state FROM attendance WHERE state IS NOT NULL AND state != ''")
            states = sorted({(r[0] or '').strip().title() for r in cur.fetchall()})

            cur.execute("SELECT DISTINCT city FROM attendance WHERE city IS NOT NULL AND city != ''")
            districts_all = sorted({(r[0] or '').strip().title() for r in cur.fetchall()})
    except Exception as e:
        st.warning(f"Could not load filter values: {e}")
    return project_names, states, districts_all


def ensure_datetime_col(df, col_candidates=("timestamp","time","datetime","login_time")):
    if df is None or df.empty:
        return df
    df = df.copy()
    found = None
    for c in df.columns:
        if c.lower() in col_candidates:
            found = c
            break
    if found:
        df["timestamp"] = pd.to_datetime(df[found], errors="coerce")
    else:
        # if timestamp already exists leave it; else create NaT
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.NaT
    # derive date/month/year/hour, weekday
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["month"] = pd.to_datetime(df["timestamp"]).dt.to_period("M").astype(str)
    df["year"] = pd.to_datetime(df["timestamp"]).dt.year
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["weekday"] = pd.to_datetime(df["timestamp"]).dt.day_name()
    return df

def generate_pdf_from_df(df, title, min_col_width=50, font_name="Helvetica", font_size=8):
    """Generates a landscape PDF for a single dataframe and returns BytesIO.
    Dynamically computes column widths so content doesn't overflow the page."""
    pdf_buffer = BytesIO()
    left_margin = right_margin = top_margin = bottom_margin = 18
    doc = SimpleDocTemplate(pdf_buffer, pagesize=landscape(A4),
                            leftMargin=left_margin, rightMargin=right_margin,
                            topMargin=top_margin, bottomMargin=bottom_margin)
    styles = getSampleStyleSheet()
    elements = []

    # logo + title
    if os.path.exists(LOGO_PATH):
        try:
            elements.append(RLImage(LOGO_PATH, width=1.8*inch, height=1*inch))
        except Exception:
            pass
    elements.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    if df is None or df.empty:
        elements.append(Paragraph("No records found for the selected filters.", styles["Normal"]))
        doc.build(elements)
        pdf_buffer.seek(0)
        return pdf_buffer

    # choose display columns
    display_cols = list(df.columns)
    df_display = df[display_cols].astype(str).fillna("")

    # compute available page width
    page_w = landscape(A4)[0] - left_margin - right_margin

    # approximate characters per column (header or longest cell)
    max_chars = []
    for col in display_cols:
        longest_cell = df_display[col].map(len).max() if not df_display.empty else 0
        max_chars.append(max(len(col), int(longest_cell)))

    total_chars = sum(max_chars) if sum(max_chars) > 0 else len(display_cols)
    col_widths = []
    for chars in max_chars:
        w = max(min_col_width, (chars / total_chars) * page_w)
        col_widths.append(w)

    # if total exceeds page width due to min_col_width, scale down proportionally
    total_w = sum(col_widths)
    if total_w > page_w:
        scale = page_w / total_w
        col_widths = [w * scale for w in col_widths]

    # prepare wrapped rows
    wrapped_data = []
    for row in df_display.itertuples(index=False):
        wrapped_row = []
        for i, cell in enumerate(row):
            approx_chars = max(20, int(col_widths[i] * 0.5))  # conservative approx
            parts = simpleSplit(str(cell), font_name, font_size, approx_chars)
            wrapped_row.append("\n".join(parts))
        wrapped_data.append(wrapped_row)

    data = [display_cols] + wrapped_data
    table = Table(data, repeatRows=1, colWidths=col_widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2E86C1")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), font_size),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey])
    ]))
    elements.append(table)
    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer

def generate_full_report_pdf(df_att, df_log, df_work):
    """Generate a combined PDF with Attendance, Logout and Working Hours sections."""
    pdf_buffer = BytesIO()
    left_margin = right_margin = top_margin = bottom_margin = 18
    doc = SimpleDocTemplate(pdf_buffer, pagesize=landscape(A4),
                            leftMargin=left_margin, rightMargin=right_margin,
                            topMargin=top_margin, bottomMargin=bottom_margin)
    styles = getSampleStyleSheet()
    elements = []

    if os.path.exists(LOGO_PATH):
        try:
            elements.append(RLImage(LOGO_PATH, width=1.8*inch, height=1*inch))
        except Exception:
            pass
    elements.append(Paragraph("<b>Full Map Data Report</b>", styles["Heading1"]))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # Helper to add a dataframe as a table section
    def add_section(title, df):
        elements.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        if df is None or df.empty:
            elements.append(Paragraph("No records found for the selected filters.", styles["Normal"]))
            elements.append(PageBreak())
            return

        cols = [c for c in df.columns if c.lower() not in ["image_path", "imagepath"]]
        df_display = df[cols].astype(str).fillna("")
        page_w = landscape(A4)[0] - left_margin - right_margin

        max_chars = []
        for col in cols:
            longest_cell = df_display[col].map(len).max() if not df_display.empty else 0
            max_chars.append(max(len(col), int(longest_cell)))
        total_chars = sum(max_chars) if sum(max_chars) > 0 else len(cols)
        col_widths = [max(50, (chars / total_chars) * page_w) for chars in max_chars]
        total_w = sum(col_widths)
        if total_w > page_w:
            scale = page_w / total_w
            col_widths = [w * scale for w in col_widths]

        wrapped_data = []
        for row in df_display.itertuples(index=False):
            wrapped_row = []
            for i, cell in enumerate(row):
                approx_chars = max(20, int(col_widths[i] * 0.5))
                parts = simpleSplit(str(cell), "Helvetica", 8, approx_chars)
                wrapped_row.append("\n".join(parts))
            wrapped_data.append(wrapped_row)

        data = [cols] + wrapped_data
        table = Table(data, repeatRows=1, colWidths=col_widths)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2E86C1")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey])
        ]))
        elements.append(table)
        elements.append(PageBreak())

    add_section("Attendance Records", df_att)
    add_section("Logout Records", df_log)
    add_section("Working Hours Summary", df_work)

    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer

# ---------------- PRECHECKS ----------------
if not os.path.exists(DB_PATH):
    st.error(f"Database not found at: {DB_PATH}")
    st.stop()

# ---------------- LOAD TABLES ----------------
# Read raw data (attendance, logout, employees). Read-only operations only.
try:
    df_att_raw = read_table("SELECT * FROM attendance")
except:
    df_att_raw = pd.DataFrame()
try:
    df_log_raw = read_table("SELECT * FROM logout")
except:
    df_log_raw = pd.DataFrame()
try:
    df_emp_raw = read_table("SELECT * FROM employees")
except:
    df_emp_raw = pd.DataFrame()

# Normalize column names to stripped strings
def normalize_cols(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

df_att_raw = normalize_cols(df_att_raw)
df_log_raw = normalize_cols(df_log_raw)
df_emp_raw = normalize_cols(df_emp_raw)

# Standardize text fields to Title case where appropriate
for df in [df_emp_raw, df_att_raw, df_log_raw]:
    if df is None or df.empty:
        continue
    for col in df.columns:
        if col.lower() in ["project_name","project name","state","district","city","name"]:
            df[col] = df[col].astype(str).str.strip().str.title()

# Ensure timestamp/datetime columns exist and are parsed
df_att = ensure_datetime_col(df_att_raw)
df_log = ensure_datetime_col(df_log_raw)
df_emp = df_emp_raw.copy().fillna("")

# ---------------- MERGE EMPLOYEE METADATA (PREFER id THEN name) ----------------
# Build list of emp columns available
if not df_emp.empty:
    emp_cols_all = [c for c in ["id","name","project_name","state","district"] if c in df_emp.columns]
else:
    emp_cols_all = []

# Merge into attendance using id if possible else name
if not df_att.empty and not df_emp.empty:
    if "id" in df_att.columns and "id" in df_emp.columns:
        try:
            df_att = df_att.merge(df_emp[emp_cols_all], on="id", how="left", suffixes=("","_emp"))
        except Exception:
            pass
    elif "name" in df_att.columns and "name" in df_emp.columns:
        try:
            df_att = df_att.merge(df_emp[emp_cols_all], on="name", how="left", suffixes=("","_emp"))
        except Exception:
            pass

# Merge into logout using id if possible else name
if not df_log.empty and not df_emp.empty:
    if "id" in df_log.columns and "id" in df_emp.columns:
        try:
            df_log = df_log.merge(df_emp[emp_cols_all], on="id", how="left", suffixes=("","_emp"))
        except Exception:
            pass
    elif "name" in df_log.columns and "name" in df_emp.columns:
        try:
            df_log = df_log.merge(df_emp[emp_cols_all], on="name", how="left", suffixes=("","_emp"))
        except Exception:
            pass

# Ensure district column exists in attendance and logout: employees.district preferred else attendance.city
if df_att is None:
    df_att = pd.DataFrame()
if df_log is None:
    df_log = pd.DataFrame()

if "district" not in df_att.columns:
    df_att["district"] = ""
else:
    df_att["district"] = df_att["district"].astype(str).str.strip().str.title()

if "city" in df_att.columns:
    df_att["city"] = df_att["city"].astype(str).str.strip().str.title()
    df_att.loc[df_att["district"].isnull() | (df_att["district"]==""), "district"] = df_att.loc[df_att["district"].isnull() | (df_att["district"]==""), "city"]
# --- Normalize all text columns for consistent filtering ---
for df_norm in [df_att, df_log]:
    for col in ["state", "district", "city"]:
        if col in df_norm.columns:
            df_norm[col] = df_norm[col].astype(str).str.strip().str.title()


if "district" not in df_log.columns:
    df_log["district"] = ""
else:
    df_log["district"] = df_log["district"].astype(str).str.strip().str.title()
if "city" in df_log.columns:
    df_log["city"] = df_log["city"].astype(str).str.strip().str.title()
    df_log.loc[df_log["district"].isnull() | (df_log["district"]==""), "district"] = df_log.loc[df_log["district"].isnull() | (df_log["district"]==""), "city"]

# ---------------- DYNAMIC FILTER VALUES ----------------
project_names, states, districts_all = fetch_distincts()
names = sorted(df_att["name"].dropna().unique()) if "name" in df_att.columns and not df_att.empty else []

# ---------------- SIDEBAR FILTER UI ----------------
st.sidebar.markdown("<div class='card'>", unsafe_allow_html=True)
st.sidebar.header("Filters")

filter_type = st.sidebar.radio("Time filter", ["All","Date","Month","Year"])

# Copies for filtering
df_filtered_att = df_att.copy()
df_filtered_log = df_log.copy()
df_filtered_work = pd.DataFrame()  # will be built later

# Time filters
if filter_type == "Date":
    selected_date = st.sidebar.date_input("Select Date")
elif filter_type == "Month":
    months = sorted(df_att["month"].dropna().unique()) if "month" in df_att.columns else []
    selected_month = st.sidebar.selectbox("Select Month", ["All"] + months)
elif filter_type == "Year":
    years = sorted(df_att["year"].dropna().unique()) if "year" in df_att.columns else []
    selected_year = st.sidebar.selectbox("Select Year", ["All"] + years)

# Other filters
selected_name = st.sidebar.selectbox("Name", ["All"] + names)
selected_project = st.sidebar.selectbox("Project", ["All"] + project_names)
selected_state = st.sidebar.selectbox("State", ["All"] + states)

# District list dynamic depending on selected_state
# District list dynamic depending on selected_state (from attendance table only)
if selected_state != "All":
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            # Get districts/cities only for the selected state from attendance table
            cur.execute("""
                SELECT DISTINCT TRIM(city)
                FROM attendance
                WHERE state IS NOT NULL 
                  AND TRIM(state) != ''
                  AND LOWER(TRIM(state)) = LOWER(TRIM(?))
                  AND city IS NOT NULL 
                  AND TRIM(city) != ''
            """, (selected_state,))

            filtered_districts = sorted({(r[0] or '').strip().title() for r in cur.fetchall()})
    except Exception as e:
        st.warning(f"District filter load failed: {e}")
        filtered_districts = districts_all
else:
    filtered_districts = districts_all

selected_district = st.sidebar.selectbox("District / City", ["All"] + filtered_districts)

# small info
st.sidebar.markdown("---")
st.sidebar.info("Supervisor: read-only dashboard. Exports contain only filtered data.")
st.sidebar.caption(f"Active filters → Project: {selected_project}, State: {selected_state}, District: {selected_district}")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# ---------------- APPLY FILTERS TO DATAFRAMES ----------------
def apply_filters_to_df(df):
    if df is None or df.empty:
        return df
    df2 = df.copy()
    # time filters
    if filter_type == "Date" and "date" in df2.columns:
        df2 = df2[df2["date"] == selected_date]
    if filter_type == "Month" and selected_month and selected_month != "All" and "month" in df2.columns:
        df2 = df2[df2["month"] == selected_month]
    if filter_type == "Year" and selected_year and selected_year != "All" and "year" in df2.columns:
        df2 = df2[df2["year"] == selected_year]
    # name
    if selected_name != "All" and "name" in df2.columns:
        df2 = df2[df2["name"] == selected_name]
    # project
    if selected_project != "All" and "project_name" in df2.columns:
        df2 = df2[df2["project_name"] == selected_project]
    # state filter
    if selected_state != "All" and "state" in df2.columns:
        df2 = df2[df2["state"].str.strip().str.lower() == selected_state.strip().lower()]

    # district or city filter — use attendance table’s “city” as primary reference
    # district or city filter — match either column case-insensitively
    if selected_district != "All":
        district_lower = selected_district.strip().lower()
        df2_cols = df2.columns.str.lower()

        if "district" in df2_cols and "city" in df2_cols:
            df2 = df2[
                (df2["district"].astype(str).str.strip().str.lower() == district_lower) |
                (df2["city"].astype(str).str.strip().str.lower() == district_lower)
            ]
        elif "district" in df2_cols:
            df2 = df2[df2["district"].astype(str).str.strip().str.lower() == district_lower]
        elif "city" in df2_cols:
            df2 = df2[df2["city"].astype(str).str.strip().str.lower() == district_lower]

    return df2


df_att_f = apply_filters_to_df(df_att)
df_log_f = apply_filters_to_df(df_log)

# ---------------- BUILD WORK HOURS (PREFER id THEN name) ----------------
def build_work_hours(df_att, df_log):
    import numpy as np
    import pandas as pd

    # --- Parse timestamps safely ---
    df_att["timestamp"] = pd.to_datetime(df_att["timestamp"], errors="coerce")
    df_log["timestamp"] = pd.to_datetime(df_log["timestamp"], errors="coerce")

    df_att["date"] = df_att["timestamp"].dt.date
    df_log["date"] = df_log["timestamp"].dt.date

    # --- 1️⃣ Get check-in from attendance (first timestamp per name/date) ---
    att_first = (
        df_att.sort_values("timestamp")
        .groupby(["name", "date"], as_index=False)
        .first()
        .rename(columns={"timestamp": "check_in"})
    )

    # --- 2️⃣ Get check-out from logout (last timestamp per name/date) ---
    log_last = (
        df_log.sort_values("timestamp")
        .groupby(["name", "date"], as_index=False)
        .last()
        .rename(columns={"timestamp": "check_out"})
    )

    # --- 3️⃣ Merge both datasets (outer join) ---
    dfw = pd.merge(att_first, log_last[["name", "date", "check_out"]], on=["name", "date"], how="outer")

    # --- 4️⃣ Compute total working hours (use 0 if missing check_in/check_out) ---
    dfw["check_in"] = pd.to_datetime(dfw["check_in"], errors="coerce")
    dfw["check_out"] = pd.to_datetime(dfw["check_out"], errors="coerce")

    dfw["total_hours"] = np.where(
        dfw["check_in"].notna() & dfw["check_out"].notna(),
        (dfw["check_out"] - dfw["check_in"]).dt.total_seconds() / 3600,
        0
    )
    dfw["total_hours"] = dfw["total_hours"].fillna(0).round(2)

    # --- 5️⃣ Add status column ---
    dfw["status"] = np.where(dfw["total_hours"] >= 8, "On Time", "Less Hours")

    # --- 6️⃣ Keep project_name/state/district info (if exists) ---
    if "project_name" not in dfw.columns:
        dfw["project_name"] = ""
    if "state" not in dfw.columns:
        dfw["state"] = ""
    if "city" in dfw.columns:
        dfw["district"] = dfw["city"]
    else:
        dfw["district"] = ""

    # --- 7️⃣ Fill empty values safely ---
    for col in ["check_in", "check_out", "project_name", "state", "district"]:
        dfw[col] = dfw[col].fillna("0")

    # --- 8️⃣ Final column order ---
    cols = [
        "name", "project_name", "date",
        "check_in", "check_out",
        "total_hours", "status",
        "state", "district"
    ]
    dfw = dfw[[c for c in cols if c in dfw.columns]]

    return dfw


df_work = build_work_hours(df_att_f, df_log_f)

# ---------------- DASHBOARD LAYOUT ----------------
st.markdown("<div class='card'><h1 style='color:#1f77b4'>Supervisor Dashboard</h1></div>", unsafe_allow_html=True)
st.markdown("<br/>", unsafe_allow_html=True)

# Top row metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Attendance Records", len(df_att_f))
with col2:
    st.metric("Logout Records", len(df_log_f))
with col3:
    pct = (len(df_log_f) / len(df_att_f) * 100) if len(df_att_f) else 0
    st.metric("Attendance Status (%)", f"{pct:.1f}%")
with col4:
    st.metric("Employees in View", df_att_f["name"].nunique() if "name" in df_att_f.columns else 0)

st.markdown("---")

# Two-column main content: left charts, right heatmaps & extras
left_col, right_col = st.columns([2,1])

with left_col:
    # Daily Attendance count
    st.markdown("### 📈 Daily Attendance")
    if "date" in df_att_f.columns and "name" in df_att_f.columns and not df_att_f.empty:
        daily_counts = df_att_f.groupby("date")["name"].count().reset_index().sort_values("date")
        fig = px.bar(daily_counts, x="date", y="name", labels={"name":"Count","date":"Date"},
                     title="Daily Attendance Count", template="plotly_white")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No attendance data for selected filters.")

    # Hourly Trends: check-in & check-out on same line
    st.markdown("### ⏱ Hourly Trend (Check-ins & Check-outs)")
    # prepare hourly aggregates
    hr_df = pd.DataFrame({"hour": range(0,24)})
    if not df_att_f.empty and "hour" in df_att_f.columns:
        ci = df_att_f.dropna(subset=["hour"]).groupby("hour")["name"].count().reindex(range(24), fill_value=0).reset_index()
        ci.columns = ["hour","checkin_count"]
    else:
        ci = pd.DataFrame({"hour":range(24),"checkin_count":[0]*24})
    if not df_log_f.empty and "hour" in df_log_f.columns:
        co = df_log_f.dropna(subset=["hour"]).groupby("hour")["name"].count().reindex(range(24), fill_value=0).reset_index()
        co.columns = ["hour","checkout_count"]
    else:
        co = pd.DataFrame({"hour":range(24),"checkout_count":[0]*24})
    hr_comb = pd.merge(ci, co, on="hour")
    fig2 = px.line(hr_comb, x="hour", y=["checkin_count","checkout_count"], labels={"value":"Count","hour":"Hour","variable":"Type"},
                   title="Hourly Trend (Check-ins vs Check-outs)", template="plotly_white")
    fig2.update_layout(legend=dict(title=None))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🔥 Heatmaps (Check-in / Check-out)")
    # two heatmaps side-by-side
    hm1, hm2 = st.columns(2)

    # heatmap helpers: pivot weekday x hour
    def heatmap_matrix(df_in):
        # ensure weekday order Mon..Sun and hours 0..23
        if df_in is None or df_in.empty or "weekday" not in df_in.columns or "hour" not in df_in.columns:
            # empty matrix 7x24
            weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            mat = pd.DataFrame(0, index=weekdays, columns=range(24))
            return mat
        mat = df_in.dropna(subset=["weekday","hour"]).pivot_table(index="weekday", columns="hour", values="name", aggfunc="count", fill_value=0)
        # reindex to full weekdays and hours
        weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        mat = mat.reindex(weekdays).fillna(0)
        for h in range(24):
            if h not in mat.columns:
                mat[h] = 0
        mat = mat[sorted(mat.columns)]
        return mat

    hm_checkin = heatmap_matrix(df_att_f)
    hm_checkout = heatmap_matrix(df_log_f)

    with hm1:
        st.markdown("#### Check-in Heatmap")
        fig_hm1 = px.imshow(hm_checkin, labels=dict(x="Hour", y="Weekday", color="Count"), x=list(hm_checkin.columns), y=hm_checkin.index,
                            aspect="auto", text_auto=False)
        fig_hm1.update_layout(template="plotly_white")
        st.plotly_chart(fig_hm1, use_container_width=True, key="checkin_heatmap")

    with hm2:
        st.markdown("#### Check-out Heatmap")
        fig_hm2 = px.imshow(hm_checkout, labels=dict(x="Hour", y="Weekday", color="Count"), x=list(hm_checkout.columns), y=hm_checkout.index,
                            aspect="auto", text_auto=False)
        fig_hm2.update_layout(template="plotly_white")
        st.plotly_chart(fig_hm2, use_container_width=True, key="checkout_heatmap")

with right_col:
    # Attendance vs Logout pie
    st.markdown("### 📊 Attendance vs Logout")
    st.write("")
    att_count = len(df_att_f)
    log_count = len(df_log_f)
    df_status = pd.DataFrame({"Status":["Present","Logout"], "Count":[att_count, log_count]})
    fig_pie = px.pie(df_status, values="Count", names="Status", title="Attendance vs Logout", hole=0.4, template="plotly_white")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Top active employees
    st.markdown("### 👥 Top Active Employees")
    if not df_att_f.empty and "name" in df_att_f.columns:
        top_emp = df_att_f.groupby("name")["timestamp"].count().reset_index().rename(columns={"timestamp":"att_count"}).sort_values("att_count", ascending=False).head(10)
        fig_top = px.bar(top_emp, x="name", y="att_count", title="Top 10 Active Employees", template="plotly_white")
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("No attendance data for this view.")

    # State-wise distribution
    st.markdown("### 🗺️ State-wise Attendance")
    if not df_att_f.empty and "state" in df_att_f.columns:
        state_counts = df_att_f.groupby("state")["name"].count().reset_index().sort_values("name", ascending=False)
        fig_state = px.bar(state_counts, x="state", y="name", title="State-wise Attendance", template="plotly_white")
        st.plotly_chart(fig_state, use_container_width=True)
    else:
        st.info("State distribution not available for selected filters.")

st.markdown("---")

# ---------------- WORKING HOURS EFFICIENCY (separate full-width row) ----------------
st.markdown("## 🕒 Working Hours Efficiency")

# Prepare display table with required columns and color mapping
display_cols = ["id", "name", "date", "check_in", "check_out", "total_hours", "project_name", "state", "district", "status"]
for c in display_cols:
    if c not in df_work.columns:
        df_work[c] = ""

# --- Compute status flag based on total hours ---
df_work["total_hours"] = pd.to_numeric(df_work["total_hours"], errors="coerce").fillna(0).round(2)
df_work["status"] = np.where(df_work["total_hours"] >= 8, "On Time", "Less Hours")

# --- Ensure check-in/check-out are datetime objects ---
df_work["check_in"] = pd.to_datetime(df_work.get("check_in"), errors="coerce")
df_work["check_out"] = pd.to_datetime(df_work.get("check_out"), errors="coerce")

# --- Format for display ---
df_work_display = df_work[display_cols].copy()
df_work_display["check_in"] = df_work_display["check_in"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("0")
df_work_display["check_out"] = df_work_display["check_out"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("0")
df_work_display["date"] = df_work_display["date"].astype(str).fillna("")

# --- Color styling for total_hours and status ---
def flag_color(val):
    try:
        val_num = float(val)
    except:
        return ""
    if val_num >= 8:
        return "background-color: lightgreen; color: black;"
    if 0 < val_num < 8:
        return "background-color: #ff4d4d; color: white;"
    return ""

def status_color(val):
    if val == "On Time":
        return "background-color: lightgreen; color: black;"
    elif val == "Less Hours":
        return "background-color: #ff4d4d; color: white;"
    return ""

if not df_work_display.empty:
    styled = df_work_display.style.applymap(flag_color, subset=["total_hours"])
    styled = styled.applymap(status_color, subset=["status"])
    st.markdown("<style>table {border-collapse: collapse;} td, th {padding: 6px 10px !important;}</style>", unsafe_allow_html=True)
    st.write(styled.to_html(), unsafe_allow_html=True)
else:
    st.info("No work hours data for the selected filters.")

# --- Bar chart with red/green bars ---
if not df_work_display.empty and "name" in df_work_display.columns:
    df_work_display["bar_color"] = np.where(
        df_work_display["total_hours"] >= 8, "green", "red"
    )

    fig_wh = px.bar(
        df_work_display.sort_values("total_hours", ascending=False).head(30),
        x="name",
        y="total_hours",
        color="bar_color",
        color_discrete_map={"green": "lightgreen", "red": "#ff4d4d"},
        title="Working Hours by Employee",
        template="plotly_white",
    )
    fig_wh.update_layout(showlegend=False)
    st.plotly_chart(fig_wh, use_container_width=True)

# ---------------- EXPORT (Excel & PDF) ----------------
st.markdown("---")
st.markdown("### 📥 Export Filtered Data")

# Export Attendance filtered
if not df_att_f.empty:
    excel_buf_att = BytesIO()
    with pd.ExcelWriter(excel_buf_att, engine="openpyxl") as writer:
        # limit columns for export (exclude large image_path)
        cols = [c for c in df_att_f.columns if c.lower() not in ["image_path","imagepath"]]
        df_att_f[cols].to_excel(writer, index=False, sheet_name="Attendance")
    excel_buf_att.seek(0)
    st.download_button("⬇️ Download Filtered Attendance (Excel)", data=excel_buf_att, file_name=f"Attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Single-click PDF download (no intermediate button)
    pdf_buf_att = generate_pdf_from_df(df_att_f[[c for c in df_att_f.columns if c.lower() not in ['image_path','imagepath']]], "Attendance Report")
    st.download_button("📄 Download Attendance PDF", data=pdf_buf_att, file_name=f"Attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
else:
    st.info("No attendance data to export for current filters.")

# Export Logout filtered
if not df_log_f.empty:
    excel_buf_log = BytesIO()
    with pd.ExcelWriter(excel_buf_log, engine="openpyxl") as writer:
        cols = [c for c in df_log_f.columns if c.lower() not in ["image_path","imagepath"]]
        df_log_f[cols].to_excel(writer, index=False, sheet_name="Logout")
    excel_buf_log.seek(0)
    st.download_button("⬇️ Download Filtered Logout (Excel)", data=excel_buf_log, file_name=f"Logout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    pdf_buf_log = generate_pdf_from_df(df_log_f[[c for c in df_log_f.columns if c.lower() not in ['image_path','imagepath']]], "Logout Report")
    st.download_button("📄 Download Logout PDF", data=pdf_buf_log, file_name=f"Logout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
else:
    st.info("No logout data to export for current filters.")

# Export Work Summary filtered
if not df_work_display.empty:
    excel_buf_work = BytesIO()
    with pd.ExcelWriter(excel_buf_work, engine="openpyxl") as writer:
        df_work_display.to_excel(writer, index=False, sheet_name="Work_Hours")
    excel_buf_work.seek(0)
    st.download_button("⬇️ Download Working Hours (Excel)", data=excel_buf_work, file_name=f"WorkingHours_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    pdf_buf_work = generate_pdf_from_df(df_work_display, "Working Hours Efficiency")
    st.download_button("📄 Download Working Hours PDF", data=pdf_buf_work, file_name=f"WorkingHours_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
else:
    st.info("No working hours data to export for current filters.")

# Combined Full Report PDF (Attendance + Logout + Working Hours)
full_pdf_buf = generate_full_report_pdf(df_att_f, df_log_f, df_work_display)
st.download_button("🧾 Download Full Map Data Report (PDF)", data=full_pdf_buf, file_name=f"FullReport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")

st.markdown("---")
st.caption("© 2025 Company Attendance AI — Supervisor Dashboard (read-only)")
