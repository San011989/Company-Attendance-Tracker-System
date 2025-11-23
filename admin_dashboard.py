# admin_dashboard.py (replace your current file with this)
import os
import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import schedule
import time
import threading
import subprocess
from reportlab.lib.utils import simpleSplit

# ---------------- CONFIG ----------------
BASE_DIR = r"C:\Users\HP\Company_Tracker_System"
DB_PATH = os.path.join(BASE_DIR, "database", "attendance.db")
PDF_EXPORT_DIR = os.path.join(BASE_DIR, "reports")
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
INIT_SCRIPT = os.path.join(BASE_DIR, "init_database.py")

os.makedirs(PDF_EXPORT_DIR, exist_ok=True)

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Admin Dashboard", layout="wide")
st.title("🏢 Company Attendance — Admin Dashboard")

# ---------------- AUTO DB REFRESH SCHEDULER ----------------
def run_db_refresh():
    try:
        subprocess.run(["python", INIT_SCRIPT], check=True)
        print(f"Database refreshed at {datetime.now()}")
    except Exception as e:
        print(f"DB refresh error: {e}")

def schedule_refresh():
    schedule.every().day.at("00:05").do(run_db_refresh)
    while True:
        schedule.run_pending()
        time.sleep(60)

if "scheduler_started" not in st.session_state:
    threading.Thread(target=schedule_refresh, daemon=True).start()
    st.session_state.scheduler_started = True

# ---------------- LOGIN ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    with st.form("login_form"):
        st.subheader("🔐 Admin Login")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            if user == "admin" and pwd == "12345":
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.stop()

# ---------------- LOGOUT BUTTON ----------------
col1, col2 = st.columns([9, 1])
with col2:
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
        except Exception:
            pass
        st.success("Logged out successfully.")
        st.rerun()

# ---------------- DATABASE LOAD ----------------
@st.cache_resource
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_data(ttl=60, show_spinner=False)
def load_data():
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)

        # attendance is always expected
        df_att = pd.read_sql_query("SELECT * FROM attendance", conn)

        # logout may or may not exist
        try:
            df_log = pd.read_sql_query("SELECT * FROM logout", conn)
        except Exception:
            df_log = pd.DataFrame()

        # employees may or may not exist
        try:
            df_emp = pd.read_sql_query("SELECT * FROM employees", conn)
        except Exception:
            df_emp = pd.DataFrame()

        # normalize employee table columns
        if not df_emp.empty:
            df_emp.columns = df_emp.columns.str.strip().str.lower()
            for col in ["project_name", "state", "district", "name"]:
                if col in df_emp.columns:
                    df_emp[col] = df_emp[col].astype(str).str.strip().str.title()
            df_emp = df_emp.fillna("")

        # normalize attendance & logout timestamp/date fields
        for df in [df_att, df_log]:
            if df is None or df.empty:
                continue
            df.columns = df.columns.str.strip().str.lower()
            # find a time-like column
            time_col = None
            for col in df.columns:
                if col.lower() in ["timestamp", "time", "datetime", "login_time"]:
                    time_col = col
                    break
            if time_col:
                df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
                df["date"] = df["timestamp"].dt.date
                df["month"] = df["timestamp"].dt.to_period("M").astype(str)
                df["year"] = df["timestamp"].dt.year
            else:
                df["timestamp"], df["date"], df["month"], df["year"] = None, None, None, None

        conn.close()
        return df_att, df_log, df_emp
    except Exception as e:
        st.error(f"⚠️ Database read error: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_att, df_log, df_emp = load_data()

# ensure not None
df_att = df_att if df_att is not None else pd.DataFrame()
df_log = df_log if df_log is not None else pd.DataFrame()
df_emp = df_emp if df_emp is not None else pd.DataFrame()

# normalize columns for consistent access
def normalize_text_columns(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    return df

df_emp = normalize_text_columns(df_emp, ["project_name", "state", "district", "name"])
df_att = normalize_text_columns(df_att, ["project_name", "state", "city", "name"])
df_log = normalize_text_columns(df_log, ["project_name", "state", "city", "name"])

# create a district column in attendance & logout derived from city if district missing
if "district" not in df_att.columns:
    df_att["district"] = ""
else:
    df_att["district"] = df_att["district"].astype(str).str.strip().str.title()

if "city" in df_att.columns:
    df_att["city"] = df_att["city"].astype(str).str.strip().str.title()
    df_att.loc[df_att["district"].isnull() | (df_att["district"] == ""), "district"] = df_att.loc[df_att["district"].isnull() | (df_att["district"] == ""), "city"]

if "district" not in df_log.columns:
    df_log["district"] = ""
else:
    df_log["district"] = df_log["district"].astype(str).str.strip().str.title()

if "city" in df_log.columns:
    df_log["city"] = df_log["city"].astype(str).str.strip().str.title()
    df_log.loc[df_log["district"].isnull() | (df_log["district"] == ""), "district"] = df_log.loc[df_log["district"].isnull() | (df_log["district"] == ""), "city"]

# Normalize name fields for merge/filter
if "name" in df_emp.columns:
    df_emp["name"] = df_emp["name"].astype(str).str.strip().str.title()
if "name" in df_att.columns:
    df_att["name"] = df_att["name"].astype(str).str.strip().str.title()
if "name" in df_log.columns:
    df_log["name"] = df_log["name"].astype(str).str.strip().str.title()

# ---------------- DYNAMIC FILTERS (robust union) ----------------
project_names = []
states = []
districts_all = []

try:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # gather project names from employees and attendance
    p_emp = []
    p_att = []
    try:
        cur.execute("SELECT DISTINCT project_name FROM employees WHERE project_name IS NOT NULL AND project_name != ''")
        p_emp = [r[0] for r in cur.fetchall() if r[0]]
    except Exception:
        pass
    try:
        cur.execute("SELECT DISTINCT project_name FROM attendance WHERE project_name IS NOT NULL AND project_name != ''")
        p_att = [r[0] for r in cur.fetchall() if r[0]]
    except Exception:
        pass
    project_names = sorted({(p or "").strip().title() for p in (p_emp + p_att) if p})

    # gather states from employees and attendance
    s_emp = []
    s_att = []
    try:
        cur.execute("SELECT DISTINCT state FROM employees WHERE state IS NOT NULL AND state != ''")
        s_emp = [r[0] for r in cur.fetchall() if r[0]]
    except Exception:
        pass
    try:
        cur.execute("SELECT DISTINCT state FROM attendance WHERE state IS NOT NULL AND state != ''")
        s_att = [r[0] for r in cur.fetchall() if r[0]]
    except Exception:
        pass
    states = sorted({(s or "").strip().title() for s in (s_emp + s_att) if s})

    # districts: employees.district + attendance.city
    d_emp = []
    d_att_city = []
    try:
        cur.execute("SELECT DISTINCT district FROM employees WHERE district IS NOT NULL AND district != ''")
        d_emp = [r[0] for r in cur.fetchall() if r[0]]
    except Exception:
        pass
    try:
        cur.execute("SELECT DISTINCT city FROM attendance WHERE city IS NOT NULL AND city != ''")
        d_att_city = [r[0] for r in cur.fetchall() if r[0]]
    except Exception:
        pass
    districts_all = sorted({(d or "").strip().title() for d in (d_emp + d_att_city) if d})

    conn.close()
except Exception as e:
    st.warning(f"⚠️ Could not load dynamic filters from database: {e}")
    project_names, states, districts_all = [], [], []

project_names = project_names if project_names else []
states = states if states else []
districts_all = districts_all if districts_all else []

# ---------------- MERGE employee info into attendance & logout (so project/state/district available) ----------------
emp_info = df_emp.copy()
if not df_att.empty and not emp_info.empty and "name" in df_att.columns and "name" in emp_info.columns:
    merge_cols = [c for c in emp_info.columns if c in ["name", "project_name", "state", "district"]]
    if merge_cols:
        df_att = df_att.merge(emp_info[merge_cols], on="name", how="left", suffixes=("", "_emp"))
        # after merge, prefer merged values if present
        for col in ["project_name", "state", "district"]:
            if col in df_att.columns and f"{col}_emp" in df_att.columns:
                df_att[col] = df_att[col].fillna(df_att[f"{col}_emp"])
                df_att.drop(columns=[f"{col}_emp"], inplace=True, errors="ignore")

if not df_log.empty and not emp_info.empty and "name" in df_log.columns and "name" in emp_info.columns:
    merge_cols = [c for c in emp_info.columns if c in ["name", "project_name", "state", "district"]]
    if merge_cols:
        df_log = df_log.merge(emp_info[merge_cols], on="name", how="left", suffixes=("", "_emp"))
        for col in ["project_name", "state", "district"]:
            if col in df_log.columns and f"{col}_emp" in df_log.columns:
                df_log[col] = df_log[col].fillna(df_log[f"{col}_emp"])
                df_log.drop(columns=[f"{col}_emp"], inplace=True, errors="ignore")

# ensure district exists everywhere
for df in [df_att, df_log]:
    if "district" not in df.columns:
        df["district"] = ""
    df["district"] = df["district"].astype(str).str.strip().str.title()

# ---------------- SIDEBAR FILTER UI ----------------
st.sidebar.header("📅 Filters")
filter_type = st.sidebar.radio("Filter by", ["All", "Date", "Month", "Year"])

# copy frames to filter
df_filtered_att = df_att.copy() if df_att is not None else pd.DataFrame()
df_filtered_log = df_log.copy() if df_log is not None else pd.DataFrame()
df_filtered_summary = df_att.copy() if df_att is not None else pd.DataFrame()
df_filtered_edit = df_att.copy() if df_att is not None else pd.DataFrame()

# --- Time filters (unchanged logic) ---
if filter_type == "Date":
    selected_date = st.sidebar.date_input("Select Date")
    for df in [df_filtered_att, df_filtered_log, df_filtered_summary, df_filtered_edit]:
        if "date" in df.columns:
            df.dropna(subset=["date"], inplace=True)
            try:
                df.query("date == @selected_date", inplace=True)
            except Exception:
                df[:] = df[df["date"] == selected_date]
elif filter_type == "Month":
    months = sorted(df_att["month"].dropna().unique()) if "month" in df_att.columns else []
    selected_month = st.sidebar.selectbox("Select Month", months) if months else None
    if selected_month:
        for df in [df_filtered_att, df_filtered_log, df_filtered_summary, df_filtered_edit]:
            if "month" in df.columns:
                df.dropna(subset=["month"], inplace=True)
                try:
                    df.query("month == @selected_month", inplace=True)
                except Exception:
                    df[:] = df[df["month"] == selected_month]
elif filter_type == "Year":
    years = sorted(df_att["year"].dropna().unique()) if "year" in df_att.columns else []
    selected_year = st.sidebar.selectbox("Select Year", years) if years else None
    if selected_year:
        for df in [df_filtered_att, df_filtered_log, df_filtered_summary, df_filtered_edit]:
            if "year" in df.columns:
                df.dropna(subset=["year"], inplace=True)
                try:
                    df.query("year == @selected_year", inplace=True)
                except Exception:
                    df[:] = df[df["year"] == selected_year]

# --- Additional filters (Project / State / District / Name) ---
# Name dropdown still from attendance (unchanged)
names = sorted(df_att["name"].dropna().unique()) if not df_att.empty and "name" in df_att.columns else []
selected_name = st.sidebar.selectbox("Filter by Name", ["All"] + names)

# Project & State dropdowns from union lists
selected_project = st.sidebar.selectbox("Filter by Project", ["All"] + project_names)
selected_state = st.sidebar.selectbox("Filter by State", ["All"] + states)

# Build filtered list of districts depending on state selection (robust union)
filtered_districts = []
if selected_state != "All":
    # use employees table first for districts matching state (case-insensitive)
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        q = "SELECT DISTINCT district FROM employees WHERE lower(state)=lower(?) AND district IS NOT NULL AND district != ''"
        cur.execute(q, (selected_state,))
        d_emp_state = [r[0] for r in cur.fetchall() if r[0]]
        # also include attendance city values (attendance may not have state info)
        cur.execute("SELECT DISTINCT city FROM attendance WHERE city IS NOT NULL AND city != ''")
        d_att_city = [r[0] for r in cur.fetchall() if r[0]]
        conn.close()
        filtered_districts = sorted({(d or "").strip().title() for d in (d_emp_state + d_att_city) if d})
    except Exception:
        # fallback to in-memory emp_info filtering
        if not emp_info.empty and "state" in emp_info.columns:
            filtered_districts = sorted(emp_info[emp_info["state"].str.lower() == selected_state.lower()]["district"].dropna().unique())
        else:
            filtered_districts = districts_all
else:
    filtered_districts = districts_all

selected_district = st.sidebar.selectbox("Filter by District/City", ["All"] + filtered_districts)

# ---------------- APPLY FILTERS (robust case-insensitive masking) ----------------
def apply_text_filter(df, col, selected_value):
    """Apply a case-insensitive equality filter on df[col] for selected_value.
       Returns filtered df (works even if column missing)."""
    if selected_value == "All" or col not in df.columns:
        return df
    mask = df[col].astype(str).fillna("").str.strip().str.lower() == str(selected_value).strip().lower()
    return df[mask]

for df_name, df in [("att", df_filtered_att), ("log", df_filtered_log),
                    ("summary", df_filtered_summary), ("edit", df_filtered_edit)]:
    if df is None or df.empty:
        continue
    if selected_name != "All" and "name" in df.columns:
        df_masked = apply_text_filter(df, "name", selected_name)
        # assign back to corresponding filtered variable
        if df_name == "att":
            df_filtered_att = df_masked
            df = df_filtered_att
        elif df_name == "log":
            df_filtered_log = df_masked
            df = df_filtered_log
        elif df_name == "summary":
            df_filtered_summary = df_masked
            df = df_filtered_summary
        else:
            df_filtered_edit = df_masked
            df = df_filtered_edit

    # project
    if selected_project != "All" and "project_name" in df.columns:
        df_masked = apply_text_filter(df, "project_name", selected_project)
        if df_name == "att":
            df_filtered_att = df_masked
            df = df_filtered_att
        elif df_name == "log":
            df_filtered_log = df_masked
            df = df_filtered_log
        elif df_name == "summary":
            df_filtered_summary = df_masked
            df = df_filtered_summary
        else:
            df_filtered_edit = df_masked
            df = df_filtered_edit

    # state
    if selected_state != "All" and "state" in df.columns:
        df_masked = apply_text_filter(df, "state", selected_state)
        if df_name == "att":
            df_filtered_att = df_masked
            df = df_filtered_att
        elif df_name == "log":
            df_filtered_log = df_masked
            df = df_filtered_log
        elif df_name == "summary":
            df_filtered_summary = df_masked
            df = df_filtered_summary
        else:
            df_filtered_edit = df_masked
            df = df_filtered_edit

    # district
    if selected_district != "All" and "district" in df.columns:
        df_masked = apply_text_filter(df, "district", selected_district)
        if df_name == "att":
            df_filtered_att = df_masked
        elif df_name == "log":
            df_filtered_log = df_masked
        elif df_name == "summary":
            df_filtered_summary = df_masked
        else:
            df_filtered_edit = df_masked

# ---------------- sidebar counts ----------------
st.sidebar.info(f"Total Attendance: {len(df_filtered_att)}")
st.sidebar.info(f"Total Logout: {len(df_filtered_log)}")

# ---------------- PDF GENERATOR ----------------
def generate_pdf_report(df, title):
    pdf_path = os.path.join(PDF_EXPORT_DIR, f"{title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=landscape(A4), leftMargin=15, rightMargin=15, topMargin=15, bottomMargin=15)
    styles = getSampleStyleSheet()
    elements = []
    if os.path.exists(LOGO_PATH):
        elements.append(RLImage(LOGO_PATH, width=1.8*inch, height=1*inch))
    elements.append(Paragraph(f"<b>{title}</b>", styles["Heading1"]))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    elements.append(Spacer(1, 10))
    if df.empty:
        elements.append(Paragraph("No records found.", styles["Normal"]))
        doc.build(elements)
        return pdf_path
    display_cols = [c for c in df.columns if c.lower() not in ["imagepath", "image_path"]]
    df_display = df[display_cols].astype(str)
    wrapped_data = []
    for row in df_display.itertuples(index=False):
        wrapped_row = []
        for cell in row:
            lines = simpleSplit(str(cell), 'Helvetica', 7, 150)
            wrapped_row.append('\n'.join(lines))
        wrapped_data.append(wrapped_row)
    data = [display_cols] + wrapped_data
    page_width = landscape(A4)[0] - 60
    col_width = page_width / len(display_cols) if len(display_cols) > 0 else page_width
    col_widths = [col_width for _ in display_cols]
    table = Table(data, repeatRows=1, colWidths=col_widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E86C1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("© 2025 Company Attendance AI System", styles["Normal"]))
    doc.build(elements)
    return pdf_path

# ---------------- TABS ----------------
tab_summary, tab_att, tab_log, tab_edit = st.tabs(
    ["📈 Summary & Charts", "🧾 Attendance", "📤 Logout", "✏️ Edit Records"]
)

# ---------------- SUMMARY TAB ----------------
with tab_summary:
    st.subheader("📊 Attendance & Work Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Attendance Records", len(df_filtered_att))
        st.metric("Total Logout Records", len(df_filtered_log))

    with col2:
        if "date" in df_filtered_summary.columns and "name" in df_filtered_summary.columns:
            daily_counts = df_filtered_summary.groupby("date")["name"].count().reset_index()
        else:
            daily_counts = pd.DataFrame()
        if not daily_counts.empty:
            fig = px.bar(daily_counts, x="date", y="name", title="Daily Attendance Count")
            st.plotly_chart(fig, use_container_width=True)

    df_status = pd.DataFrame({
        "Status": ["Present", "Logout"],
        "Count": [len(df_filtered_att), len(df_filtered_log)]
    })
    fig_pie = px.pie(df_status, values="Count", names="Status", title="Attendance vs Logout Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Working Hours Efficiency (derive from work_summary if present, otherwise from attendance)
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        try:
            df_work = pd.read_sql_query("SELECT * FROM work_summary", conn)
        except Exception:
            df_work = pd.read_sql_query("SELECT * FROM attendance", conn)
        conn.close()

        if not df_work.empty:
            # normalize
            df_work.columns = df_work.columns.str.strip().str.lower()
            if "name" in df_work.columns:
                df_work["name"] = df_work["name"].astype(str).str.strip().str.title()

            # merge emp info if possible
            if not emp_info.empty and "name" in df_work.columns and "name" in emp_info.columns:
                emp_merge_cols = [c for c in ["name", "state", "district", "project_name"] if c in emp_info.columns]
                if emp_merge_cols:
                    emp_copy = emp_info[emp_merge_cols].copy()
                    emp_copy["name"] = emp_copy["name"].astype(str).str.strip().str.title()
                    df_work = df_work.merge(emp_copy, on="name", how="left")

            # derive total_hours if present or safe fallback
            if "total_hours" not in df_work.columns and "hours" in df_work.columns:
                df_work["total_hours"] = df_work["hours"]
            df_work["total_hours"] = pd.to_numeric(df_work.get("total_hours", 0), errors="coerce").fillna(0)

            if "status_flag" not in df_work.columns:
                df_work["status_flag"] = df_work["total_hours"].apply(lambda x: "Green" if x >= 8 else "Red")

            for txt_col in ["state", "district", "project_name"]:
                if txt_col in df_work.columns:
                    df_work[txt_col] = df_work[txt_col].astype(str).str.title()

            # Apply same sidebar filters to this df_work
            if selected_name != "All" and "name" in df_work.columns:
                df_work = df_work[df_work["name"].str.strip().str.lower() == selected_name.strip().lower()]
            if selected_project != "All" and "project_name" in df_work.columns:
                df_work = df_work[df_work["project_name"].str.strip().str.lower() == selected_project.strip().lower()]
            if selected_state != "All" and "state" in df_work.columns:
                df_work = df_work[df_work["state"].str.strip().str.lower() == selected_state.strip().lower()]
            if selected_district != "All" and "district" in df_work.columns:
                df_work = df_work[df_work["district"].str.strip().str.lower() == selected_district.strip().lower()]

            def color_flag(val):
                if str(val).strip().lower() == "green":
                    return "background-color: lightgreen; color: black;"
                if str(val).strip().lower() == "red":
                    return "background-color: #ff4d4d; color: white;"
                return ""

            styled_df = df_work.style.applymap(color_flag, subset=["status_flag"]) if "status_flag" in df_work.columns else df_work.style

            st.markdown("### 🕒 Working Hours Efficiency")
            st.markdown("<style>table {border-collapse: collapse;} td, th {padding: 6px 10px !important;}</style>", unsafe_allow_html=True)
            st.write(styled_df.to_html(), unsafe_allow_html=True)

            fig_hours = px.bar(
                df_work,
                x="name" if "name" in df_work.columns else df_work.index,
                y="total_hours",
                color="status_flag" if "status_flag" in df_work.columns else None,
                title="Working Hours by Employee (Red < 8 hrs, Green ≥ 8 hrs)"
            )
            st.plotly_chart(fig_hours, use_container_width=True)

            # Downloads (Excel / PDF) - same as before
            from io import BytesIO
            st.markdown("### 📥 Download Working Hours Efficiency Report")
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df_work.to_excel(writer, index=False, sheet_name="Working Hours Efficiency")
            excel_buffer.seek(0)
            st.download_button(
                label="📊 Download Working Hours Efficiency (Excel)",
                data=excel_buffer,
                file_name=f"Working_Hours_Efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            def create_efficiency_pdf(dataframe):
                pdf_buffer = BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=landscape(A4))
                elements = []
                styles = getSampleStyleSheet()
                elements.append(Paragraph("Working Hours Efficiency Report", styles["Title"]))
                elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
                elements.append(Spacer(1, 12))
                table_data = [list(dataframe.columns)] + dataframe.astype(str).values.tolist()
                table = Table(table_data, repeatRows=1)
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.gray),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.beige, colors.white])
                ]))
                elements.append(table)
                doc.build(elements)
                pdf_buffer.seek(0)
                return pdf_buffer

            pdf_data = create_efficiency_pdf(df_work)
            st.download_button(
                label="📄 Download Working Hours Efficiency (PDF)",
                data=pdf_data,
                file_name=f"Working_Hours_Efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.warning(f"⚠️ Could not load work_summary/attendance for efficiency: {e}")

# ---------------- ATTENDANCE TAB ----------------
with tab_att:
    st.subheader("📋 Attendance Data")
    st.dataframe(df_filtered_att, use_container_width=True)

    # Excel Download
    if not df_filtered_att.empty:
        excel_path = os.path.join(PDF_EXPORT_DIR, f"Attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        df_filtered_att.to_excel(excel_path, index=False)
        with open(excel_path, "rb") as f:
            st.download_button("⬇️ Download Attendance Excel", f, file_name=os.path.basename(excel_path))

    # PDF Download
    if st.button("📄 Generate Attendance PDF"):
        pdf_path = generate_pdf_report(df_filtered_att, "Attendance Report")
        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Download Attendance PDF", f, file_name=os.path.basename(pdf_path))

# ---------------- LOGOUT TAB ----------------
with tab_log:
    st.subheader("📤 Logout Records")
    st.dataframe(df_filtered_log, use_container_width=True)

    if not df_filtered_log.empty:
        excel_path = os.path.join(PDF_EXPORT_DIR, f"Logout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        df_filtered_log.to_excel(excel_path, index=False)
        with open(excel_path, "rb") as f:
            st.download_button("⬇️ Download Logout Excel", f, file_name=os.path.basename(excel_path))

    if st.button("📄 Generate Logout PDF"):
        pdf_path = generate_pdf_report(df_filtered_log, "Logout Report")
        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Download Logout PDF", f, file_name=os.path.basename(pdf_path))

# ---------------- EDIT TAB ----------------
with tab_edit:
    st.subheader("✏️ Edit Attendance Records")
    editable_df = st.data_editor(df_filtered_edit, num_rows="dynamic", use_container_width=True)
    if st.button("💾 Save Changes to Database"):
        try:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            editable_df.to_sql("attendance", conn, if_exists="replace", index=False)
            conn.close()
            st.success("✅ Attendance records updated successfully.")
            try:
                st.cache_data.clear()
            except Exception:
                pass
        except Exception as e:
            st.error(f"Error saving changes: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("© 2025 Company Attendance AI System — Admin Control Panel")

st.markdown("### 🗂 Database Tables Viewer (For Admin Only)")
if st.checkbox("Show Raw Database Tables"):
    conn = sqlite3.connect(DB_PATH)
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    st.write("Available Tables:", tables)
    selected_table = st.selectbox("Select a Table to View", tables["name"].tolist())
    if selected_table:
        df_preview = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
        st.write(f"📋 Showing {len(df_preview)} rows from `{selected_table}`:")
        st.dataframe(df_preview, use_container_width=True)
    conn.close()
