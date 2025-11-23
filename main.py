import streamlit as st
import init_database

# ---------------------- AUTHENTICATION SYSTEM ----------------------

# Hard-coded users (you can replace with DB later)
USERS = {
    "Raju@sexyboy": "raju12345",
}

# Initialize session state for login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

def login_page():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("Login successful! Redirecting...")
            st.rerun()
        else:
            st.error("Invalid username or password")

def logout_button():
    st.sidebar.write("---")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()


# ---------------------- PROTECT ALL PAGES ----------------------

if not st.session_state.authenticated:
    login_page()
    st.stop()    # 🔒 Stop the app until login

# If logged in, show logout option
logout_button()


# ---------------------- PAGE SETUP ----------------------

About_page = st.Page(
    "About.py",
    default=True,
)
project_1_page = st.Page("get_faces_from_camera_tkinter.py")
project_2_page = st.Page("app_streamlit_login.py")
project_3_page = st.Page("app_streamlit_logout.py")
project_4_page = st.Page("init_database.py")
project_5_page = st.Page("admin_dashboard.py")
project_6_page = st.Page("supervisor_dashboard.py")

# ---------------------- NAVIGATION ----------------------

st.markdown("""
<style>

    /* STYLE ONLY SECTION HEADERS (“Info”, “Projects”) */
    div[data-testid="stSidebarNavSectionHeader"] > div,
    div[data-testid="stSidebarNavSectionHeader"] > span {
        font-size: 22px !important;
        font-weight: 900 !important;
        color: #0E4C92 !important;
        text-transform: uppercase !important;
        margin-top: 18px !important;
        margin-bottom: 8px !important;
        letter-spacing: 1px !important;
    }

    /* RESET: Make sure page keys stay normal */
    div[data-testid="stSidebarNav"] ul li p {
        font-size: 16px !important;
        font-weight: 400 !important;
        color: inherit !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
    }

</style>
""", unsafe_allow_html=True)




pg = st.navigation(
    {
        "Info": [About_page],
        "Projects": [
            project_1_page,
            project_2_page,
            project_3_page,
            project_5_page,
            project_6_page,
        ],
    }
)



# ---------------------- SHARED HEADER ----------------------

st.logo("assets/logo.png")

# ---------------------- RUN PAGES ----------------------

pg.run()



