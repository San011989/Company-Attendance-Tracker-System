import streamlit as st
import base64
from pathlib import Path

st.set_page_config(page_title="About JSI", layout="wide")

# ------------------ LOGO ENCODER FUNCTION ------------------
def encode_image(image_path: str) -> str:
    """Return base64 string of an image file. Use a local path to your logo."""
    p = Path(image_path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ------------------ USER CONFIG ------------------
# Replace this with the path to your logo file. Example: "assets/logo.png"
LOGO_PATH = r"assets/logo.png"

CONTACT = {
    "Organization": "John Snow India (JSI)",
    "Address": "Some address line, City, State, PIN",
    "Phone": "+91-XXXXXXXXXX",
    "Email": "contact@example.com",
}

ABOUT_TITLE = "About JSI"
ABOUT_TEXT = (
    "JSI has been present in India for over 30 years catalyzing solutions to the country’s most urgent healthcare needs. "
    "This commitment was further reinforced in 2013, when John Snow India Private Limited (JSIPL) was formally established as a 100% owned subsidiary of JSI Research and Training Institute, Inc. "
    "The India entity combines rich global experience and technical expertise with deep, local knowledge and networks. The organization’s strength lies in nurturing strong partnerships with governments, private sector and civil society to serve the needs of multiple stakeholders."
)

# ------------------ STYLES: Import a clean modern font and center the content ------------------
# This CSS uses Google Fonts. If running in an air-gapped environment, change the font-family to a system font.
CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;600&family=Inter:wght@300;400;600&display=swap');

.section-wrapper {{
    font-family: 'Lora', 'Inter', Georgia, serif;
    padding: 30px 40px;
}}

.logo-container img {{
    max-width: 180px;
    height: auto;
    display: block;
    margin-bottom: 10px;
}}

.contact-box {{
    text-align: right;
    font-family: 'Inter', Arial, sans-serif;
    font-size: 14px;
}}

.about-card {{
    background: rgba(255,255,255,0.9);
    padding: 40px 60px;
    border-radius: 12px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.06);
    max-width: 900px;
    margin: 20px auto;
}}

.about-title {{
    text-align: center;
    font-size: 34px;
    font-weight: 600;
    margin-bottom: 12px;
}}

.about-text {{
    text-align: center;
    font-size: 17px;
    line-height: 1.8;
    color: #222;
}}

/* Responsive tweaks */
@media (max-width: 768px) {{
    .contact-box {{ text-align: left; margin-top: 10px; }}
    .logo-container img {{ max-width: 140px; }}
    .about-card {{ padding: 28px 20px; }}
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------ TOP BAR: logo (left) and contact details (right) ------------------
col1, col2 = st.columns([1, 3])

with col1:
    logo_b64 = encode_image(LOGO_PATH)
    if logo_b64:
        logo_html = f"<div class='logo-container'><img src='data:image/png;base64,{logo_b64}' alt='Logo'></div>"
        st.markdown(logo_html, unsafe_allow_html=True)
    else:
        # Fallback placeholder
        st.markdown("<div class='logo-container'><h3>JSI Logo</h3></div>", unsafe_allow_html=True)

with col2:
    contact_html = "<div class='contact-box'>"
    contact_html += f"<strong>{CONTACT['Organization']}</strong><br/>"
    contact_html += f"{CONTACT['Address']}<br/>"
    contact_html += f"Phone: {CONTACT['Phone']}<br/>"
    contact_html += f"Email: {CONTACT['Email']}"
    contact_html += "</div>"
    st.markdown(contact_html, unsafe_allow_html=True)

st.write("\n")

# ------------------ CENTERED ABOUT SECTION ------------------
# Use a single column to center the card
st.markdown("<div class='section-wrapper'>", unsafe_allow_html=True)
st.markdown(f"<div class='about-card'>\n  <div class='about-title'>{ABOUT_TITLE}</div>\n  <div class='about-text'>{ABOUT_TEXT}</div>\n</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------ OPTIONAL FOOTER ------------------
st.markdown("<div style='text-align:center; margin-top:20px; font-size:12px; color:#666;'>\n&copy; {year} John Snow India (JSI) - All rights reserved.\n</div>".format(year=2025), unsafe_allow_html=True)

