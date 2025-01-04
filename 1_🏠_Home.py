import streamlit as st
import base64
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
st.set_page_config(layout='wide')
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add background image (adjust the filename if needed)
add_bg_from_local("./img/back.png")
col1, spacer, col2 = st.columns([0.5, 0.1, 0.5]) 

with col1:
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True) 
    st.markdown(
    """
    <h1 style="color: #03f8fc;">Secure your dream job with a video resume that truly represents you.</h1>
    """,
    unsafe_allow_html=True,
)
    st.markdown("""
    <p style="color: white; font-size: 16px;">
        With our application, effortlessly create stunning video resumes that showcase your achievements and qualifications in an engaging way. 
        Simply upload your resume, and let the AI handle the rest.
    </p>
    """, unsafe_allow_html=True)
# Display button and handle click
if st.button("Create Your Video Resume"):
    if  st.session_state.authenticated :
        st.switch_page("pages\\3_⬆️_Upload_Resume.py")
    else :
        st.switch_page("pages\\2_🔒_Login.py")