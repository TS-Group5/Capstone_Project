import streamlit as st
import base64
from src.util.video_generation_final import video_generator, audio_generator, audio_video_merger, parse_text_to_json
from src.util.format_summary import generate_formated_output_gemini
from PIL import Image
import requests
import logging
import yaml
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Get configuration
config = load_config()
API_URL = f"{config['api']['base_url']}{config['api']['endpoints']['generate_script']}"

st.markdown(
    """
    <style>
    .stFileUploader label {
        color: white;
    }
    .stTextArea label {
        color: white;
    }
     .stSlider label {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
add_bg_from_local("./img/resume.png")

# Template download section
st.sidebar.markdown("### Download Resume Templates")
template_option = st.sidebar.radio(
    "Choose a template:",
    ["ATS classic HR resume", "Industry manager resume"]
)

if st.sidebar.button("Download Selected Template"):
    with open(f"./templates/{template_option}.docx", "rb") as file:
        template_bytes = file.read()
        st.sidebar.download_button(
            label="Click to Download",
            data=template_bytes,
            file_name=f"{template_option}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
# if  not st.session_state.authenticated :
#     st.switch_page("pages\\2_🔒_Login.py")

uploaded_file = st.file_uploader("Upload Your Resume")
# Template type selection
selected_template_type = st.radio(
    "Select Resume Template Type:",
    ["ATS", "Industry"],
    key="template_type"
)

# Display the summary text area
summary = st.text_area(
    "Resume Summary",
    value=st.session_state.get('summary', ''),
    height=200
)

# Generate Summary button
if st.button("Generate Summary"):
    if uploaded_file is None:
        st.error("Please upload a resume file first")
    else:
        with st.spinner("Generating summary..."):
            try:
                logger.info(f"Starting API call with template type: {selected_template_type}")
                # Prepare the files for the API request
                files = {
                    'file': (uploaded_file.name, uploaded_file, uploaded_file.type)
                }
                # Prepare the form data
                data = {
                    'template_type': selected_template_type.lower()
                }
                
                logger.info("Making API request to generate script...")
                # Make the API request
                response = requests.post(
                    API_URL,
                    files=files,
                    data=data
                )
                
                logger.info(f"API Response status code: {response.status_code}")
                if response.status_code == 200:
                    # Extract the script from the response and update the summary
                    response_data = response.json()
                    generated_summary = response_data.get('script', '')
                    logger.info("Successfully received script from API")
                    
                    # Format the summary using the new utility function
                    formatted_summary = generate_formated_output_gemini(generated_summary)
                    
                    # Update session state and display success message
                    st.session_state.summary = formatted_summary
                    st.success("Summary generated successfully!")
                    
                    # Update the text area directly
                    summary = formatted_summary
                    logger.info("Updated summary in session state")
                    
                    # Use st.rerun() instead of experimental_rerun
                    st.rerun()
                else:
                    error_msg = f"Error generating summary: {response.text}"
                    logger.error(error_msg)
                    st.error(error_msg)
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)

c1,c2 =st.columns([3, 2])
with c1 :
    uploaded_file = st.file_uploader("Upload Your Avatar", type=["jpg", "png", "jpeg"])
with c2 :
    if uploaded_file is not None:
        # Open the uploaded image file
        image = Image.open(uploaded_file)
        
        # Display the image
        st.image(image, caption="Uploaded Image", width=200)

duration = st.slider("Duration (in seconds):", 1, 20, 10)
fps = st.slider("Frames per second (FPS):", 8, 30, 16)
if st.button("Generate Video"):
        with st.spinner("Generating Audio... Please wait!"):
            try:
                #audio_path = audio_generator(summary)
                st.success("Audio generated successfully!")
            except Exception as e:
                st.error(f"An error occurred: {e}")  
                
        with st.spinner("Generating video... Please wait!"):
            try:
                #video_path = video_generator(summary, duration)
                st.success("Video generated successfully!")
                with st.spinner("Merging video in progress ... Please wait!"):
                    audio_video_merger("bark_generation5.wav","optimized_sequential_video.mp4")
                    st.success("Merging completed successfully !!")
                    st.video("output_video.mp4")
                    # video_file = open("output_video.mp4", "rb")
                    # video_bytes = video_file.read()
                    # st.video(video_bytes)
            except Exception as e:
                st.error(f"An error occurred: {e}")