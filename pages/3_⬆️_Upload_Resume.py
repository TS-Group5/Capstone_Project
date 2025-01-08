import streamlit as st
import base64
from src.util.video_generation_final import video_generator, audio_generator, audio_video_merger, parse_text_to_json
from src.util.format_summary import generate_formated_output_gemini
from PIL import Image
import requests
import logging
import yaml
import os
from moviepy.editor import AudioFileClip, VideoFileClip, TextClip, CompositeVideoClip
from moviepy.editor import concatenate_videoclips
from src.db.db_connector import get_url_by_id
# Configure logging
from src.util.lipsync import fetch_video
from src.util.overlap import vdo_with_circular_bgvdo
from src.util.aws_helper import upload_image_to_public_s3
from src.db.db_connector import getKey
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": getKey("IMAGEMAGICK_BINARY")})
# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Get configuration
config = load_config()
API_URL = f"{config['api']['base_url']}{config['api']['endpoints']['generate_script']}"

st.markdown("""
    <style>
    .tv-frame {
        border: 10px solid black;  /* TV border color */
        border-radius: 15px;  /* Rounded corners */
        padding: 5px;
        background-color: #000000; /* TV screen background color */
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.8); /* Shadow for the TV effect */
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .tv-screen {
        width: 384px;  /* Width of the video */
        height: 216px;  /* Height of the video */
        border-radius: 10px;  /* Rounded corners of the screen */
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
             /* Reduce default padding and margin from the top */
    .block-container {
        padding-top: 10px; /* Adjust this value as needed */
    }
    .center-heading {
        text-align: center;  /* Center the text */
        font-size: 36px;  /* Adjust font size */
        font-weight: bold;  /* Make the text bold */
        margin: 0px;  /* Add space at the top */
        color: white;  /* Set font color to white */
        background-color: rgba(245, 245, 245, 0.5); 
        border: 2px solid white;  /* Add a white border */
        padding: 20px;  /* Add padding inside the rectangle */
        border-radius: 10px;  /* Optional: Rounded corners */
        display: inline-block;  /* Fit the content size */
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .column-button {
        display: flex;
        align-items: center;  /* Vertical alignment */
        justify-content: center;  /* Horizontal alignment */
        height: 50px;  /* Adjust height as needed */
    }
    </style>
""", unsafe_allow_html=True)
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
    .stRadio > label {
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
st.sidebar.markdown("### Download Resume Templates")
template_option = st.sidebar.radio(
    "Choose a template:",
    ["ATS classic HR resume", "Industry manager resume"]
)
if 'user_info' in st.session_state :
    user_info=st.session_state.user_info

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
if  not st.session_state.authenticated :
    st.switch_page("app\\pages\\2_🔒_Login.py")
st.markdown('<center><h1 class="center-heading">&nbsp&nbspWelcome to SpotLightCV</h1><center>', unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #ccc; margin-top: 20px;'>", unsafe_allow_html=True)

#################################################################################################

# # Main script to link video_capture.py
# from pages.video_capture import toggle_camera_preview, record_video

# # Initialize session states
# if "preview_started" not in st.session_state:
#     st.session_state.preview_started = False
# if "cap" not in st.session_state:
#     st.session_state.cap = None
# if "video_frame_placeholder" not in st.session_state:
#     st.session_state.video_frame_placeholder = st.empty()
# if "background_option" not in st.session_state:
#     st.session_state.background_option = "Black"

# st.markdown("### Capture Video")
# st.markdown("**Capture close-up video of your face reading the phrase - The quick brown fox jumps over the lazy dog.**")

# # Background selection
# image_dir = os.path.join(os.path.dirname(os.getcwd()), 'Capstone_Project', 'Images')
# image_files = [img for img in os.listdir(image_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]

# bg_options = ["White", "Black", "Blurred", "Original"] + [f"Image: {img}" for img in image_files]
# st.session_state.background_option = st.selectbox("Choose your background:", bg_options, index=0, format_func=lambda x: x.split(": ")[1] if x.startswith("Image: ") else x)

# # Display available images as thumbnails for context
# st.markdown("#### Background Image Previews")
# for img in image_files:
#     img_path = os.path.join(image_dir, img)
#     st.image(img_path, caption=img, use_container_width=True)

# camera_button = st.button("Camera Preview", on_click=toggle_camera_preview)
# record_button = st.button("Start Recording", on_click=record_video)


#################################################################################################
c1,c2,c3 =st.columns([1, 3,1])
with c1:
     selected_template_type = st.radio(
    "Select Resume Template Type:",
    ["ATS", "Industry"],
    key="template_type"
)
with c2:
    uploaded_file = st.file_uploader("Upload Your Resume")
# Template type selection
with c3:
    st.markdown('<div class="column-button">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)
c1,c2 =st.columns([4,1])
# Display the summary text 
with c1:
    summary = st.text_area(
        "Resume Summary",
        value=st.session_state.get('summary', ''),
        height=200
    )


with c1:
    
     uploaded_file = st.file_uploader("Upload your avatar(video)", type=["mp4"])
with c2:
   
     gender = st.selectbox(
    'Choose your Gender:',
    ('Male', 'Female')
)
with c2:
    if uploaded_file is not None:
        avatar_path = "src/avatar_video/avatar.mp4"
    
    # Save the uploaded video file locally
        with open(avatar_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.video(uploaded_file)
    else:
        if gender =="Male":
            st.video("src/avatar_video/male_afatar.mp4")
        else :
              st.video("src/avatar_video/female_avatar.mp4")
             
    st.text("Your Avatar")
     
# Generate Summary button

message_placeholder = st.empty()
public_url= None
with c2:
    st.markdown('<div class="column-button">', unsafe_allow_html=True)
    if st.button("Generate Video"):
            #upload avatar
            if uploaded_file is not None:
                bucket_name = "aimlops-cohort3-group5-capstone-project"
                public_url = upload_image_to_public_s3(avatar_path, bucket_name)
                print(f"aws url ={public_url}")
            video_caption = {}
            scenes= generate_formated_output_gemini(summary,)
            required_sections = ['Introduction', 'Experience', 'Skills', 'Achievement', 'Goals', 'Contact']
            for section in required_sections:
                if section in scenes:
                    print(f"--- {scenes[section]['Caption']} ---")
                    
                    with st.spinner("Generating Audio... Please wait!"):
                            try:
                                 avatart_aws_url=public_url
                                 audio_path = audio_generator(scenes[section]['Audio'], section,user_info.get("id"), gender,avatart_aws_url )
                                 
                                 message_placeholder.success("Audio generated successfully!")
                            except Exception as e:
                                st.error(f"An error occurred: while generating audio {e}")  
                        
                    with st.spinner("Generating video... Please wait!"):
                            try:
                                #video_path = video_generator(scenes[section]['Visual'], duration, section)
                                 message_placeholder.success("Video generated successfully!")
                            except Exception as e:
                                st.error(f"An error occurred: while generating video {e}")  
            with st.spinner("Merging video in progress ... Please wait!"):
                    try :
                            #os.environ['IMAGEMAGICK_BINARY'] = r'src\util\ImageMagick-7.1.1-43-Q16-x64-dll.exe'
                            print(f"working..........."+scenes['Introduction']['Caption'])

                          
                            vdo_with_circular_bgvdo(r"src/video/Introduction.mp4", r"src/avatar_video/Introduction.mp4", r"src/merged_video/Introduction.mp4",150, 20, 20, scenes['Introduction']['Caption'])
                            vdo_with_circular_bgvdo(r"src/video/Experience.mp4", r"src/avatar_video/Experience.mp4", r"src/merged_video/Experience.mp4",150, 20, 20, scenes['Experience']['Caption'])
                            vdo_with_circular_bgvdo(r"src/video/Skills.mp4", r"src/avatar_video/Skills.mp4", r"src/merged_video/Skills.mp4",150, 20, 20, scenes['Skills']['Caption'])
                            vdo_with_circular_bgvdo(r"src/video/Achievement.mp4", r"src/avatar_video/Achievement.mp4", r"src/merged_video/Achievement.mp4",150, 20, 20, scenes['Achievement']['Caption'])
                            vdo_with_circular_bgvdo(r"src/video/Goals.mp4", r"src/avatar_video/Goals.mp4", r"src/merged_video/Goals.mp4",150, 20, 20, scenes['Goals']['Caption'])
                            vdo_with_circular_bgvdo(r"src/video/Contact.mp4", r"src/avatar_video/Contact.mp4", r"src/merged_video/Contact.mp4",150, 20, 20, scenes['Contact']['Caption'])
                          
                            video_paths = [
                                r"src/merged_video/Introduction.mp4",
                                r"src/merged_video/Experience.mp4",
                                r"src/merged_video/Skills.mp4",
                                r"src/merged_video/Achievement.mp4",
                                r"src/merged_video/Goals.mp4",
                                r"src/merged_video/Contact.mp4"
                            ]

                        # Load video clips
                            video_clips = [VideoFileClip(video) for video in video_paths]

    # Concatenate video clips
                            final_clip = concatenate_videoclips(video_clips, method="compose")

                        # Save the merged video
                            output_path = "src/final_video/merged_video.mp4"
                            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

                            # Close all clips
                            for clip in video_clips:
                                clip.close()
                                final_clip.close()


                            message_placeholder.success("Merging completed successfully !!")
                            video_file = open(output_path, "rb")
                            video_bytes = video_file.read()

                            st.sidebar.video(video_bytes)
                                  
                    except Exception as e:
                                st.error(f"An error occurred: {e}")
                                
    
                