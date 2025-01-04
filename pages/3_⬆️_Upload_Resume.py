import streamlit as st
import base64
from src.util.video_generation_final import video_generator, audio_generator, audio_video_merger, parse_text_to_json
from src.util.format_summary import generate_formated_output_gemini
from PIL import Image
from moviepy.editor import VideoFileClip, concatenate_videoclips
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
st.sidebar.selectbox("Select Model:", ["GPT", "T5", "RAG"]) 
text_contents = '''
Foo, Bar
123, 456
789, 000
'''
st.sidebar.download_button('Download Sample Resume', text_contents, 'text/csv')
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
# if  not st.session_state.authenticated :
#     st.switch_page("pages\\2_🔒_Login.py")


uploaded_file = st.file_uploader("Upload Your Resume")
summary = st.text_area(
    "Resume Summary",
   
)

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
        video_caption = {}
        scenes= generate_formated_output_gemini(summary,)
        required_sections = ['Introduction', 'Experience', 'Skills', 'Achievement', 'Goals', 'Contact']
        for section in required_sections:
            if section in scenes:
                print(f"--- {scenes[section]['Caption']} ---")
                 
                video_caption[section] = scenes[section]['Caption']
                   
                with st.spinner("Generating Audio... Please wait!"):
                        try:
                            audio_path = audio_generator(scenes[section]['Audio'], section)
                            st.success("Audio generated successfully!")
                        except Exception as e:
                            st.error(f"An error occurred: while generating audio {e}")  
                    
                with st.spinner("Generating video... Please wait!"):
                        try:
                            video_path = video_generator(scenes[section]['Visual'], duration, section)
                            st.success("Video generated successfully!")
                        except Exception as e:
                            st.error(f"An error occurred: while generating video {e}")  
            
       
        
        with st.spinner("Merging video in progress ... Please wait!"):
                try :
                        audio_video_merger("Introduction.wav","Introduction.mp4", "Introduction", video_caption.get('Introduction'))
                        audio_video_merger("Experience.wav","Experience.mp4", "Experience",  video_caption.get('Experience'))
                        audio_video_merger("Skills.wav","Skills.mp4", "Skills",  video_caption.get('Skills'))
                        audio_video_merger("Achievement.wav","Achievement.mp4","Achievement",  video_caption.get('Achievement'))
                        audio_video_merger("Goals.wav","Goals.mp4","Goals",  video_caption.get('Goals'))
                        audio_video_merger("Contact.wav","Contact.mp4","Contact",  video_caption.get('Contact'))
                        video_paths = [
                            "Introduction.mp4",
                            "Experience.mp4",
                            "Skills.mp4",
                            "Achievement.mp4",
                            "Contact.mp4"
                        ]

                    # Load video clips
                        video_clips = [VideoFileClip(video) for video in video_paths]

# Concatenate video clips
                        final_clip = concatenate_videoclips(video_clips, method="compose")

                    # Save the merged video
                        output_path = "merged_video.mp4"
                        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

                        # Close all clips
                        for clip in video_clips:
                            clip.close()
                            final_clip.close()


                        st.success("Merging completed successfully !!")
                        st.video("output_video.mp4")
                except Exception as e:
                            st.error(f"An error occurred: {e}")
                             
    
                