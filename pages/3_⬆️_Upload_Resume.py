import streamlit as st
import base64
from video_generation_final import video_generator, audio_generator, audio_video_merger
from PIL import Image
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
                