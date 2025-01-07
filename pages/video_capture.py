# video_capture.py

import streamlit as st
import cv2
import os
import tempfile
import time
import numpy as np
import mediapipe as mp

def toggle_camera_preview():
    if st.session_state.preview_started:
        stop_camera_preview()
    else:
        start_camera_preview()

def remove_background(frame, bg_option):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(frame_rgb)
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5

        if bg_option == "Black":
            bg_color = (0, 0, 0)  # Black background
            bg_image = np.zeros(frame.shape, dtype=np.uint8)
            bg_image[:] = bg_color
        elif bg_option == "White":
            bg_color = (255, 255, 255)  # White background
            bg_image = np.zeros(frame.shape, dtype=np.uint8)
            bg_image[:] = bg_color
        elif bg_option.startswith("Image: "):
            image_name = bg_option.split(": ")[1]
            image_path = os.path.join(os.path.dirname(os.getcwd()), "Capstone_Project", "Images", image_name)
            if os.path.exists(image_path):
                bg_image = cv2.imread(image_path)
                bg_image = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
            else:
                bg_image = np.zeros(frame.shape, dtype=np.uint8)  # Fallback to black if image is not found
        elif bg_option == "Blurred":
            bg_image = cv2.GaussianBlur(frame, (55, 55), 0)
        else:
            bg_image = frame  # No background change

        output_frame = np.where(condition, frame, bg_image).astype(np.uint8)
        return output_frame

def start_camera_preview():
    st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for faster initialization
    st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set max resolution
    st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not st.session_state.cap.isOpened():
        st.error("Unable to open camera. Please check permissions.")
        return
    st.session_state.preview_started = True
    while st.session_state.preview_started:
        ret, frame = st.session_state.cap.read()
        if ret:
            bg_option = st.session_state.get("background_option", "Black")
            frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Reduce preview size to half
            frame_no_bg = remove_background(frame_resized, bg_option)
            st.session_state.video_frame_placeholder.image(frame_no_bg, channels="BGR", caption="Camera Preview", use_container_width=True)
        else:
            st.error("Failed to capture video frame.")
            break
        time.sleep(0.03)  # Reduce delay for faster responsiveness

def stop_camera_preview():
    st.session_state.preview_started = False
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None

def record_video():
    if not st.session_state.preview_started:
        st.error("Start the camera preview before recording.")
        return

    st.info("Recording... Please ensure your face occupies 50% of the frame.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_file:
        video_path = video_file.name

    st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for faster initialization
    st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set max resolution
    st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not st.session_state.cap.isOpened():
        st.error("Unable to open camera for recording.")
    else:
        frame_width = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))

        for _ in range(100):  # Record for 5 seconds at ~20 FPS
            ret, frame = st.session_state.cap.read()
            if ret:
                bg_option = st.session_state.get("background_option", "Black")
                frame_no_bg = remove_background(frame, bg_option)
                frame_resized = cv2.resize(frame_no_bg, (0, 0), fx=0.5, fy=0.5)  # Reduce preview size to half
                st.session_state.video_frame_placeholder.image(frame_resized, channels="BGR", caption="Recording...", use_container_width=True)
                out.write(frame_no_bg)
            else:
                break

        st.session_state.cap.release()
        out.release()

        st.success("Video recording completed!")

        # Autosave the video
        save_directory ="C:\\Users\\ankit\\AppData\\Local\\Temp"
        os.makedirs(save_directory, exist_ok=True)

        existing_files = [f for f in os.listdir(save_directory) if f.startswith("user_video_clip") and f.endswith(".mp4")]
        next_index = len(existing_files) + 1
        autosave_path = os.path.join(save_directory, f"user_video_clip_{next_index}.mp4")

        try:
            os.rename(video_path, autosave_path)
        except FileExistsError:
            st.error("A file with the same name already exists. Please remove it or try again.")
            return

        st.info(f"Video autosaved to: {autosave_path}")

        # Play the video
        st.video(autosave_path)

        # Download button
        with open(autosave_path, "rb") as file:
            st.download_button(
                label="Download Video",
                data=file,
                file_name=f"user_video_clip_{next_index}.mp4",
                mime="video/mp4"
            )

        stop_camera_preview()