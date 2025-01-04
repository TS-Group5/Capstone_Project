import torch
import re
import moviepy.editor as mp
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

#merger import
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.editor import *
import cv2
#Audio generation
import os
from scipy.io.wavfile import write as write_wav
import numpy as np
import nltk  # we'll use this to split into sentences
from bark.generation import generate_text_semantic, preload_models
from bark.api import semantic_to_waveform
from bark import SAMPLE_RATE

# Download the necessary NLTK data
nltk.data.path.append('C:/Users/ankit/AppData/Roaming/nltk_data/tokenizers/punkt')
nltk.download('punkt')

def image_generator(prompt):
    """
    Function to generate an image using the prompt.
    """
    from diffusers import StableDiffusionPipeline
    
    print("Generating image for the prompt...")
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipeline.to("cuda")  # Move to GPU for faster processing
    
    # Generate the image
    image = pipeline(prompt, num_inference_steps=50).images[0]
    return image

def export_to_video(frames, file_name, fps=15):
    """
    Export frames to a video file using OpenCV.
    
    Args:
        frames (list): List of PIL Image frames.
        file_name (str): The name of the output video file.
        fps (int): Frames per second for the video.
    """
    # Convert the frames from PIL to numpy arrays
    frame_arrays = [cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR) for frame in frames]

    # Get the dimensions of the frames
    height, width, _ = frame_arrays[0].shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(file_name, fourcc, fps, (width, height))

    # Write frames to the video file
    for frame in frame_arrays:
        out.write(frame)

    # Release the video writer
    out.release()
    print(f"Video saved as {file_name}")

def video_generator(prompt, video_len, file_name):
    """
    Generate a video for the given prompt and duration.
    """
    print(f"Video prompt: {prompt}")

    # Load and configure the pipeline
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",  # Public model (assuming no special access required)
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipeline.enable_model_cpu_offload()  # Offload model layers to CPU to save GPU memory
    pipeline.to("cuda")  # Move pipeline to GPU for faster inference

    # Generate and resize the image
    image = image_generator(prompt)  # Use the image generator function
    image = image.resize((512, 288))  # Resize for optimization and faster processing

    # Define parameters
    fps = 15  # Frames per second (you can adjust based on your need)
    duration = video_len  # Video duration in seconds
    num_frames = fps * duration  # Total number of frames

    print(f"Generating all {num_frames} frames at once...")
    frames = pipeline(
        image=image,
        decode_chunk_size=num_frames,  # Generate all frames in one go
        generator=torch.manual_seed(42),
    ).frames[0]

    # Export the generated frames to a video file
    export_to_video(frames, file_name + ".mp4", fps=fps)
    print(f"Video generated and saved as {file_name}.mp4")

# #Image Genertor
# def image_generator(prompt):
#     from diffusers import StableDiffusionPipeline
#     from PIL import Image

#     # Load the Stable Diffusion pipeline
#     print("Loading Stable Diffusion pipeline...")
#     pipeline = StableDiffusionPipeline.from_pretrained(
#         "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
#     )
#     pipeline.to("cuda")  # Use GPU for faster inference

#     # Generate the image
#     print("Generating image...")
#     image = pipeline(prompt, num_inference_steps=25).images[0]  # Reduced inference steps for faster generation

#     # Resize the image for video compatibility
#     target_size = (384, 216)  # Standard size for video input
#     print(f"Resizing image to {target_size}...")
#     image = image.resize(target_size, Image.LANCZOS)

#     # Return the resized image
#     return image

#Audio Generator
# Function to use GPU for audio generation
def audio_generator(script, file_name):
    print(f"Audio prompt: {script}")

    # Preload models for Bark
    preload_models()
    
    # Check if GPU is being used (example for TensorFlow or PyTorch)
    try:

        if not torch.cuda.is_available():
            raise RuntimeError("GPU is not available. Please ensure proper GPU setup.")
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    except ImportError:
        print("PyTorch is not installed. Ensure your library dependencies support GPU usage.")

    # Split the script into sentences
    sentences = nltk.sent_tokenize(script)
    
    # Configuration for Bark
    GEN_TEMP = 0.6
    SPEAKER = "v2/en_speaker_6"
    SAMPLE_RATE = 22050  # Adjust as per your requirements
    silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

    # Generate audio for each sentence
    pieces = []
    for sentence in sentences:
        semantic_tokens = generate_text_semantic(
            sentence,
            history_prompt=SPEAKER,
            temp=GEN_TEMP,
            min_eos_p=0.05,  # Controls how likely the generation is to end
        )
        # Ensure semantic_to_waveform leverages GPU if supported
        audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER)
        pieces.append(audio_array)
        pieces.append(silence)

    # Concatenate all audio pieces into a single NumPy array
    final_audio = np.concatenate(pieces)

    # Write the concatenated audio to a WAV file
    write_wav(file_name + ".wav", SAMPLE_RATE, final_audio.astype(np.float32))
    print(f"Audio file saved as {file_name}.wav")

#Merging the Audio Video outcomes
def audio_video_merger(audio_file, video_file, output_file, caption):
    try:
        # Load audio and video
        audio_clip = AudioFileClip(audio_file)
        video_clip = VideoFileClip(video_file)
        
        # Set the audio for the video clip
        video_clip = video_clip.set_audio(audio_clip)

        # Optional: Add caption to the video (if required)
        video_clip = video_clip.subclip(0, video_clip.duration)  # Clip duration if needed

        # Write the final video to output file
        video_clip.write_videofile(f"{output_file}.mp4", codec="libx264", audio_codec="aac")

        video_clip.close()
        audio_clip.close()
    except Exception as e:
        print(f"An error occurred while merging {audio_file} and {video_file}: {e}")

def merge_videos():
    video_paths = [
        "Introduction.mp4",
        "Experience.mp4",
        "Skills.mp4",
        "Achievement.mp4",
        "Goals.mp4",
    ]
    
    try:
        # Load all video clips
        video_clips = [VideoFileClip(video) for video in video_paths]
        
        # Concatenate all clips into one
        final_clip = concatenate_videoclips(video_clips, method="compose")

        # Save the final merged video
        output_path = "merged_video.mp4"
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # Close all clips
        for clip in video_clips:
            clip.close()

        final_clip.close()

        print("Merging completed successfully!!")
        print("merged_video.mp4")
    except Exception as e:
        print(f"An error occurred during video merging: {e}")

# Example usage:

def parse_text_to_json(structured_text) :
    pattern = re.compile(
    r"Scene (\d+): (.*?)\nCaption:\n\"(.*?)\"\n\nAudio Script:\n\"(.*?)\"\n\nImage Prompt:\n\"(.*?)\"",
    re.DOTALL,
)
    matches = pattern.findall(structured_text)
    scenes = {}
    for match in matches:
        scene_number, title, caption, audio_script, image_prompt = match
        scenes[f"Scene {scene_number}"] = {
            "Title": title.strip(),
            "Caption": caption.strip(),
            "Audio Script": audio_script.strip(),
            "Image Prompt": image_prompt.strip(),
        }
    return scenes
