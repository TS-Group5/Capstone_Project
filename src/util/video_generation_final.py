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


def load_video_diffusion_model():
    print("Loading Stable Video Diffusion model...")
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipeline.to("cuda")  # Use GPU for faster inference
    return pipeline

def generate_video_frame(pipeline, prompt, video_len, fps=15):
    print(f"Generating video for prompt: {prompt}")
    
    # Define parameters for video generation
    generator = torch.manual_seed(42)
    num_frames = fps * video_len

    # Generate the video frames
    frames = pipeline(
        prompt=prompt,
        num_inference_steps=15,
        num_frames=num_frames,
        generator=generator
    ).frames[0]
    
    return frames

def save_video(frames, output_path, fps=15):
    height, width, _ = np.array(frames[0]).shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

    out.release()
    print(f"Video saved to {output_path}")

#Video Generator
def video_generator(prompt, video_len, file_name):
    # Load the video generation model
    video_diff_pipe = load_video_diffusion_model()

    # Generate the video frames
    frames = generate_video_frame(video_diff_pipe, prompt, video_len)

    # Save the generated video
    save_video(frames, file_name + ".mp4")


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
def audio_video_merger(audio_input, video_input, output, text_to_add, font_size=16, font_color='white', rect_color='red', rect_opacity=0.5):
   
    print(f"audio_video_merger text_to_add = {text_to_add}, audio_input ={audio_input}, video_input= {video_input},output ={output}")
    """
    Merges audio and video, adds text overlay with a rectangle background at the bottom, and saves the final output.

    Parameters:
    - audio_input: Path to the audio file.
    - video_input: Path to the video file.
    - output: Output file name (without extension).
    - text_to_add: Text to overlay on the video.
    - font_size: Size of the font for the text.
    - font_color: Color of the text ('white' by default).
    - rect_color: Background color of the rectangle ('red' by default).
    - rect_opacity: Opacity of the rectangle (0.5 by default).
    """
    # Paths to your video and audio files
    video_path = video_input
    audio_path = audio_input
    output_path = output + ".mp4"

    try:
        # Load the video and audio
        video1 = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)

        # Set the audio to the video
        video = video1.set_audio(audio)
        
        # Loop the video to match the duration of the audio
        video = video.fx(vfx.loop, duration=audio.duration)

        # Create the text clip
        text_clip = TextClip(
            text_to_add,
            fontsize=font_size,
            color=font_color,
            font="Arial"
        ).set_position(('center', 'bottom')).set_duration(video.duration)

        # Create the rectangle background
        rect_width = video.size[0]  # Width of the video
        rect_height = font_size + 20  # Height of the rectangle (adjust as needed)
        rectangle = ColorClip(
            size=(rect_width, rect_height),
            color=(255, 0, 0)  # Red color
        ).set_opacity(rect_opacity).set_position(('center', 'bottom')).set_duration(video.duration)

        # Combine the text and rectangle
        video_with_text = CompositeVideoClip([video, rectangle, text_clip])

        # Export the final video
        video_with_text.write_videofile(output_path, codec="libx264", audio=True, verbose=True, fps=16)
        print("Video with audio and text inside a rectangle saved as:", output_path)

    finally:
        # Release resources
        video.close()
        audio.close()
        if 'video' in locals():
            video.close()
        if 'video_with_text' in locals():
            video_with_text.close()
        print("Resources released.")

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
