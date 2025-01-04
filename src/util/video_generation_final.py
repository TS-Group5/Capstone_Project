import torch
import re
import moviepy.editor as mp
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

#merger import
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.editor import *

#Audio generation
import os
from scipy.io.wavfile import write as write_wav
import numpy as np
import nltk  # we'll use this to split into sentences
from bark.generation import generate_text_semantic, preload_models
from bark.api import semantic_to_waveform
from bark import SAMPLE_RATE

# Download the necessary NLTK data
nltk.data.path.append('C:/Users/ankit\AppData/Roaming/nltk_data/tokenizers/punkt')
nltk.download('punkt')

#Video Generator
def video_generator(prompt, video_len, file_name):
    # Load the pipeline for video generation
    print(f"video prompt = {prompt}")
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    )
    pipeline.enable_model_cpu_offload()
    pipeline.to("cuda")  # Enable GPU acceleration

    # Load and resize the input image
    image = image_generator(prompt)
    image = image.resize((512, 288))  # Reduced resolution for faster processing

    # Define parameters
    generator = torch.manual_seed(42)
    fps = 15  # Lower FPS for optimization
    duration = video_len  # 10 seconds of video
    num_frames = fps * duration
    decode_chunk_size = 64  # Frames generated per chunk

    # Generate frames sequentially
    frames = []
    for i in range(0, num_frames, decode_chunk_size):
        print(f"Generating frames {i} to {i + decode_chunk_size}")
        chunk_frames = pipeline(
            image=image,
            decode_chunk_size=min(decode_chunk_size, num_frames - i),
            generator=generator,
        ).frames[0]
        frames.extend(chunk_frames)

    # Save the video
    export_to_video(frames, file_name+".mp4", fps=fps)
    print("Video generated and saved as 'optimized_sequential_video.mp4'")

#Image Genertor
def image_generator(prompt):
    from diffusers import StableDiffusionPipeline
    #for video
    from diffusers import StableVideoDiffusionPipeline
    from diffusers.utils import load_image, export_to_video
    # Load the Stable Diffusion pipeline
    print("Loading Stable Diffusion pipeline...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    )
    pipeline.to("cuda")  # Use GPU for faster inference

    # Generate the image                                                                                                                                                                           
    print("Generating image...")
    image = pipeline(prompt, num_inference_steps=50).images[0]

    # return the generated image
    return image

#Audio Generator
def audio_generator(script, file_name):
    
    print(f"audion prompt = {script}")
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Preload models for Bark
    preload_models()

    # Split the script into sentences
    sentences = nltk.sent_tokenize(script)

    # Configuration for Bark
    GEN_TEMP = 0.6
    SPEAKER = "v2/en_speaker_6"
    silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

    # Generate audio for each sentence
    pieces = []
    for sentence in sentences:
        semantic_tokens = generate_text_semantic(
            sentence,
            history_prompt=SPEAKER,
            temp=GEN_TEMP,
            min_eos_p=0.05,  # this controls how likely the generation is to end
        )
        audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER)
        pieces.append(audio_array)
        pieces.append(silence)

    # Concatenate all audio pieces into a single NumPy array
    final_audio = np.concatenate(pieces)

    # Write the concatenated audio to a WAV file
    write_wav(file_name+".wav", SAMPLE_RATE, final_audio.astype(np.float32))

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
        video1.close()
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
