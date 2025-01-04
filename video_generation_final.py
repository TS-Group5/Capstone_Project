import torch
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
nltk.download('punkt')

#Video Generator
def video_generator(prompt, video_len):
    # Load the pipeline for video generation
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
    decode_chunk_size = 32  # Frames generated per chunk

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
    export_to_video(frames, "optimized_sequential_video.mp4", fps=fps)
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
def audio_generator(script):
    
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
    write_wav("bark_generation5.wav", SAMPLE_RATE, final_audio.astype(np.float32))

#Merging the Audio Video outcomes
def audio_video_merger(audio_input, video_input):
    # Paths to your video and audio files
    video_path = "optimized_sequential_video.mp4"
    audio_path = "bark_generation5.wav"
    output_path = "output_with_audio.mp4"

    try:
        # Load the video and audio
        video1 = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)

        # Set the audio to the video
        video = video1.set_audio(audio)
        # Loop the video to match the duration of the audio
        video = video.fx(mp.vfx.loop, duration=audio.duration)
        
        # Export the final video
        video.write_videofile("output_video.mp4", codec="libx264", audio=True, verbose=True, fps=16)
        print("Video with audio saved as:", output_path)

    finally:
        # Release resources
        video.close()
        audio.close()
        if 'video' in locals():
            video.close()
        print("Resources released.")


