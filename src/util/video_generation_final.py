import os
import torch
import re
import moviepy.editor as mp
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from src.db.db_connector import insert_lipsync_data
#merger import
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.editor import *
#Audio generation
from scipy.io.wavfile import write as write_wav
import numpy as np
import nltk  # we'll use this to split into sentences
from bark.generation import generate_text_semantic, preload_models
from bark.api import semantic_to_waveform
from bark import SAMPLE_RATE
from src.util.aws_helper import upload_image_to_public_s3
from src.util.lipsync import generate_lip_sync

# Download the necessary NLTK data
nltk.data.path.append('/Users/anilkumar/nltk_data/tokenizers/punkt')
nltk.download('punkt')

#Video Generator
def video_generator(prompt, file_name):
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
    duration = 10  # 10 seconds of video
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
    export_to_video(frames, "src/video/"+file_name+".mp4", fps=fps)
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
# Function to use GPU for audio generation
def audio_generator(script, file_name, user_id, gender,avatart_aws_url):
    print(f"Audio prompt: {script}")
    
    # Ensure GPU is visible to the environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Preload models for Bark   
    preload_models()
    
    # Check if GPU is being used (example for TensorFlow or PyTorch)
    try:
        import torch
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
    if gender == "Female":
       SPEAKER= "v2/en_speaker_9"
       if avatart_aws_url is  None:
           avatart_aws_url="https://aimlops-cohort3-group5-capstone-project.s3.ap-south-1.amazonaws.com/male3.mp4"
    else :
        SPEAKER= "v2/en_speaker_6"
        if avatart_aws_url is  None:
           avatart_aws_url="https://aimlops-cohort3-group5-capstone-project.s3.ap-south-1.amazonaws.com/female2.mp4"

    print(f"avatar url ===={avatart_aws_url}")

   
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
    write_wav("src/audio/"+file_name +".wav", SAMPLE_RATE, final_audio.astype(np.float32))
    
    #upload the auto to s3 for lipsync
    file_name = "src/audio/"+file_name + ".wav"
    print("audio file name  "+file_name)
    bucket_name = "aimlops-cohort3-group5-capstone-project"
    public_audio_url = upload_image_to_public_s3(file_name, bucket_name)
    if public_audio_url:
        response_id=generate_lip_sync("https://aimlops-cohort3-group5-capstone-project.s3.ap-south-1.amazonaws.com/male3.mp4", public_audio_url)
        insert_lipsync_data( user_id, file_name, "audio", response_id)
        print("Public URL of the uploaded file:", public_audio_url)	

    print(f"Audio file saved as {file_name}.wav")

#Merging the Audio Video outcomes

def audio_video_merger(audio_file, video_file, output_file, caption=None):
    try:
        # Load audio and video
        audio_clip = AudioFileClip(audio_file)
        video_clip = VideoFileClip(video_file)
        
        # Get the duration of the audio and video
        audio_duration = audio_clip.duration
        video_duration = video_clip.duration

        # If video is shorter than audio, loop the video to match the audio duration
        if video_duration < audio_duration:
            num_loops = int(audio_duration // video_duration) + 1  # Repeat enough times
            video_clip = concatenate_videoclips([video_clip] * num_loops)
            video_clip = video_clip.subclip(0, audio_duration)  # Trim to match audio duration

        # If video is longer than audio, trim the video to match the audio duration
        elif video_duration > audio_duration:
            video_clip = video_clip.subclip(0, audio_duration)

        # Set the audio for the video clip
        print(f"audio_clip  ====={audio_file}")
        video_clip = video_clip.set_audio(audio_clip)

        # Optional: Add caption to the video (if required)
        if caption:
            video_clip = video_clip.set_duration(audio_duration)  # Ensure caption matches audio duration

        # Write the final video to output file
        video_clip.write_videofile(f"{output_file}.mp4", codec="libx264", audio_codec="aac")

        # Close all clips
        video_clip.close()
        audio_clip.close()
        
    except Exception as e:
        print(f"An error occurred while merging {audio_file} and {video_file}: {e}")

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
