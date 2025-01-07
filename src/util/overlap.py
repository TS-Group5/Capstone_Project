from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip, ColorClip, TextClip
import numpy as np
from moviepy.video.fx.all import speedx

def create_circular_mask(diameter):
    # Create a blank square image with the same width and height
    size = (diameter, diameter)
    mask = np.zeros((diameter, diameter), dtype=np.uint8)

    # Define the center and radius
    center = (diameter // 2, diameter // 2)
    radius = diameter // 2

    # Create a circular region
    y, x = np.ogrid[:diameter, :diameter]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask[dist_from_center <= radius] = 255

    return mask

def vdo_with_circular_bgvdo(bg_video_file, sec_bg_video_file, output_file, bgwidth, margin_bottom, margin_right, txt_caption):
    # Load secondary video
    sec_bg_video = VideoFileClip(sec_bg_video_file)
    sec_duration = sec_bg_video.duration + 2  # Target duration for background video

    # Load background video and slow it down to match target duration
    bg_video = VideoFileClip(bg_video_file, audio=False)
    bg_video = bg_video.fx(speedx, bg_video.duration / sec_duration)  # Adjust speed
    bg_w, bg_h = bg_video.size

    # Resize secondary video
    sec_bg_video = sec_bg_video.resize((bgwidth, bgwidth))

    # Create circular mask
    mask_array = create_circular_mask(bgwidth)
    mask_clip = ImageClip(mask_array, ismask=True).set_duration(sec_bg_video.duration)

    # Apply the mask to a transparent overlay
    transparent_overlay = ColorClip((bgwidth, bgwidth), color=(0, 0, 0)).set_opacity(0).set_duration(sec_bg_video.duration)
    circular_sec_bg = CompositeVideoClip([transparent_overlay.set_mask(mask_clip), sec_bg_video], size=(bgwidth, bgwidth))

    # Calculate bottom-right position
    sec_position = (bg_w - bgwidth - margin_right, bg_h - bgwidth - margin_bottom)

    # Position the circular video and retain audio
    circular_sec_bg = circular_sec_bg.set_pos(sec_position)
    text_clip = TextClip(txt_caption, fontsize=25, color='white', bg_color='black', size=(bg_w, 100))
    text_clip = text_clip.set_pos('top',20).set_duration(5)
    # Create the final composite video
    final = CompositeVideoClip([bg_video, circular_sec_bg, text_clip])
   # final = final.set_audio(sec_bg_video.audio)

    # Write the output video with audio
    final.write_videofile(output_file, audio_codec='aac', codec="libx264")
