import os
import sys
import time
import subprocess
from pathlib import Path
import shutil
import tempfile
import imageio_ffmpeg

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from create_mask import process_all_videos as generate_video_masks, FRAMES, MASKS

# Configuration
INPUT_VIDEOS_DIR = ".\\input_videos"
OUTPUT_VIDEOS_DIR = ".\\output_videos"
PROPINTER_SCRIPT = ".\\ProPainter\\inference_propainter.py"
USE_FP16 = True
DELETE_TMP = 1
TMP_DIRS = ["results", "frames", "masks"]

def check_dependencies():
    """Check required dependencies"""
    if not os.path.exists(PROPINTER_SCRIPT):
        print(f"Error: ProPainter script not found - {PROPINTER_SCRIPT}")
        return False
    if not os.path.exists(INPUT_VIDEOS_DIR):
        print(f"Error: Input video directory not found - {INPUT_VIDEOS_DIR}")
        return False
    os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)
    return True

def run_propainter(video_path, mask_path, output_path):
    cmd = [
        "python", os.path.join("ProPainter/inference_propainter.py"),
        "--video", video_path,
        "--mask", mask_path,
        "--fp16"
    ]

    temp_audio_path = None

    try:
        startT = time.time()
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        gen_time = time.time() - startT
        print(f"cmd: {cmd} cost: {gen_time:.2f}s")

        if result.returncode == 0:
            video_basename = os.path.basename(video_path)
            video_name = os.path.splitext(video_basename)[0]
            default_output = os.path.join("results", video_name, "inpaint_out.mp4")

            if not os.path.exists(default_output):
                print(f"Error: ProPainter output not found: {default_output}")
                return False

            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            temp_audio_path = os.path.join(tempfile.gettempdir(), f"temp_audio_{os.urandom(4).hex()}.aac")

            # 尝试提取音频，不管成功失败都继续
            print("Extracting audio (if any)...")
            extract_cmd = [
                ffmpeg_bin, "-y",
                "-i", video_path,
                "-vn", "-acodec", "aac", "-b:a", "192k",
                temp_audio_path
            ]
            audio_ok = False
            ar = subprocess.run(extract_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if ar.returncode == 0 and os.path.exists(temp_audio_path):
                audio_ok = True

            if audio_ok:
                print("Merging audio...")
                merge_cmd = [
                    ffmpeg_bin, "-y",
                    "-i", default_output,
                    "-i", temp_audio_path,
                    "-c:v", "copy", "-c:a", "aac", "-shortest",
                    output_path
                ]
                merge_result = subprocess.run(merge_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                if merge_result.returncode == 0:
                    print(f"Success: saved with audio → {output_path}")
                else:
                    shutil.copy(default_output, output_path)
                    print(f"Success: saved without audio → {output_path}")
            else:
                # 无音频 或 提取失败，直接拷贝
                shutil.copy(default_output, output_path)
                print(f"Success: saved without audio → {output_path}")

            return True

        else:
            err = result.stderr.decode('utf-8', 'ignore')
            print(f"ProPainter failed: {err}")
            return False

    except Exception as e:
        print(f"Exception: {str(e)}")
        return False

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass

def process_video():
    # Step 1: Generate masks
    print("="*50)
    print("Step 1: Generating video masks")
    print("="*50)
    generate_video_masks(INPUT_VIDEOS_DIR)

    # Step 2: Watermark removal
    print("\n" + "="*50)
    print("Step 2: Running ProPainter for watermark removal")
    print("="*50)
    
    supported_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
    video_files = [f for f in os.listdir(INPUT_VIDEOS_DIR) if f.lower().endswith(supported_extensions)]

    if not video_files:
        print(f"Info: No supported video files found in {INPUT_VIDEOS_DIR}")
        return

    for video_file in video_files:
        try:
            video_name = Path(video_file).stem
            video_path = os.path.join(INPUT_VIDEOS_DIR, video_file)
            
            # Get mask path
            mask_dir = os.path.join(MASKS, video_name)
            mask_path = os.path.join(mask_dir, "mask.png")
            if not os.path.exists(mask_path):
                print(f"Warning: Mask file not found for {video_file}, skip watermark removal")
                continue
            
            # Build output path
            video_ext = Path(video_file).suffix
            output_video_name = f"{video_name}_no_watermark{video_ext}"
            output_path = os.path.join(OUTPUT_VIDEOS_DIR, output_video_name)
            
            # Run watermark removal
            print(f"Processing: {video_file}")
            success = run_propainter(video_path, mask_path, output_path)
            
            if success:
                print(f"Success: Watermark removed - {output_video_name}")
            else:
                print(f"Warning: Watermark removal failed for {video_file}")

        except Exception as e:
            print(f"Error: Failed to process {video_file} - {str(e)}")

    print("\nProcess completed!")
    print(f"Output videos saved to: {OUTPUT_VIDEOS_DIR}")

    if DELETE_TMP == 1:
        print("🗑️  clean tmp data...")
        for dir_name in TMP_DIRS:
            try:
                if os.path.exists(dir_name):
                    shutil.rmtree(dir_name)
                    os.makedirs(dir_name, exist_ok=True)
                    print(f"✅ clean {dir_name} done")
            except Exception as e:
                print(f"⚠️  clean {dir_name} fail: {str(e)}")

if __name__ == "__main__":
    if not check_dependencies():
        sys.exit(1)
    process_video()

