import os
import cv2
import numpy as np
import time
from pathlib import Path
from paddleocr import TextDetection
import subprocess
import imageio_ffmpeg

FRAMES = "frames"
MASKS = "masks"

VERBOSE = 0
def printVerb(*args, **kwargs):
    if VERBOSE == 1:
        print(*args, **kwargs)

# Initialize text detector
text_detector = TextDetection(
    model_name="PP-OCRv5_server_det"
    #enable_hpi=True,
    #thresh=0.1,
    #box_thresh=0.1
)

def create_merged_mask(frame_dir):
    """
    Draw text detection regions from all frames onto a single mask image
    :param frame_dir: Path to frame directory
    :return: Path to saved merged mask
    """
    # Create mask directory
    mask_dir = frame_dir.replace(FRAMES, MASKS)
    os.makedirs(mask_dir, exist_ok=True)

    # Get and sort all frame files
    file_list = sorted(os.listdir(frame_dir))
    total_frames = len(file_list)
    print(f"📌 frame_dir:{frame_dir} total frame: {total_frames} ...")

    # Initialize merged mask (set to None, determine size after reading first frame)
    merged_mask = None
    total_draw_count = 0  # Count total detected text regions

    # Traverse all frames
    for idx, f in enumerate(file_list):
        fp = os.path.join(frame_dir, f).replace("\\", "/")
        img = cv2.imread(fp)

        # Skip invalid frame
        if img is None:
            print(f"⚠️  Frame {idx+1}：{f} - Invalid image, skipped")
            continue

        # Initialize merged mask with the size of first valid frame
        if merged_mask is None:
            h, w = img.shape[:2]
            merged_mask = np.zeros((h, w), dtype=np.uint8)
            print(f"📏 Mask initialized with size: width={w}, height={h}")

        frame_draw_count = 0  # Count text regions in current frame
        try:
            # Text region detect
            startT = time.time()
            det_results = text_detector.predict(fp)
            det_time = (time.time() - startT) * 1000

            region_info = ""
            if det_results:
                dt_polys = det_results[0].get("dt_polys", [])
                dt_scores = det_results[0].get("dt_scores", [])

                # Traverse each text region in current frame
                for poly_idx, poly in enumerate(dt_polys):
                    score = dt_scores[poly_idx] if poly_idx < len(dt_scores) else 0.0
                    # only draw when score > 0.9
                    if score <= 0.9:
                        printVerb(f"⚠️  Frame {idx+1} Region {poly_idx+1} - Skip (score={score:.4f} ≤ 0.9)")
                        continue

                    # Make sure poly coordinates are within image bounds
                    points = []
                    h, w = merged_mask.shape[:2]
                    for (x, y) in poly:
                        x_clipped = max(0, min(w - 1, int(x)))
                        y_clipped = max(0, min(h - 1, int(y)))
                        points.append([x_clipped, y_clipped])

                    # Draw polygon on merged mask
                    if len(points) >= 4:
                        points_np = np.array(points, np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(merged_mask, [points_np], 255)
                        frame_draw_count += 1
                        total_draw_count += 1

                        # Print single region info
                        region_info = f"Region:{poly_idx+1} | Confidence:{score:.4f} | "

                        log_str = f"📊 Frame {idx+1}: {f} - {region_info} Cost: {det_time:.2f}ms | Region count：{frame_draw_count}"
                        print(log_str, end="\r", flush=True)
                        #print(log_str)

        except Exception as e:
            print(f"❌ Frame {idx+1}：{f} - Error: {str(e)}")

    # Save merged mask (named mask.png as requested)
    if merged_mask is not None:
        merged_mask_path = os.path.join(mask_dir, "mask.png")
        cv2.imwrite(merged_mask_path, merged_mask)
        printVerb(f"\n📝 Merged mask saved to：{merged_mask_path}")
        printVerb(f"📈 Total detected text regions：{total_draw_count}")
    else:
        print("\n❌ No valid frames found, cannot generate mask")
        merged_mask_path = None

    return mask_dir

def extract_frames(video_path, output_dir="./frames"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_bin,
            "-i", video_path,
            os.path.join(output_dir, "frame_%06d.png"),
            "-hide_banner",
            "-loglevel", "error"
        ]

        printVerb("run FFmpeg extract frame:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        frame_files = [f for f in os.listdir(output_dir) if f.startswith("frame_")]
        if len(frame_files) == 0:
            print(f"❌ no frames：{video_path}")
            return False
        
        print(f"✅ parse frame: {video_path} → {len(frame_files)}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg fail：{e.stderr}")
        return False
    except Exception as e:
        print(f"❌ parse frame error：{str(e)}")
        return False

def process_all_videos(input_videos_dir="./input_videos"):
    """
    Process all video files in input directory: extract frames → generate mask
    :param input_videos_dir: Directory containing input videos
    """
    # Check input directory
    if not os.path.isdir(input_videos_dir):
        print(f"❌ Input videos directory not found: {input_videos_dir}")
        return
    
    # Supported video extensions (can be extended)
    supported_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
    
    # Get all video files
    video_files = [
        f for f in os.listdir(input_videos_dir)
        if f.lower().endswith(supported_extensions)
    ]
    
    if not video_files:
        print(f"ℹ️ No video files found in {input_videos_dir} (supported: {supported_extensions})")
        return
    
    print(f"📽️ Found {len(video_files)} video(s) to process:\n" + "\n".join([f"  - {f}" for f in video_files]))
    print("-" * 50)
    
    # Process each video
    for video_file in video_files:
        try:
            # Full path to video
            video_path = os.path.join(input_videos_dir, video_file)
            # Video name (without extension) for directory naming
            video_name = Path(video_file).stem
            # Frame output directory
            frame_dir = os.path.join(FRAMES, video_name)
            
            print(f"\n📌 Processing video: {video_file}")
            printVerb(f"   Video name: {video_name}")
            printVerb(f"   Frame directory: {frame_dir} video_path: {video_path}")
            
            # Step 1: Extract frames with imageio-ffmpeg
            if not extract_frames(video_path, frame_dir):
                print(f"⚠️ Skipping mask generation for {video_file} due to frame extraction failure")
                continue
            
            # Step 2: Generate merged mask (keep original function structure)
            print(f"\n🔍 Detecting text regions and generating mask for {video_name}...")
            mask_dir = create_merged_mask(frame_dir)
            print(f"✅ Mask generated for {video_name}, saved to: {mask_dir}")
            print("-" * 50)
            
        except Exception as e:
            print(f"\n❌ Failed to process {video_file}: {str(e)}")
            print("-" * 50)
            continue
    
    print("\n🎉 All videos processed!")

if __name__ == "__main__":
    # Process all videos in input_videos directory
    process_all_videos()
