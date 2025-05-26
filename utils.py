import os
import shutil
import subprocess

import PIL.Image
import cv2
import pandas as pd

# import argparse # Removed unused argparse
# parser = argparse.ArgumentParser()
# # parser.add_argument(
# # "--scale", "-s", type=int, default=8, help="num imgs b/ the time segment = s-1"
# # )
# # args = parser.parse_args()
# # scale = args.scale


def convert_to_mp4(input_file, output_file, fps, total_duration, duration, get_middle=False):
    """
    Converts an input video file (typically .avi) to .mp4 format using ffmpeg.
    Optionally extracts a specific segment from the middle of the video.

    Args:
        input_file (str): Path to the input video file.
        output_file (str): Path to save the output .mp4 video file.
        fps (float or int): Frames per second for the output video.
        total_duration (float): Total duration of the input video in seconds.
        duration (float): Duration of the segment to extract in seconds.
                          If `get_middle` is False, this is the total duration of the output.
        get_middle (bool, optional): If True, extracts `duration` seconds from the
                                     middle of the input video. Otherwise, extracts from
                                     the beginning. Defaults to False.
    """
    if get_middle:
        start_time = max(0, (total_duration / 2) - (duration / 2))
    else:
        start_time = 0

    command = [
        'ffmpeg',
        '-i', input_file,
        '-ss', str(start_time),
        '-t', str(duration),
        '-r', str(fps),
        '-c:v', 'libx264',            # Video codec for .mp4
        '-crf', '23',                 # Constant rate factor for quality
        '-preset', 'fast',            # Encoding speed preset
        '-c:a', 'aac',                # wavs codec (AAC for .mp4)
        '-b:a', '192k',               # wavs bitrate
        '-strict', 'experimental',    # Required for AAC in some FFmpeg versions
        output_file                   # Output .mp4 file
    ]

    try:
        subprocess.run(command, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(
            f"{input_file.split('/')[-1]}(get_middle: {get_middle}) - Actual: {total_duration:.2f} | Extracted: {duration:.2f} seconds starting at {start_time:.2f} seconds.")
    except subprocess.CalledProcessError as e:
        print(
            f"Error converting {input_file} to mp4: {e.stderr.decode('utf-8')}")
        # Decide if you want to raise the exception, log to a file, or handle otherwise
        # For now, just printing the error. Consider adding to UCF_fails or a similar log.


def get_fps_nimgs(video_path):
    """
    Retrieves the frames per second (fps) and total number of frames of a video.

    Args:
        video_path (str): Path to the video file.

    Returns:
        tuple[float, int]: A tuple containing:
            - fps (float): Frames per second of the video.
            - n_imgs (int): Total number of frames in the video.
            Returns (0, 0) if the video cannot be read.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    n_imgs = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # Total number of frames
    if not fps or not n_imgs:
        print(f'{video_path} cannot be read!')
        return 0, 0
    cap.release()
    return fps, n_imgs


def extract_imgs(video_path, output_dir, fps, total_duration, duration):
    """
    Extracts frames from a video and saves them as .png files.

    Note:
        - Logic for obtaining a middle frame (if needed for specific tasks like
          interpolation) should ideally be handled by adjusting `start_time` and
          `duration` in the calling function or by selecting specific frames
          after extraction.
        - This function currently extracts all frames at the specified FPS.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted .png frames.
        fps (float or int): The frame rate at which to extract frames.
        total_duration (float): Total duration of the input video (currently unused
                                in the ffmpeg command logic but kept for context).
        duration (float): Duration of the video segment to extract frames from
                          (currently unused as ffmpeg extracts all frames based on fps).
    """
    # start_time = 0 # Currently unused, ffmpeg extracts all frames
    command = [
        'ffmpeg',
        # '-ss', str(start_time),  # Start time
        '-i', video_path,          # Input video file
        '-vf', f'fps={fps}',     # Frame rate (e.g., 5 imgs per second)
        # '-t', str(duration),     # Duration of frame extraction
        f'{output_dir}/%04d.png'   # Output pattern for imgs
    ]
    # Run the command
    try:
        subprocess.run(command, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        with open('UCF_fails', 'a') as f:
            f.write(f'img {video_path} - Error: {e.stderr.decode("utf-8")}\n')


def extract_wavs(video_path, output_dir, duration):
    """
    Extracts the audio track from a video and saves it as a .wav file.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted .wav audio file.
                          The output filename will be '0001.wav'.
        duration (float): Duration of the audio segment to extract in seconds.
    """
    start_time = 0  # Extracts from the beginning for the specified duration
    command = [
        'ffmpeg',
        '-ss', str(start_time),    # Start time for audio extraction
        '-i', video_path,          # Input video file
        '-t', str(duration),       # Duration of audio to extract
        '-q:a', '0',               # Highest quality audio
        '-map', 'a',               # Select only audio stream
        f'{output_dir}/0001.wav'   # Output .wav file path
    ]
    try:
        subprocess.run(command, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        with open('UCF_fails', 'a') as f:
            f.write(f'wav {video_path} - Error: {e.stderr.decode("utf-8")}\n')


def setup_raw():
    """
    Processes the UCF-101 dataset to extract specific raw frames (first, middle, last)
    and the corresponding audio track for each video.

    The function iterates through video categories and individual videos in the
    specified `ucf_root` directory. For each video, it:
    1. Retrieves FPS and total frame count.
    2. Extracts all frames initially using `extract_imgs`.
    3. Identifies and keeps only the first, middle, and last frames. If the total
       number of frames is even, the last frame before true middle is removed to ensure
       an odd number for a clear middle frame.
    4. Extracts the audio using `extract_wavs`.
    5. Skips processing if the image directory for a video already exists.

    Output structure:
        - `img_root/<category>/<video_name_without_ext>/`: Contains 0001.png, middle.png, last.png
        - `wav_root/<category>/<video_name_without_ext>/`: Contains 0001.wav
    """
    ucf_root = '../data/UCF-101'
    img_root = '../data/UCF-101_imgs'
    wav_root = '../data/UCF-101_wavs'
    for category in sorted(os.listdir(ucf_root)):
        # categories = ['PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PlayingDholak']
        # categories = ['Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BrushingTeeth', 'CuttingInKitchen', 'Hammering', 'HulaHoop', 'TableTennisShot', 'TennisSwing', 'ThrowDiscus']
        if '_' in category:     # ignore the metadata file
            continue
        # if 'Lunges' in category:    # Resume from certain category
        #     flag = True
        # elif not flag:
        #     continue
        # data/UCF-101/PlayingCello
        avi_vids = sorted(os.listdir(os.path.join(ucf_root, category)))
        for i, vid in enumerate(avi_vids):

            # data/UCF-101/PlayingCello/v_PlayingCello_g01_c02.avi
            avi_vid = os.path.join(ucf_root, category, vid)
            img_dir = os.path.join(img_root, category, vid[:-4])
            wav_dir = os.path.join(wav_root, category, vid[:-4])
            if os.path.exists(img_dir):
                continue
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(wav_dir, exist_ok=True)

            fps, n_imgs = get_fps_nimgs(avi_vid)
            if fps == 0:
                print(
                    f"Error: FPS is 0 for video {avi_vid}. Skipping this video.")
                with open('UCF_fails', 'a') as f:
                    f.write(f'fps_zero {avi_vid}\n')
                continue  # Skip to the next video
            duration = n_imgs / fps
            extract_imgs(avi_vid, img_dir, fps,
                         total_duration=duration, duration=duration)
            extracted_imgs = sorted(os.listdir(img_dir))

            # Get the start, end, and the middle frame(gt for interpolation)
            if len(extracted_imgs) % 2 == 0:
                print(f'**'*30)
                os.remove(os.path.join(img_dir, extracted_imgs[-1]))
                extracted_imgs = sorted(os.listdir(img_dir))
            save_frames = [extracted_imgs[0], extracted_imgs[len(
                extracted_imgs) // 2], extracted_imgs[-1]]
            for f in extracted_imgs:
                if f not in save_frames:
                    os.remove(os.path.join(img_dir, f))
            extract_wavs(avi_vid, wav_dir, duration)
            # break
        # break


def check_missing():
    """
    Checks for missing image or wave files in the processed UCF-101 dataset.

    It iterates through the processed image directories and checks if each
    corresponding video instance has:
        - At least 3 images (expected: first, middle, last).
        - At least 1 wave file.
    If any are missing, it logs the video identifier and the type of missing
    data to 'fails.txt'.
    """
    img_dir = '../data/UCF-101_imgs'
    wav_dir = '../data/UCF-101_wavs'
    fails = []
    for category in sorted(os.listdir(img_dir)):
        cat_img_dir = os.path.join(img_dir, category)
        cat_wav_dir = os.path.join(wav_dir, category)
        for vid in sorted(os.listdir(cat_img_dir)):
            vid_dir_img = os.path.join(cat_img_dir, vid)
            vid_dir_wav = os.path.join(cat_wav_dir, vid)
            how_failed = f'{vid}'
            try:
                if not os.path.exists(vid_dir_img) or len(os.listdir(vid_dir_img)) < 3:
                    how_failed += f' missing imgs (found {len(os.listdir(vid_dir_img)) if os.path.exists(vid_dir_img) else 0})'
                    fails.append(how_failed)
                elif not os.path.exists(vid_dir_wav) or len(os.listdir(vid_dir_wav)) < 1:
                    how_failed += ' missing wav'
                    fails.append(how_failed)
            except FileNotFoundError:
                how_failed += ' directory not found'
                fails.append(how_failed)
    if fails:
        with open('fails.txt', 'a') as f:
            for fail in fails:
                f.write(f'{fail}\n')


def remove_missing():
    """
    Removes directories for videos that have missing image or audio files
    based on the criteria used in `setup_raw` (needs 3 images, 1 wav).

    It iterates through categories in the WAV directory (can also be IMG, assuming they are parallel).
    - Removes empty category directories in both `img_dir` and `wav_dir`.
    - For each video, if the image directory has fewer than 3 files or the
      wave directory has fewer than 1 file, it removes both the image and
      wave directories for that video.
    Logs removed category or video directories to 'removed_wavs' (consider renaming this log file).
    """
    img_dir = '../data/UCF-101_imgs'
    wav_dir = '../data/UCF-101_wavs'
    removed_items_log = []  # Renamed 'fails' to be more descriptive

    # Iterate based on wav_dir, could be img_dir too
    for category in sorted(os.listdir(wav_dir)):
        cat_img_dir = os.path.join(img_dir, category)
        cat_wav_dir = os.path.join(wav_dir, category)

        # Check and remove empty category directories
        if os.path.exists(cat_img_dir) and not os.listdir(cat_img_dir):
            shutil.rmtree(cat_img_dir)
            removed_items_log.append(
                f"Removed empty img category: {cat_img_dir}")
        if os.path.exists(cat_wav_dir) and not os.listdir(cat_wav_dir):
            shutil.rmtree(cat_wav_dir)
            removed_items_log.append(
                f"Removed empty wav category: {cat_wav_dir}")

        # Check individual videos if category directories still exist
        if os.path.exists(cat_img_dir) and os.path.exists(cat_wav_dir):
            # Iterate images, assume wav is parallel
            for vid in sorted(os.listdir(cat_img_dir)):
                vid_dir_img = os.path.join(cat_img_dir, vid)
                # Corresponding wav video directory
                vid_dir_wav = os.path.join(cat_wav_dir, vid)

                missing_imgs = not os.path.exists(
                    vid_dir_img) or len(os.listdir(vid_dir_img)) < 3
                missing_wav = not os.path.exists(
                    vid_dir_wav) or len(os.listdir(vid_dir_wav)) < 1

                if missing_imgs or missing_wav:
                    if os.path.exists(vid_dir_img):
                        shutil.rmtree(vid_dir_img)
                        removed_items_log.append(
                            f"Removed img video: {vid_dir_img}")
                    if os.path.exists(vid_dir_wav):
                        shutil.rmtree(vid_dir_wav)
                        removed_items_log.append(
                            f"Removed wav video: {vid_dir_wav}")

    if removed_items_log:
        with open('removed_items_log.txt', 'a') as f:  # Changed log file name
            for item in removed_items_log:
                f.write(f'{item}\n')


def create_csv():
    """
    Creates a CSV file ('../data/UCF-101.csv') that catalogs the paths to
    the first frame (I0), last frame (I1), audio file (A), and ground truth
    middle frame (GT) for each processed video in the UCF-101 dataset.

    It assumes the directory structure and file naming conventions established
    by the `setup_raw` function:
        - Images: `img_dir/<category>/<video_id>/<frame_name>.png`
                  (expects 3 frames, typically 0001.png, middle.png, last.png)
        - Audio: `wav_dir/<category>/<video_id>/0001.wav`
    """
    img_dir = '../data/UCF-101_imgs'
    wav_dir = '../data/UCF-101_wavs'
    img0s, img1s, wavs, gts = [], [], [], []
    for category in sorted(os.listdir(img_dir)):
        # [v_ApplyEyeMakeup_g01_c01, ..., v_ApplyEyeMakeup_g25_c07]
        vids = sorted(os.listdir(os.path.join(img_dir, category)))
        for vid in vids:
            imgs = sorted(os.listdir(os.path.join(img_dir, category, vid)))
            # Inputs
            img0s.append(os.path.join(img_dir, category, vid, imgs[0]))
            img1s.append(os.path.join(img_dir, category, vid, imgs[-1]))

            wavs.append(os.path.join(wav_dir, category, vid, os.listdir(
                os.path.join(wav_dir, category, vid))[0]))

            # Ground Truth
            gts.append(os.path.join(img_dir, category, vid, imgs[1]))

    data = {
        "I0": img0s,
        "I1": img1s,
        "A": wavs,
        "GT": gts
    }
    df = pd.DataFrame(data)
    csv_path = "../data/UCF-101.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV file created at {csv_path} with {len(df)} entries.")


def create_labels():
    """
    Reads the UCF-101 dataset CSV (created by `create_csv`), shuffles it,
    and adds a 'split' column to designate each entry as 'train' or 'test'.

    The split ratio is 80% for training and 20% for testing.
    The shuffled DataFrame with the new 'split' column is saved to
    '../data/UCF-101_split.csv'.
    Prints the value counts of the 'split' column.
    """
    file_path = "../data/UCF-101.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(
            f"Error: CSV file not found at {file_path}. Please run create_csv() first.")
        return

    df = df.sample(frac=1, random_state=42).reset_index(
        drop=True)  # Randomly shuffle the rows

    train, test = 0.8, 0.2

    n = len(df)
    train_end = int(train * n)
    test_end = train_end + int(test * n)
    split_labels = ["train"] * train_end + ["test"] * \
        (test_end - train_end) + ["test"] * (n - test_end)
    df["split"] = split_labels
    output_file = "../data/UCF-101_split.csv"
    df.to_csv(output_file, index=False)

    print(df["split"].value_counts())

# Removed PIL, numpy, torchaudio, torch, torchvision imports as they are not used after removing the data analysis block.
# If these are needed for other parts of the project, they should be re-added where appropriate.

# Removed create_labels() and create_csv() calls as they are likely part of a script execution flow, not general utility.
# Removed the extensive data analysis block (df.iterrows(), plotting histograms, etc.)
