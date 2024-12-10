import os
import shutil
import subprocess

import PIL.Image
import cv2
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--scale", "-s", type=int, default=8, help="num imgs b/ the time segment = s-1"
# )
# args = parser.parse_args()
# scale = args.scale

def convert_to_mp4(input_file, output_file, fps, total_duration, duration, get_middle=False):
    """Convert .avi to .mp4, extracting only the middle 10 seconds (or less if the video is shorter)."""

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
    
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"{input_file.split('/')[-1]}(get_middle: {get_middle}) - Actual: {total_duration:.2f} | Extracted: {duration:.2f} seconds starting at {start_time:.2f} seconds.")


def get_fps_nimgs(video_path):
    """Get the duration of the video in seconds."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # imgs per second
    n_imgs = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # Total number of imgs
    if not fps or not n_imgs:
        print(f'{video_path} cannot be read!')
        return 0, 0
    cap.release()
    return fps, n_imgs



def extract_imgs(video_path, output_dir, fps, total_duration, duration):

    """
    Extract .png files from given video_path, 
    11/20 - Logic to obtain the middle frame is implemented by obtaining the video at first that corresponds to it
    11/25 - Extracted with default FPS when generating raw frames
    """
    start_time = 0
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
        subprocess.run(command, check=True)
    except:
        with open('UCF_fails', 'a') as f:
            f.write(f'img {video_path}')



def extract_wavs(video_path, output_dir, duration):
    """
    Extract .wav files from given video_path
    """
    
    start_time = 0
    command = [
        'ffmpeg', '-ss', str(start_time),  # Start time
        '-i', video_path,                  # Input video
        '-t', str(duration),               # Duration
        '-q:a', '0', '-map', 'a',          # Extract wavs only
        f'{output_dir}/0001.wav'           # Output wavs file
    ]
    try:
        subprocess.run(command, check=True)
    except:
        with open('UCF_fails', 'a') as f:
            f.write(f'wav {video_path}')


def setup_raw():
    """
    Given UCF-101 dataset, extract the raw frames and audio
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
        avi_vids = sorted(os.listdir(os.path.join(ucf_root, category)))    # data/UCF-101/PlayingCello
        for i, vid in enumerate(avi_vids):
            
            avi_vid = os.path.join(ucf_root, category, vid)                # data/UCF-101/PlayingCello/v_PlayingCello_g01_c02.avi
            img_dir = os.path.join(img_root, category, vid[:-4])
            wav_dir = os.path.join(wav_root, category, vid[:-4])
            if os.path.exists(img_dir):
                continue
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(wav_dir, exist_ok=True)

            fps, n_imgs = get_fps_nimgs(avi_vid)
            duration = n_imgs / fps
            extract_imgs(avi_vid, img_dir, fps, total_duration=duration, duration=duration)
            extracted_imgs = sorted(os.listdir(img_dir))
            
            # Get the start, end, and the middle frame(gt for interpolation)
            if len(extracted_imgs) % 2 == 0:
                print(f'**'*30)
                os.remove(os.path.join(img_dir, extracted_imgs[-1]))
                extracted_imgs = sorted(os.listdir(img_dir))
            save_frames = [extracted_imgs[0], extracted_imgs[len(extracted_imgs) // 2], extracted_imgs[-1]]
            for f in extracted_imgs:
                if f not in save_frames:
                    os.remove(os.path.join(img_dir, f))
            extract_wavs(avi_vid, wav_dir, duration)
            # break
        # break

def check_missing():
    """
    Finds the img and wav pair of which one or the other is missing
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
           if len(os.listdir(vid_dir_img)) < 3:
            how_failed += f' misisng imgs{len(os.listdir(vid_dir_img))}' 
           elif len(os.listdir(vid_dir_wav)) < 1:
            how_failed += ' misisng wav'
            fails.append(how_failed)
    with open('fails.txt', 'a') as f:
        for fail in fails:
            f.write(f'{fail}\n')


def remove_missing():
    """
    Removes the videos that do not include extractable audio
    """
    img_dir = '../data/UCF-101_imgs'
    wav_dir = '../data/UCF-101_wavs'
    fails = []
    for category in sorted(os.listdir(wav_dir)):
       cat_img_dir = os.path.join(img_dir, category)
       cat_wav_dir = os.path.join(wav_dir, category)
       if len(os.listdir(cat_img_dir)) == 0:
            shutil.rmtree(cat_img_dir)
            fails.append(category)
       if len(os.listdir(cat_wav_dir)) == 0:
            shutil.rmtree(cat_wav_dir)
            fails.append(cat_wav_dir)
       for vid in sorted(os.listdir(cat_img_dir)):
            vid_dir_img = os.path.join(cat_img_dir, vid)
            vid_dir_wav = os.path.join(cat_wav_dir, vid)
            if len(os.listdir(vid_dir_img)) < 3 or len(os.listdir(vid_dir_wav)) < 1:
                shutil.rmtree(vid_dir_img)
                shutil.rmtree(vid_dir_wav)      
    with open('removed_wavs', 'a') as f:
        for fail in fails:
            f.write(f'{fail}\n')
        
def create_csv():

    img_dir = '../data/UCF-101_imgs'
    wav_dir = '../data/UCF-101_wavs'
    img0s, img1s, wavs, gts = [], [], [], []
    for category in sorted(os.listdir(img_dir)):
        vids = sorted(os.listdir(os.path.join(img_dir, category)))    # [v_ApplyEyeMakeup_g01_c01, ..., v_ApplyEyeMakeup_g25_c07]
        for vid in vids:
            imgs = sorted(os.listdir(os.path.join(img_dir, category, vid)))  
            # Inputs
            img0s.append(os.path.join(img_dir, category, vid, imgs[0]))
            img1s.append(os.path.join(img_dir, category, vid, imgs[-1]))

            wavs.append(os.path.join(wav_dir, category, vid, os.listdir(os.path.join(wav_dir, category, vid))[0]))

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

def create_labels():
    file_path = "../data/UCF-101.csv"
    df = pd.read_csv(file_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Randomly shuffle the rows

    train, test = 0.8, 0.2

    n = len(df)
    train_end = int(train * n)
    # val_end = train_end + int(test * n)
    test_end = train_end + int(test * n)
    split_labels = ["train"] * train_end + ["test"] * (test_end - train_end) + ["test"] * (n - test_end)
    # split_labels = ["train"] * train_end + ["val"] * (val_end - train_end) + ["test"] * (n - val_end)
    df["split"] = split_labels
    output_file = "../data/UCF-101_split.csv"
    df.to_csv(output_file, index=False)

    print(df["split"].value_counts())

from PIL import Image
import numpy as np
import torchaudio
import torch
from torchvision import transforms

# create_labels()
# create_csv()
# from torchaudio import load
# df = pd.read_csv("../data/UCF-101.csv")

# vid_dur = {}
# sqrt_dists = []
# # print(len(df))
# for i, row in enumerate(df.iterrows()):
#     r = df.iloc[i]
#     wav, sr = load(r['A'])
#     n_samples = wav.shape[-1]
#     duration = n_samples / sr
#     id = r['A'].split('/')[-2]
#     vid_dur[id] = duration
#     sqrt_dists.append((duration - 6.8)**2)
#     # print(durations)
#     # break

# # min: 0.6792, max: 71.0008
# # variance: 13.9619
# # std: 3.7366
# # print(vid_dur)
# durations = list(vid_dur.values())
# print(f'avg_duration: {sum(durations)/len(df)}')
# # print(f'min: {min(durations):.4f}, max: {max(durations):.4f}')
# mid_ind = int(len(durations)//2)
# median = sorted(durations)[mid_ind]
# print(f'median: {median}')
# variance = sum(sqrt_dists) / len(df)
# std = variance ** (1/2)
# print(f'variance: {variance:.4f}')
# print(f'std: {std:.4f}')

# import matplotlib.pyplot as plt
# def plot_histogram(data, bins=50, xlabel='Duration (seconds)', ylabel='Frequency', title='Distribution of Audio Durations'):
#     plt.hist(data, bins=bins, alpha=0.7, edgecolor='black')
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.savefig('distr.png')

