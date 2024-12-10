import os
import cv2
import torch
import torchvision
import torch.nn.functional as F

from VFIBenchmark.metrics.VFIPS import calc_vfips
from VFIBenchmark.metrics import basicMetric as metric
# from VFIBenchmark.calc_metric import Metrics

import glob
import shutil
import warnings
import subprocess
from tools import IOBuffer, Tools
from infer_video import inf_vid
import pandas as pd

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import os
import cv2
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--scale", "-s", type=int, default=2, help="num frames b/ the time segment = s-1"
)
parser.add_argument(
    "--optimizer", type=str, default='adam'
)
parser.add_argument(
    "--lr", type=float, default=0.0002, help="num frames b/ the time segment = s-1"
)
parser.add_argument(
    "--epoch", type=int, default=0, help="num frames b/ the time segment = s-1"
)
args = parser.parse_args()

scale = args.scale



def get_video_length(video_path):
    """Get the duration of the video in seconds."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    if fps == 0:
        return 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # Total number of frames
    # print(f'frame_count: {frame_count}, fps: {fps}')
    duration = frame_count / fps  # Duration in seconds
    cap.release()
    return duration

def video_to_4d_tensor(video_path, resize_shape=None):
    """
    Convert an mp4 video into a 4D tensor (T, C, H, W).
    
    Args:
        video_path (str): Path to the .mp4 file.
        resize_shape (tuple, optional): (Height, Width) to resize frames. Default is None.
    
    Returns:
        torch.Tensor: 4D tensor of shape (T, C, H, W).
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR (OpenCV) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize if necessary
        if resize_shape is not None:
            frame = cv2.resize(frame, resize_shape)
        
        # Convert frame to a PyTorch tensor and normalize
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # (C, H, W)
        frames.append(frame_tensor)
    
    cap.release()
    
    # Stack frames into a tensor of shape (T, C, H, W)
    video_tensor = torch.stack(frames)  # (T, C, H, W)

    # print(video_tensor.shape)

    longer_video = video_tensor.repeat_interleave(4, dim=0)[:12]
    
    return longer_video



def compute_metrics(filename, dis_dir, ref_dir, dis_mp4, ref_mp4):
    
    # filename = 'output2/metrics2.csv'
    # if os.path.exists(filename):
    #     with open(filename, 'r') as f:
    #         lines = f.readlines()
    #         if 'vid_id' not in lines[0]:
    #             f.seek(0)
    #             f.write('vid_id, FLoLPIPS, VFIPS, LPIPS, PSNR, SSIM, MSSSIM\n')
    #     f.close()

    # Video metadata
    metrics = []
    cap_dis = cv2.VideoCapture(dis_mp4)
    cap_ref = cv2.VideoCapture(ref_mp4)
    metrics.append(dis_dir.split('/')[-2])
    print(f'{metrics[-1]} - dis: {cap_dis.get(cv2.CAP_PROP_FRAME_COUNT)} | ref: {cap_ref.get(cv2.CAP_PROP_FRAME_COUNT)}')

    # Video quality
    from flolpips.flolpips import calc_flolpips
    metrics.append(f'{calc_flolpips(dis_mp4, ref_mp4, scale=scale):.3f}')
    os.remove(ref_mp4)
    os.remove(dis_mp4)
    metrics.append(f'{calc_vfips(dis_dir, ref_dir):.3f}')
    metrics.append(f"{metric.calc_lpips(dis_dir, ref_dir, device='cuda'):.3f}")
    
    # Image quality
    metrics.append(f'{metric.calc_psnr(dis_dir, ref_dir):.2f}')
    metrics.append(f'{metric.calc_ssim(dis_dir, ref_dir):.3f}')
    metrics.append(f'{metric.calc_msssim(dis_dir, ref_dir):.3f}')
    
    with open(filename, 'a') as f:
    
        for i, m in enumerate(metrics):
            if i < len(metrics) - 1:
                f.write(f'{m}, ')
        f.write(f'{m}\n')
        print('Finished writing to file.')
    f.close()
    


def psnr(pred, gt, data_range=1.0, size_average=False):
    pred = torchvision.io.read_image(pred) / 255.0
    gt = torchvision.io.read_image(gt) / 255.0
    diff = (pred - gt).div(data_range)
    mse = diff.pow(2).mean(dim=(-3, -2, -1))
    psnr = -10 * torch.log10(mse + 1e-8)
    if size_average:
        return torch.mean(psnr)
    else:
        return psnr


from pytorch_msssim import ms_ssim, ssim
def ssim_(pred, gt):
    a = torchvision.io.read_image(pred).unsqueeze(0) / 255.0
    b = torchvision.io.read_image(gt).unsqueeze(0) / 255.0
    return ssim(a, b, data_range=1.0, size_average=False)


def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 

    return x

def calculate_fvd(videos1, videos2, device, model):

    
    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = {}

    # for calculate FVD, each clip_timestamp must >= 10
    for clip_timestamp in range(10, videos1.shape[-3]+1):
       
        # get a video clip
        # videos_clip [batch_size, channel, timestamps[:clip], h, w]
        videos_clip1 = videos1[:, :, : clip_timestamp]
        videos_clip2 = videos2[:, :, : clip_timestamp]

        # get FVD features
        feats1 = get_fvd_feats(videos_clip1, i3d=model, device=device)
        feats2 = get_fvd_feats(videos_clip2, i3d=model, device=device)
      
        # calculate FVD when timestamps[:clip]
        fvd_results[clip_timestamp] = frechet_distance(feats1, feats2)

    # result = {
    #     "value": fvd_results,
    #     "video_setting": videos1.shape,
    #     "video_setting_name": "batch_size, channel, time, heigth, width",
    # }

    # return torch.mean(fvd_results)
    vals = list(fvd_results.values())
    return sum(vals) / len(vals)

if __name__ == "__main__":

    
    from flolpips.flolpips import calc_flolpips
    fvd_method = 'styleganv'
    if fvd_method == 'styleganv':
        from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
    elif fvd_method == 'videogpt':
        from fvd.videogpt.fvd import load_i3d_pretrained
        from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats
        from fvd.videogpt.fvd import frechet_distance
    fvd_model = load_i3d_pretrained(device='cuda')
    epoch = 1
    gt_root = '../data/GT/scale/2'
    pervfi_root = '../data/PerVFI/scale/2'  
    asvfi_root = f'../data/ASVFI/scale/{args.scale}_{args.optimizer}_{args.lr}'   # Change the ep_# each run
    ucf_root = '../data/UCF-101_imgs'
    
    flolpips = []
    fvds = []
    psnrs = []
    ssims = []
    vfips = []
    vid_names = []
    for category in sorted(os.listdir(asvfi_root)):
        for vid_name in sorted(os.listdir(os.path.join(asvfi_root, category))):
            vid_names.append(vid_name)
            
            asvfi_dir = os.path.join(asvfi_root, category, vid_name)
            pervfi_dir = os.path.join(pervfi_root, category, vid_name, 'dis')
            gt_dir = os.path.join(gt_root, category, vid_name)
            gt_frame_dir = os.path.join(ucf_root, category, vid_name)

            asvfi_vid = os.path.join(asvfi_dir, f'{vid_name}.mp4')
            pervfi_vid = os.path.join(pervfi_dir, f'{vid_name}.mp4')
            gt_vid = os.path.join(gt_dir, f'{vid_name}.mp4')

            asvfi_frame = os.path.join(asvfi_dir, sorted(os.listdir(asvfi_dir))[1])
            pervfi_frame = os.path.join(pervfi_dir, sorted(os.listdir(pervfi_dir))[1])
            gt_frame = os.path.join(gt_frame_dir, sorted(os.listdir(gt_frame_dir))[1])

            asvf_v = get_video_length(asvfi_vid)
            per_v = get_video_length(pervfi_vid)
            gt_v = get_video_length(gt_vid)
            # print(f'pervfi_frame: {pervfi_frame}, gt_frame: {gt_frame}')
            if asvf_v != gt_v:
                print(f'Not matching{vid_name}: GT - {gt_v} asvfi_vid - {asvfi_vid}')
            # if per_v != gt_v:   # Run each model separately
                # print(f'Not matching{vid_name}: GT - {gt_v} pervfi_vid - {per_v}')
            else:
                flolpips.append(calc_flolpips(asvfi_vid, gt_vid, scale=2))
                vfips.append(calc_vfips(asvfi_dir, gt_frame_dir))
                fvds.append(calculate_fvd(video_to_4d_tensor(asvfi_vid).unsqueeze(0), video_to_4d_tensor(gt_vid).unsqueeze(0), device='cuda', model=fvd_model))
                psnrs.append(psnr(asvfi_frame, gt_frame).item())
                ssims.append(ssim_(asvfi_frame, gt_frame).item())

                # flolpips.append(calc_flolpips(pervfi_vid, gt_vid, scale=2))
                # vfips.append(calc_vfips(pervfi_dir, gt_frame_dir))
                # fvds.append(calculate_fvd(video_to_4d_tensor(pervfi_vid).unsqueeze(0), video_to_4d_tensor(gt_vid).unsqueeze(0), device='cuda', model=fvd_model))
                # psnrs.append(psnr(pervfi_frame, gt_frame).item())
                # ssims.append(ssim_(pervfi_frame, gt_frame).item())
                # print(flolpips[-1], vfips[-1], fvds[-1], psnrs[-1], ssims[-1])
        #     break
        # break

    # print(len(vid_names))
    # print(len(flolpips))
    # print(len(vfips))
    # print(len(fvds))
    # print(len(psnrs))
    # print(len(ssims))
    
    """
    Understand why Difference in FloLPIPS is relevant in PerVFI
    """
    df = pd.DataFrame({
        'Video': sorted(vid_names),
        'FloLPIPS': flolpips,
        'VFIPS': vfips,
        'FVD': fvds,
        'PSNR': psnrs,
        'SSIM': ssims
    })
    avg = {
        'Video': 'Average',
        'FloLPIPS': df['FloLPIPS'].mean(),
        'VFIPS': df['VFIPS'].mean(),
        'FVD': df['FVD'].mean(),
        'PSNR': df['PSNR'].mean(),
        'SSIM': df['SSIM'].mean(),
    }
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    df.to_csv(f'../ASVFI/evals/ASVFI_{args.optimizer}_{args.lr}.csv', index=False)
    # print(f'PerVFI: {sum(perv_flop)}, {len(perv_flop)}')
    # print(f'ASVFI: {sum(asvf_flop)}, {len(asvf_flop)}')
    
    # print(f'PerVFI: {sum(perv_flop)/len(perv_flop)}')
    # print(f'ASVFI: {sum(asvf_flop)/len(asvf_flop)}')
    # pred = f'output4/Bowling/v_Bowling_g01_c01/ref/0001.png'
    # gt = f'output4/Bowling/v_Bowling_g01_c01/dis/0001.png'

    # print(torch.sum(torchvision.io.read_image(gt)==torchvision.io.read_image(pred)))
    # print(psnr(torchvision.io.read_image(gt), torchvision.io.read_image(pred)))
    
    # pred_vid = 'input/Bowling/v_Bowling_g01_c01/0001.png'
    # gt_vid = 'output/Bowling/v_Bowling_g01_c01/ref/0001.png'
    # from flolpips.flolpips import calc_flolpips
    # print(f'{calc_flolpips(pred_vid, gt_vid):.3f}')

    # metric_csv = 'output/metrics.csv'
    # dis_dir = 'output/Bowling/v_Bowling_g01_c01/dis'
    # ref_dir = 'output/Bowling/v_Bowling_g01_c01/ref'
    # dis_mp4 = 'output/Bowling/v_Bowling_g01_c01/dis/v_Bowling_g01_c01.mp4'
    # ref_mp4 = 'output/Bowling/v_Bowling_g01_c01/ref/v_Bowling_g01_c01.mp4'
    # if os.path.exists(metric_csv):
    #     os.remove(metric_csv)    # Clear metric logging each run
    #     f = open(metric_csv, 'w')
    #     f.write('vid_id, FLoLPIPS, VFIPS, LPIPS, PSNR, SSIM, MSSSIM\n')
    #     f.close()
    # compute_metrics(metric_csv, dis_dir, ref_dir, dis_mp4, ref_mp4)
    # get_video_length('../data/UCF-101/Bowling/v_Bowling_g01_c01.avi')
    # convert_to_mp4('../data/UCF-101/Bowling/v_Bowling_g01_c01.avi', 'bowling.mp4', 25, 0, 159/25)
    # os.makedirs('bowling_frames', exist_ok=True)
    # extract_frames('../data/UCF-101/Bowling/v_Bowling_g01_c01.avi', 'bowling_frames', 25, 6.36, 6.36)
    # extract_audio('../data/UCF-101/Bowling/v_Bowling_g01_c01.avi', 'bowling.wav', 6.36)
    """
    Increase the scale and see what happens:
        - Bigger the scale, or more to generate in-bewteen the frames, the worsen it gets
        - Information gaps between the time segment, ex) bowling ball disappearing
    Next step: 
    1. Dense frames: frame at the start and the end of the vid, adjust the scales
    2. Obtain audio and run the experiment
    """
    
    # filename = 'output/metrics.csv'
    # ucf_dataroot = '../data/UCF-101'
    # # categories = ['PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PlayingDholak']
    # categories = ['Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BrushingTeeth', 'CuttingInKitchen', 'Hammering', 'HulaHoop', 'TableTennisShot', 'TennisSwing', 'ThrowDiscus']
    # for category in sorted(categories):
    #     inroot = os.path.join(f'input{scale}', category)                            # input/PlayingCello
    #     outroot = os.path.join(f'output{scale}', category)                          # output/PlayingCello
    #     avi_vids = sorted(os.listdir(os.path.join(ucf_dataroot, category)))
    #     curr_id = None
    #     for i, vid in enumerate(avi_vids):
    #         vid_id = vid.split('.')[0]
    #         n_id = vid_id.split('_')[2]
    #         print('*'*30)
    #         print(f'vid_id: {vid_id} | n_id: {n_id}')

    #         """
    #         Genereate reference video
    #         """
    #         avi_path = os.path.join(ucf_dataroot, category, vid)            # data/UCF-101/PlayingCello/v_PlayingCello_g01_c02.avi
    #         ref_path = os.path.join(outroot, vid_id, "ref")                 # output/PlayingCello/v_PlayingCello_g01_c02/ref
    #         os.makedirs(ref_path, exist_ok=True)
    #         ref_vid = os.path.join(ref_path, vid[:-4] + '.mp4')             # output/PlayingCello/v_PlayingCello_g01_c02/ref/v_PlayingCello_g01_c02.mp4

    #         total_duration = get_video_length(avi_path) # Length of ref video we set to
    #         if total_duration == 0:                     # Ignore if can't read the video
    #             shutil.rmtree(ref_path)
    #             continue
    #         int_duration = int(total_duration)          
    #         if not os.path.exists(ref_vid):
    #             # print(f'total_duration: {total_duration}')
    #             decimal_threshold = 1 / scale               # Threshold decimal point video length that picks up the frame
    #             if total_duration == int_duration:          # If video is 8 seconds, we manually trim to get the middle 7 seconds
    #                 int_duration -= 1
    #             elif (total_duration % 1) < decimal_threshold:            
    #                 int_duration -= 1
    #             convert_to_mp4(avi_path, ref_vid, fps=scale, total_duration=total_duration, duration=int_duration)
    #             print(f'Total frames in ref_vid({int_duration}): {cv2.VideoCapture(ref_vid).get(cv2.CAP_PROP_FRAME_COUNT)}')
    #         """
    #         Generate reference frames
    #             - 7 second video clip will not include the frame at 7, ex) 4 fps (0.0, 0.25, 0.5, 0.75, ... 6.75) = 4*6=24
    #         """
    #         ref_exists = ('jpg' in sorted(os.listdir(ref_path)) or 'png' in sorted(os.listdir(ref_path)))
    #         if not ref_exists:                                                # Extract frames from the ref video
    #             extract_frames(ref_vid, ref_path, fps=scale, total_duration=total_duration, duration=int_duration)
            
    #         """
    #         Generate inut frames to interpolate
    #             - Obtain the middle int(total_duration) of frames
    #             - Obtain the extra frame beyond int(duration) to interpolate the last seconds
    #         """
    #         inp_path = os.path.join(inroot, vid_id)                         # input/PlayingCello/v_PlayingCello_g01_c02
    #         os.makedirs(inp_path, exist_ok=True)
    #         inp_exists = ('jpg' in sorted(os.listdir(inp_path)) or 'png' in sorted(os.listdir(inp_path)))
    #         if not inp_exists:
    #             for i, filename in enumerate(sorted(os.listdir(ref_path))[:-2:scale]):  # -2 for excluding .mp4 and last ref frame(n.5), we want n+1.0
    #                 # Custom renaming logic
    #                 shutil.copy(
    #                     os.path.join(ref_path, filename), 
    #                     os.path.join(inp_path, f'000{i+1}.png')
    #                 )
    #             tmp_vid = os.path.join(inp_path, vid[:-4] + '.mp4')         # Utilize the original full video to obtain the last frame
    #             convert_to_mp4(avi_path, tmp_vid, fps=scale, total_duration=total_duration, duration=total_duration, get_middle=False)
    #             cap = cv2.VideoCapture(tmp_vid)
    #             print(f'cv2.VideoCapture(ref_vid).get(cv2.CAP_PROP_FRAME_COUNT): {cv2.VideoCapture(ref_vid).get(cv2.CAP_PROP_FRAME_COUNT)}')
    #             cap.set(cv2.CAP_PROP_POS_FRAMES, cv2.VideoCapture(ref_vid).get(cv2.CAP_PROP_FRAME_COUNT)+1)  # Obtain single frame after ref_vid (+0.25 after)
    #             ret, last_frame = cap.read()
    #             if ret:
    #                 cv2.imwrite(f"{inp_path}/000{i+2}.png", last_frame)
    #                 os.remove(tmp_vid)
    #                 print(f'Generated the last frame, {len(os.listdir(inp_path))}')
    #             cap.release()
    #         break
         
    #     """
    #     Interpolate the video with input frames
    #         - Generate interpolated frames(dis/0000.png) in inf_vid.py
    #     """
    #     inf_vid(dataroot=inroot, save=outroot, scale=scale)                       # output/PlayingCello/v_PlayingCello_g01_c02/dis/v_PlayingCello_g01_c02.mp4
        
    #     """
    #     Compute metrics
    #     """
    #     # for vid_id in sorted(os.listdir(outroot)):

    #     #     dis_path = os.path.join(outroot, vid_id, "dis")                    # output/PlayingCello/v_PlayingCello_g01_c02/dis
    #     #     dis_vid = os.path.join(dis_path, vid_id + '.mp4')                  # output/PlayingCello/v_PlayingCello_g01_c02/dis/v_PlayingCello_g01_c02.mp4
    #     #     os.makedirs(dis_path, exist_ok=True)
            
    #     #     ref_path = os.path.join(outroot, vid_id, "ref")                     # output/PlayingCello/v_PlayingCello_g01_c02/ref
    #     #     ref_vid = os.path.join(ref_path, f"{vid_id}.mp4")                   # output/PlayingCello/v_PlayingCello_g01_c02/ref/v_PlayingCello_g01_c02.mp4
    #     #     cap_ref = cv2.VideoCapture(ref_vid)
    #     #     cap_dis = cv2.VideoCapture(dis_vid)
    #     #     ref_frames = cap_ref.get(cv2.CAP_PROP_FRAME_COUNT)
    #     #     dis_frames = cap_dis.get(cv2.CAP_PROP_FRAME_COUNT)
    #     #     ref_vid_len = get_video_length(ref_vid)
    #     #     dis_vid_len = get_video_length(dis_vid)
    #     #     print(f'dis - {dis_frames}, {dis_vid_len} | ref - {ref_frames} {ref_vid_len}')
    #     #     compute_metrics(f'output{scale}/metrics{scale}.csv', dis_dir=dis_path, ref_dir=ref_path, dis_mp4=dis_vid, ref_mp4=ref_vid)
            
    #     break


    






 