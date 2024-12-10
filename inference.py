import os
import cv2
import torch
import torchvision
import torch.nn.functional as F

# from VFIBenchmark.metrics.VFIPS import calc_vfips
# from VFIBenchmark.metrics.flolpips import calc_flolpips
# from VFIBenchmark.metrics import basicMetric as metric

import glob
import shutil
import warnings
import subprocess
from tools import IOBuffer, Tools
import pandas as pd

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


import os
import cv2
import subprocess
import argparse
from utils import get_fps_nimgs

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

# def compute_metrics(filename, dis_dir, ref_dir, dis_mp4, ref_mp4):
    
#     # filename = 'output2/metrics2.csv'
#     # if os.path.exists(filename):
#     #     with open(filename, 'r') as f:
#     #         lines = f.readlines()
#     #         if 'vid_id' not in lines[0]:
#     #             f.seek(0)
#     #             f.write('vid_id, FLoLPIPS, VFIPS, LPIPS, PSNR, SSIM, MSSSIM\n')
#     #     f.close()

#     # Video metadata
#     metrics = []
#     cap_dis = cv2.VideoCapture(dis_mp4)
#     cap_ref = cv2.VideoCapture(ref_mp4)
#     metrics.append(dis_dir.split('/')[-2])
#     print(f'{metrics[-1]} - dis: {cap_dis.get(cv2.CAP_PROP_FRAME_COUNT)} | ref: {cap_ref.get(cv2.CAP_PROP_FRAME_COUNT)}')

#     # Video quality
#     from flolpips.flolpips import calc_flolpips
#     metrics.append(f'{calc_flolpips(dis_mp4, ref_mp4):.3f}')
    
#     os.remove(ref_mp4)
#     os.remove(dis_mp4)
#     metrics.append(f'{calc_vfips(dis_dir, ref_dir):.3f}')

#     metrics.append(f"{metric.calc_lpips(dis_dir, ref_dir, device='cuda'):.3f}")
    
#     # Image quality
#     metrics.append(f'{metric.calc_psnr(dis_dir, ref_dir):.2f}')
#     metrics.append(f'{metric.calc_ssim(dis_dir, ref_dir):.3f}')
#     metrics.append(f'{metric.calc_msssim(dis_dir, ref_dir):.3f}')
    
#     with open(filename, 'a') as f:
    
#         for i, m in enumerate(metrics):
#             if i < len(metrics) - 1:
#                 f.write(f'{m}, ')
#         f.write(f'{m}\n')
#         print('Finished writing to file.')
#     f.close()
    

def psnr(pred, gt, data_range=1.0, size_average=False):
    diff = (pred - gt).div(data_range)
    mse = diff.pow(2).mean(dim=(-3, -2, -1))
    psnr = -10 * torch.log10(mse + 1e-8)
    if size_average:
        return torch.mean(psnr)
    else:
        return psnr
    
from torchvision import transforms as img_transform
from PIL import Image
from torchaudio import transforms, load
from torchaudio.prototype.pipelines import VGGISH
import torch.nn.functional as F
def transform_img(imgs):

    img_transforms = img_transform.Compose([
        img_transform.CenterCrop((224,224)),
        img_transform.ToTensor(),
    ])
    def format_vid(imgs):
        """
        Perform weighted interpolation to match the number of timestep as audio feature
        """
        T = 17
        frames = torch.stack((imgs), dim=1)
        interpolated_frames = F.interpolate(
            frames.unsqueeze(0),  
            size=(17, 224, 224),  
            mode='trilinear',  # Trilinear interpolation for temporal and spatial consistency
            align_corners=True
        ).squeeze(0)
        return interpolated_frames
    vid = format_vid([img_transforms(Image.open(img)) for img in imgs])
    # print(f'vid: {vid.shape}')
    return vid

def transform_wav(wav):

    wav_extractor = VGGISH.get_input_processor()
    SR=44100
    audio, sr = load(wav)
    def format_wav(audio, sr):
        if sr != SR:        # Resample the sampling rate to SR=44100
            audio = transforms.Resample(sr, SR)

        target_sample = 6 * SR
        n_samples = audio.shape[-1]

        if n_samples < target_sample:   # Pad if shorter
            padding = target_sample - n_samples
            audio = F.pad(audio, (0, padding))
        else:                           # Trim if longer
            audio = audio[:, :target_sample]
        mfcc = wav_extractor(audio[0])
        return mfcc

    mfcc = format_wav(audio, SR)
    return mfcc    


def load_model(weight_path):
    print("Building ASVFI model...")
    from model import ASVFI
    model = ASVFI().cuda().eval()
    checkpoint = torch.load(weight_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Done")
    return model


def load_tests():
    import pandas as pd
    test_set = pd.read_csv('../data/UCF-101_split.csv')
    test_set = test_set[test_set['split']=='test']
    result_list = test_set['I0'].to_list()
    # Select the test set categories from UCF101
    categories = {}
    for r in result_list:
        splits = r.split('/')
        if splits[3] not in categories.keys():
            categories[splits[3]] = []
        categories[splits[3]].append(splits[4])
    categories = {key: sorted(value) for key, value in sorted(categories.items())}

    return categories

def write_frames(model, dir_read_from, dir_write_to):
    """
    Process a single video:
    1. Read I0, I1 from dir_read_from
    2. Write I0, I0.5, I1 to dir_write_to
    """
    source_frames = [os.path.join(dir_read_from, img) for img in sorted(os.listdir(dir_read_from))[::2]] # [I0, I1]
    target_frames = [os.path.join(dir_write_to, "{:04}.png".format(i)) for i in [1, 3]]
    # Write I0 and I1
    if not os.path.exists(target_frames[0]):
        for i in range(2):
            tmp_img = Image.open(source_frames[i])
            tmp_img.save(target_frames[i])

    # Write I0.5
    target_w, target_h = Image.open(source_frames[0]).size
    wav = source_frames[0].replace('_imgs', '_wavs').replace('png', 'wav')
    vid = transform_img(source_frames).unsqueeze(0).cuda()
    mfcc = transform_wav(wav).unsqueeze(0).cuda()
    with torch.no_grad():
        pred = model(vid, mfcc)[0][0]   # Later - Modify it to scale the output frames by x4, x8 
        from torchvision.transforms import functional as TF
        resized = TF.resize(pred, size=[target_h, target_w])
        resized_image = resized.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray((resized_image*255).astype('uint8'))
        pred_save_path = os.path.join(dir_write_to, "{:04}.png".format(2))
        image.save(pred_save_path)

def pred_tests():
    model = load_model(weight_path=f'weights/asvfi/{args.optimizer}_{args.lr}.pth')
    data_root = '../data/ASVFI'
    inroot = '../data/UCF-101_imgs'
    outroot = os.path.join(data_root, 'scale', f'{args.scale}_{args.optimizer}_{args.lr}')
    categories = load_tests()
    for category in sorted(categories.keys()):
        for video_name in categories[category]:
            category_inroot = os.path.join(inroot, category, video_name)
            category_outroot = os.path.join(outroot, category, video_name)
            os.makedirs(category_outroot, exist_ok=True)
            if os.path.exists(os.path.join(category_outroot, '0002.png')):
                print(f'Skipping {video_name}')
                continue
            write_frames(model, category_inroot, category_outroot)
        #     break
        # break




def construct_mp4():
    root = f'../data/ASVFI/scale/{args.scale}_{args.optimizer}_{args.lr}'
    # root = '../data/UCF-101_imgs'
    for category in sorted(os.listdir(root)):
        for vid_name in sorted(os.listdir(os.path.join(root, category))):
            dis_dir = os.path.join(root, category, vid_name)
            vid = os.path.join(dis_dir, "*.png")
            """
            Constructed a single mp4 bc not matched # of frames
            """
            # if dis_dir == '../data/UCF-101_imgs/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c04':
            #     mp4_path = '../data/GT/scale/2/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c04/v_ApplyEyeMakeup_g01_c04.mp4'
            #     target_root = '../data/GT/scale/2'
            #     target_dir = os.path.join(target_root, category, vid_name)
            #     os.makedirs(target_dir, exist_ok=True)
            #     vid_path = os.path.join(target_dir, f"{vid_name}.mp4")
            #     print(f'target_dir: {target_dir}')
            #     print(f'vid_path: {vid_path}')
            #     Tools.frames2mp4(vid, vid_path, 2)
            #     break
            #     if os.path.exists(vid_path):
            #         continue
            # target_root = '../data/GT/scale/2'
            target_root = f'../data/ASVFI/scale/{args.scale}_{args.optimizer}_{args.lr}'
            target_dir = os.path.join(target_root, category, vid_name)
            os.makedirs(target_dir, exist_ok=True)
            vid_path = os.path.join(target_dir, f"{vid_name}.mp4")
            if os.path.exists(vid_path):
                continue
            Tools.frames2mp4(vid, vid_path, 2)
        #     break
        # break
        """
        Compute FloLPIPS on dis and ref PerVFI
        """


if __name__ == "__main__":

    """
    Manually check the alignment of frames
    """
    pred_tests()
    construct_mp4()
    # pred = f'../data/PerVFI/scale/2/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/dis/0003.png'
    # gt = f'../data/UCF-101_imgs/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/0165.png'
    # print(torch.sum(torchvision.io.read_image(gt)==torchvision.io.read_image(pred)))
    # print(psnr(torchvision.io.read_image(gt), torchvision.io.read_image(pred)))
    
    

        #     break

    #     # break
    #         # category_outroot = os.path.join(outroot, category)
    #         # if os.path.exists(category_outroot):
    #         #     print(categories[category])
    #         #     if len(os.listdir(category_inroot)) == len(os.listdir(category_outroot)):
    #         #         continue
    #         #     else:   # Resume from the aborted category
    #         #         inf_vid(model=model, vids=categories[category], dataroot=category_inroot, save=category_outroot, scale=scale)
    #         # else:
    #         #     inf_vid(model=model, vids=categories[category], dataroot=category_inroot, save=category_outroot, scale=scale)



    # # print(result_list)
    # # print(categories)

    # # filename = 'output/metrics.csv'
    # # asvfi_root = '../data/ASVFI'
    # # inroot = '../data/UCF-101_imgs'
    # # outroot = os.path.join(asvfi_root, 'scale', f'{args.scale}')
    # # for category in sorted(os.listdir(inroot)):
        
    # #     category_inroot = os.path.join(inroot, category)
    # #     category_outroot = os.path.join(outroot, category)
    # #     if os.path.exists(category_outroot):
    # #         if len(os.listdir(category_inroot)) == len(os.listdir(category_outroot)):
    # #             continue
    # #         else:   # Resume from the aborted category
    # #             inf_vid(model=model, dataroot=category_inroot, save=category_outroot, scale=scale)
    # #     else:
    # #         inf_vid(model=model, dataroot=category_inroot, save=category_outroot, scale=scale)
    # #     break