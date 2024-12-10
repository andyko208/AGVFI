import os
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F

from torchaudio import transforms, load
from torchaudio.prototype.pipelines import VGGISH
from torchvision import transforms as img_transforms
from torch.utils.data import Dataset, DataLoader

SR = 44100

class UCF(Dataset):
    def __init__(self, split=None):
        super().__init__()
        self.csv = pd.read_csv('../data/UCF-101_split.csv')
        if split != None:
            self.csv = self.csv[self.csv['split']==split]

        if split=='train':
            self.img_transforms = img_transforms.Compose([
                img_transforms.CenterCrop((224,224)),
                img_transforms.RandomHorizontalFlip(0.5),
                img_transforms.RandomVerticalFlip(0.5),
                img_transforms.ToTensor(),
            ])
        else:
            self.img_transforms = img_transforms.Compose([
                img_transforms.CenterCrop((224,224)),
                img_transforms.ToTensor(),
            ])
        self.wav_extractor = VGGISH.get_input_processor()

        # self.wav_transform = transforms.MFCC(
        #     sample_rate=SR,
        #     n_mfcc=40,
        #     melkwargs={
        #         "n_fft": 2048,
        #         "hop_length": 512,
        #         "n_mels": 128,
        #     },
        #     # melkwargs={
        #     #     "n_fft": 2048,
        #     #     "hop_length": 256,
        #     #     "n_mels": 256,
        #     #     "win_length": 2048,
        #     # },
        #     log_mels=True,

        # )
        # self.normalize = img_transforms.Compose([
        #     img_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

    def format_wav(self, audio, sr):

        """
        1. Pad/Trim the waveform
        2. Extract features via VGGish Extractor
            Audio length distribution
                mean: 6.8, med: 5.9, std: 3.7
        """
        
        if sr != SR:        # Resample the sampling rate to SR=44100
            audio = transforms.Resample(sr, SR)

        target_sample = 6 * SR
        n_samples = audio.shape[-1]

        if n_samples < target_sample:   # Pad if shorter
            padding = target_sample - n_samples
            audio = F.pad(audio, (0, padding))
        else:                           # Trim if longer
            audio = audio[:, :target_sample]

        mfcc = self.wav_extractor(audio[0])
        return mfcc

    def format_vid(self, imgs):
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
        
    
    def __len__(self):
        
        return len(self.csv)
    
    def __getitem__(self, index):
        
        r = self.csv.iloc[index]
        img_paths = [r['I0'], r['I1'], r['GT']]
        audio, sr =  load(r['A'])

        imgs = [self.img_transforms(Image.open(img)) for img in img_paths]  
        vid = self.format_vid(imgs[:-1])    # (T, C, H, W)
        mfcc = self.format_wav(audio, sr)   # (T, 1, n_example, 64)
        gt = imgs[-1]

        return vid, mfcc, gt
    
def get_loader(split, batch_size, shuffle, num_workers):
    return DataLoader(UCF(split=split), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
