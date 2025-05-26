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
            self.csv = self.csv[self.csv['split'] == split]

        if split == 'train':
            self.img_transforms = img_transforms.Compose([
                img_transforms.CenterCrop((224, 224)),
                img_transforms.RandomHorizontalFlip(0.5),
                img_transforms.RandomVerticalFlip(0.5),
                img_transforms.ToTensor(),
            ])
        else:
            self.img_transforms = img_transforms.Compose([
                img_transforms.CenterCrop((224, 224)),
                img_transforms.ToTensor(),
            ])
        self.wav_extractor = VGGISH.get_input_processor()

    def format_wav(self, audio, sr):
        """
        Formats the audio waveform by resampling, padding/trimming, and extracting features using VGGish.

        The audio is processed to have a consistent sampling rate (SR) and a fixed duration (6 seconds).
        VGGish features are then extracted from the processed waveform.

        Args:
            audio (torch.Tensor): The input audio waveform.
            sr (int): The sampling rate of the input audio.

        Returns:
            torch.Tensor: Extracted audio features (MFCC-like) from VGGish.
                          Shape: (num_frames, feature_dim) e.g., (T, 1, num_vggish_frames, 64)
        """
        if sr != SR:  # Resample if the sampling rate is not the target SR
            resample_transform = transforms.Resample(orig_freq=sr, new_freq=SR)
            audio = resample_transform(audio)

        target_sample = 6 * SR  # Target duration of 6 seconds
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
        Formats a list of image tensors into a video volume by temporal interpolation.

        The input images (typically start and end frames) are interpolated to a fixed
        temporal depth (T=17 frames) to match the audio feature dimensions.

        Args:
            imgs (list[torch.Tensor]): A list of image tensors (C, H, W).

        Returns:
            torch.Tensor: An interpolated video tensor.
                          Shape: (T, C, H, W), where T=17.
        """
        T = 17  # Target number of frames (temporal depth)
        # Stack images along a new dimension (dim=1 for time) before interpolation
        # Input to interpolate should be (N, C, T_in, H, W) or (C, T_in, H, W)
        # Here, we stack to (C, num_input_images, H, W) then permute to (num_input_images, C, H, W)
        # then unsqueeze for batch dim for interpolate: (1, num_input_images, C, H, W)
        # No, F.interpolate expects (N, C, D_in, H_in, W_in) for 3D mode.
        # So, stack to (C,H,W) -> list of (C,H,W) -> stack to (num_imgs, C, H, W)
        # then permute to (C, num_imgs, H, W) and unsqueeze for batch.

        # Correct approach: stack images, then permute to (C, num_input_frames, H, W)
        # then add batch dim for interpolation.
        # However, the original code stacks to (num_input_frames, C, H, W) then permutes to (C, num_input_frames, H, W)
        # Let's keep the original logic:
        frames = torch.stack(imgs, dim=0)  # Shape: (num_input_images, C, H, W)
        # The interpolate function for 'trilinear' expects input as (N, C, D, H, W).
        # We want to interpolate along the D dimension (temporal).
        # So, current 'frames' (num_imgs, C, H, W) needs to be (C, num_imgs, H, W) then (1, C, num_imgs, H, W)
        frames_permuted = frames.permute(
            1, 0, 2, 3)  # (C, num_input_images, H, W)

        interpolated_frames = F.interpolate(
            # Add batch dimension: (1, C, num_input_images, H, W)
            frames_permuted.unsqueeze(0),
            size=(T, 224, 224),             # Target size (D_out, H_out, W_out)
            mode='trilinear',               # Trilinear interpolation
            align_corners=True              # Preserve corner values
        ).squeeze(0)                        # Remove batch dimension: (C, T, H, W)

        # Permute to (T, C, H, W) to match expected video format elsewhere
        return interpolated_frames.permute(1, 0, 2, 3)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.csv)

    def __getitem__(self, index):
        """
        Retrieves a single sample (video, audio features, ground truth image) from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - vid (torch.Tensor): The processed video tensor (T, C, H, W).
                - mfcc (torch.Tensor): The extracted audio features.
                - gt (torch.Tensor): The ground truth image tensor (C, H, W).
        """
        r = self.csv.iloc[index]
        # Expecting I0 (start frame), I1 (end frame), GT (middle frame)
        img_paths = [r['I0'], r['I1'], r['GT']]
        audio, sr = load(r['A'])  # Load audio waveform

        # Apply image transforms to each image
        imgs = [self.img_transforms(Image.open(img_path))
                for img_path in img_paths]

        # Format video using start and end frames (imgs[0] and imgs[1])
        vid = self.format_vid(imgs[:-1])    # Output shape: (T, C, H, W)

        # Format audio waveform to get MFCC-like features
        # Output shape from VGGish processor
        mfcc = self.format_wav(audio, sr)

        # Ground truth is the last image in the list (middle frame)
        gt = imgs[-1]                       # Shape: (C, H, W)

        return vid, mfcc, gt


def get_loader(split, batch_size, shuffle, num_workers):
    """
    Creates a DataLoader for the UCF dataset.

    Args:
        split (str): Dataset split, e.g., 'train', 'test', or None for all data.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.
        num_workers (int): How many subprocesses to use for data loading.

    Returns:
        torch.utils.data.DataLoader: DataLoader instance for the specified dataset split.
    """
    dataset = UCF(split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
