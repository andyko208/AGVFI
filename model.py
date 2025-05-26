import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchaudio

import importlib
import config
from resnet_3D import unet_18
from torchaudio.prototype.pipelines import VGGISH
from UNet import UNet_3D_3D

args, _ = config.get_args()


class AudioNet(nn.Module):
    """
    Audio feature extraction network using a pretrained VGGish model.

    The VGGish model is designed to extract features from audio signals.
    This class wraps the VGGish model for use within the ASVFI architecture.
    It expects mono-channel audio input.
    """

    def __init__(self):
        """
        Initializes the AudioNet with a pretrained VGGish model.
        """
        super(AudioNet, self).__init__()
        self.model = VGGISH.get_model()

    def forward(self, x):
        """
        Forward pass of the AudioNet.

        Args:
            x (torch.Tensor): Input audio tensor (raw waveform).
                              Expected shape: (batch_size, num_channels, num_samples)
                              For VGGish, num_channels is typically 1 (mono).

        Returns:
            torch.Tensor: Extracted audio features.
                          Shape: (batch_size, feature_dimension)
                          VGGish typically outputs a 128-dimensional feature vector.
        """
        x = self.model(x)
        return x


class UNet3DEncoder(nn.Module):
    """
    3D UNet encoder based on a modified ResNet-18 architecture.

    This encoder takes a video volume as input and produces a hierarchy of
    feature maps at different spatial resolutions. The architecture is inspired by
    the paper "Raiders of the Lost Art: A 3D UNet for Video Inpainting with
    Audio-Visual Fusion" (http://arxiv.org/pdf/2012.08512).
    """

    def __init__(self):
        """
        Initializes the UNet3DEncoder with a pretrained 3D ResNet-18 backbone.
        Bias is enabled for convolutional layers, and batch normalization is disabled.
        """
        super(UNet3DEncoder, self).__init__()
        unet_3D_module = importlib.import_module("resnet_3D", "model")
        unet_3D_module.useBias = True
        self.encoder = getattr(unet_3D_module, 'unet_18')(
            pretrained=True, bn=False)

    def forward(self, vid):
        """
        Forward pass of the UNet3DEncoder.

        Args:
            vid (torch.Tensor): Input video tensor.
                                Expected shape: (batch_size, num_channels, time_depth, height, width)
                                e.g., (B, 3, T, H, W)

        Returns:
            tuple[torch.Tensor, ...]: A tuple of five feature maps (x_1, x_2, x_3, x_4, x_5)
                                      from different stages of the encoder, representing
                                      multi-scale features.
        """
        x_1, x_2, x_3, x_4, x_5 = self.encoder(vid)
        return x_1, x_2, x_3, x_4, x_5


class UNet3DDecoder(nn.Module):
    """
    3D UNet decoder that reconstructs the video from encoded features.

    This decoder takes feature maps from the UNet3DEncoder and, optionally,
    fused audio-visual features, to generate the output video frames.
    It uses 3D transposed convolutions for upsampling.
    """

    def __init__(self):
        """
        Initializes the UNet3DDecoder.
        The specific UNet architecture (e.g., 'unet_3d_3d') and its parameters
        (number of input/output frames, join type, upsampling mode) are
        configured via command-line arguments.
        """
        super(UNet3DDecoder, self).__init__()
        unet_3D_module = importlib.import_module("UNet", "model")
        self.decoder = getattr(unet_3D_module, 'UNet_3D_3D')(
            args.model.lower(),
            n_inputs=args.nbr_frame,
            n_outputs=args.n_outputs,
            joinType=args.joinType,
            upmode=args.upmode
        )

    def forward(self, layer_index, vid_feat):
        """
        Forward pass for the UNet3DDecoder.

        Args:
            layer_index (int): The index of the decoder layer to use.
            vid_feat (torch.Tensor): Video features input to this decoder layer.

        Returns:
            torch.Tensor: Output feature map from the specified decoder layer.
        """
        return self.decoder(layer_index, vid_feat)


# Channel-Wise Attention Module inspired by https://github.com/FloretCat/CMRAN/blob/master/model/models.py
class New_Audio_Guided_Attention(nn.Module):
    """
    Audio-Guided Attention module with both Channel and Spatial attention.

    This module takes video and audio features as input and computes attention
    maps to selectively focus on relevant parts of the video features, guided
    by the audio information. It's inspired by the Cross-Modal Recurrent
    Attention Network (CMRAN).
    """

    def __init__(self, v_dim):
        """
        Initializes the New_Audio_Guided_Attention module.

        Args:
            v_dim (int): Dimensionality of the input video features (channel size).
        """
        super(New_Audio_Guided_Attention, self).__init__()
        self.hidden_size = v_dim
        self.relu = nn.ReLU()
        # channel attention
        self.affine_audio_1 = nn.Linear(128, self.hidden_size)
        self.affine_video_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.affine_bottleneck = nn.Linear(self.hidden_size, 256)
        self.affine_bottleneck_1 = nn.Linear(784, self.hidden_size)
        self.affine_bottleneck_2 = nn.Linear(self.hidden_size, 1)
        self.affine_v_c_att = nn.Linear(256, self.hidden_size)
        # spatial attention
        self.affine_audio_2 = nn.Linear(128, 256)
        self.affine_video_2 = nn.Linear(self.hidden_size, 256)
        self.affine_v_s_att = nn.Linear(256, 1)

        # video-guided audio attention
        self.affine_video_guided_1 = nn.Linear(self.hidden_size, 64)
        self.affine_video_guided_2 = nn.Linear(64, 128)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, video, audio):
        """
        Forward pass of the New_Audio_Guided_Attention module.

        Args:
            video (torch.Tensor): Video features.
                                  Expected shape: (batch_size, v_dim, time_depth, height, width)
                                  e.g., (B, C, T, H, W)
            audio (torch.Tensor): Audio features.
                                  Expected shape: (batch_size, time_depth, a_dim)
                                  e.g., (B, T, 128)

        Returns:
            torch.Tensor: Spatially and channel-wise attended video features.
                          Shape: (batch_size * time_depth, height * width, v_dim)
                          Effectively, it's a weighted version of the input video features.
        """
        # Reshape video for processing: (B, C, T, H, W) -> (B, T, H, W, C)
        video = video.permute(0, 2, 3, 4, 1)
        batch, t_size, h, w, v_dim = video.shape
        a_dim = audio.size(-1)

        # Flatten audio and video features for batch processing
        audio_feature = audio.view(batch * t_size, a_dim)  # (B*T, a_dim)
        visual_feature = video.reshape(
            batch, t_size, -1, v_dim)  # (B, T, H*W, C)
        raw_visual_feature = visual_feature.reshape(
            batch * t_size, -1, v_dim)  # (B*T, H*W, C)

        # Channel Attention Block
        # Process audio and video features independently
        audio_query_1 = self.relu(self.affine_audio_1(
            audio_feature)).unsqueeze(-2)  # (B*T, 1, C)
        video_query_1 = self.relu(self.affine_video_1(visual_feature)).reshape(
            batch * t_size, h * w, -1)  # (B*T, H*W, C)
        # Element-wise multiplication and further processing
        audio_video_query_raw = (
            audio_query_1 * video_query_1).mean(-2)  # (B*T, C)
        audio_video_query = self.relu(
            self.affine_bottleneck(audio_video_query_raw))  # (B*T, D)
        # Normalization using Sigmoid
        channel_att_maps = self.affine_v_c_att(audio_video_query).sigmoid().reshape(
            batch * t_size, -1, v_dim)  # (B*T, 1, C)
        c_att_visual_feat = (raw_visual_feature * (channel_att_maps + 1)
                             ).reshape(batch * t_size, h * w, -1)  # (B*T, H*W, C)

        # Spatial Attention Block
        # Process channel-attended visual feature and audio feature
        audio_query_2 = self.relu(self.affine_audio_2(
            audio_feature)).unsqueeze(-2)  # (B*T, 1, D)
        c_att_visual_query = self.relu(
            self.affine_video_2(c_att_visual_feat))  # (B*T, H*W, D)
        audio_video_query_2 = c_att_visual_query * \
            audio_query_2  # (B*T, H*W, D)
        # Normalization using Softmax
        spatial_att_maps = self.softmax(
            self.tanh(self.affine_v_s_att(audio_video_query_2)))  # (B*T, H*W, 1)
        c_s_att_visual_feat = c_att_visual_feat * \
            spatial_att_maps  # (B*T, H*W, C)

        return c_s_att_visual_feat


class ASVFI(nn.Module):
    """
    Audio-Spatial Video Frame Inpainting (ASVFI) model.

    This model integrates audio and visual information to perform video inpainting
    or frame interpolation. It consists of:
    1. An audio encoder (AudioNet) to extract features from audio.
    2. A 3D UNet video encoder (UNet3DEncoder) to extract multi-scale video features.
    3. A series of audio-guided attention fusion layers (New_Audio_Guided_Attention)
       to combine audio and video features at different scales.
    4. A 3D UNet video decoder (UNet3DDecoder) to reconstruct the video output.
    """

    def __init__(self):
        """
        Initializes the ASVFI model, including its submodules (encoders, decoder, fusion layers).
        The number of fusion layers and their dimensions are predefined.
        """
        super(ASVFI, self).__init__()

        self.audio_encoder = AudioNet()
        self.video_encoder = UNet3DEncoder()
        self.video_decoder = UNet3DDecoder()
        nf = [512, 512, 256, 128, 128, 64]
        self.fusion_layers = [
            New_Audio_Guided_Attention(n).cuda() for n in nf
        ]
        # self.linear_layers = [nn.Linear(nf[i], nf[i+1]).cuda() for i in range(len(nf)-1)] + [nn.Linear(64, 64).cuda()] # Keep for now, might be part of a future implementation

    def fuse(self, layer_index, vid_feat, aud_feat):
        """
        Fuses video features with audio features using the attention mechanism.

        Args:
            layer_index (int): The index of the fusion layer to use.
            vid_feat (torch.Tensor): Video features from a specific encoder/decoder stage.
                                     Expected shape: (batch_size, v_dim, time_depth, height, width)
            aud_feat (torch.Tensor): Audio features.
                                     Expected shape: (batch_size, time_depth, a_dim)

        Returns:
            torch.Tensor: Enhanced video features after audio-visual fusion.
                          Shape: (batch_size, v_dim, time_depth, height, width)
        """
        # Example shape: video_feats[-i]: torch.Size([1, 512, 17, 28, 28])
        batch, v_dim, t_size, h, w = vid_feat.shape
        # Attention module expects (B*T, H*W, C_vid) and (B*T, C_aud)
        # Current vid_feat is (B, C_vid, T, H, W), aud_feat is (B, T, C_aud)
        enhanced_vid_feat = self.fusion_layers[layer_index](
            vid_feat, aud_feat)  # Output: (B*T, H*W, v_dim)

        # Reshape back to (B, C_vid, T, H, W) for the decoder
        enhanced_vid_feat = enhanced_vid_feat.reshape(
            batch, t_size, h, w, v_dim)
        enhanced_vid_feat = enhanced_vid_feat.permute(
            0, 4, 1, 2, 3)  # (B, v_dim, T, H, W)

        return enhanced_vid_feat

    def decode(self, layer_index, prev_vid_feat, enhanced_vid_feat):
        """
        Decodes the enhanced video features using the corresponding decoder layer.

        Args:
            layer_index (int): The index of the decoder layer to use.
            prev_vid_feat (torch.Tensor or None): Video features from the corresponding
                                                  encoder layer (skip connection).
                                                  If None, no skip connection is used.
            enhanced_vid_feat (torch.Tensor): Enhanced video features from the fusion layer.

        Returns:
            torch.Tensor: Output feature map from the decoder layer.
        """
        decoder_out = self.video_decoder(
            layer_index, enhanced_vid_feat)  # (B, vid_i-1_dim, T, h, w)
        if prev_vid_feat is not None:
            # Concatenate with features from the corresponding encoder layer (skip connection)
            # (B, vid_i_dim, T, h, w)
            cat_vid_feats = torch.cat((prev_vid_feat, decoder_out), dim=1)
            return cat_vid_feats
        return decoder_out

    def forward(self, vid, mfcc):
        """
        Forward pass of the ASVFI model.

        Args:
            vid (torch.Tensor): Input video tensor.
                                Expected shape: (batch_size, 3, time_depth, height, width)
                                e.g., (B, 3, T, H, W)
            mfcc (torch.Tensor): Input MFCC (audio) tensor.
                                 Expected shape: (batch_size, time_depth, 1, num_frames_mfcc, num_mels_mfcc)
                                 e.g., (B, T, 1, 96, 64)

        Returns:
            list[torch.Tensor]: A list of output video frames.
                                Each tensor in the list has a shape of
                                (batch_size, 3, height, width).
                                Currently, it's designed to output a single interpolated/inpainted frame,
                                but the list structure allows for future extension to multiple frames.
        """
        # Batch mean normalization for input video
        # (Implementation detail from CAIN: https://github.com/myungsub/CAIN)
        mean_ = vid.mean(2, keepdim=True).mean(
            3, keepdim=True).mean(4, keepdim=True)
        vid = vid - mean_

        # Audio Encoding
        # Reshape MFCCs and pass through audio encoder
        batch, t_size, _, n_frames, n_mels = mfcc.shape
        # VGGish expects (batch_size, num_channels, num_frames, num_bands)
        # Here, num_channels = 1, num_frames = n_frames (e.g., 96), num_bands = n_mels (e.g., 64)
        reshaped_audio = mfcc.view(batch * t_size, 1, n_frames, n_mels)
        aud_feat = self.audio_encoder(reshaped_audio).view(
            batch, t_size, -1)  # (B, T, audio_feature_dim)

        # Video Encoding
        # video_feats is a tuple of 5 feature maps from the UNet3DEncoder
        video_feats = self.video_encoder(vid)
        # Example shapes of video_feats elements:
        # video_feats[0]: torch.Size([B, 64, T, H, W])
        # video_feats[1]: torch.Size([B, 64, T, H/2, W/2]) (example, actual depends on resnet_3D)
        # ...
        # video_feats[4]: torch.Size([B, 512, T, H/16, W/16]) (bottleneck features)

        # Decoder Path with Fusion
        # The process involves iteratively fusing video features with audio features
        # and then decoding them, incorporating skip connections from the encoder.
        # for vid in video_feats:
        #     torch.Size([1, 64, 17, 112, 112])
        #     torch.Size([1, 64, 17, 112, 112])
        #     torch.Size([1, 128, 17, 56, 56])
        #     torch.Size([1, 256, 17, 28, 28])
        #     torch.Size([1, 512, 17, 28, 28])

        """
        Repeat this step in the subsequent layer
            Just keep the original FlaVR code but keep conv2d
        """
        vid_feat = video_feats[-1]  # Output of the last encoder layer
        for layer_idx in range(4):  # Corresponds to decoder layers 0, 1, 2, 3
            enhanced_vid_feat = self.fuse(layer_idx, vid_feat, aud_feat)
            # Skip connection from corresponding encoder layer
            prev_vid_feat = video_feats[-2 - layer_idx]
            vid_feat = self.decode(layer_idx, prev_vid_feat, enhanced_vid_feat)

        # Decoder layer 4
        enhanced_vid_feat = self.fuse(4, vid_feat, aud_feat)
        # No skip connection from encoder for this layer in the original logic
        vid_feat = self.decode(4, None, enhanced_vid_feat)

        # Final decoder layer (layer 5)
        enhanced_vid_feat = self.fuse(5, vid_feat, aud_feat)
        # No skip connection. Output has dim=1 -> 3 * n_outputs
        out = self.decode(5, None, enhanced_vid_feat)

        mean_ = mean_.squeeze(2)  # Restore original mean
        out = [o + mean_ for o in out]
        # Returns a list of output images (currently a single image)
        return out
