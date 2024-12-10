import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchaudio

import importlib
import config
from resnet_3D import unet_18
# from resnet_2D import resnet18
from torchaudio.prototype.pipelines import VGGISH
from UNet import UNet_3D_3D

args, _ = config.get_args()


class AudioNet(nn.Module):
    """
    Pretrained VGGIsh to extract raw waveform
        - Takes mono-channel input from dual channel for extracting MFCC
    """
    def __init__(self):
        super(AudioNet, self).__init__()
        # self.model = models.resnet18(weights='DEFAULT')
        self.model = VGGISH.get_model()

    def forward(self, x):
        x = self.model(x)
        return x
    

class UNet3DEncoder(nn.Module):
    """
    Model architecture from http://arxiv.org/pdf/2012.08512, modified r3d_18
    """
    def __init__(self):
        super(UNet3DEncoder, self).__init__()
        unet_3D = importlib.import_module("resnet_3D" , "model")
        unet_3D.useBias = True
        self.encoder = getattr(unet_3D, 'unet_18')(pretrained=True, bn=False)

    def forward(self, vid):
        x_1 , x_2 , x_3 , x_4, x_5 = self.encoder(vid)
        return x_1 , x_2 , x_3 , x_4, x_5

class UNet3DDecoder(nn.Module):
    def __init__(self):
        super(UNet3DDecoder, self).__init__()
        unet_3D = importlib.import_module("UNet" , "model")
        self.decoder = getattr(unet_3D, 'UNet_3D_3D')(args.model.lower() , n_inputs=args.nbr_frame, n_outputs=args.n_outputs, joinType=args.joinType, upmode=args.upmode)

    def forward(self, lth, vid_feat):
        """
        Define forward function
        """
        return self.decoder(lth, vid_feat)

# Channel-Wise Attention Module from https://github.com/FloretCat/CMRAN/blob/master/model/models.py
class New_Audio_Guided_Attention(nn.Module):
    def __init__(self, v_dim):
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
        '''
        :param visual_feature: [batch, 10, 7, 7, 512]
        :param audio_feature:  [batch, 10, 128]
        :return: [batch, 10, 512]

        ours:
        :param visual_feature: [batch, 512, 17, 28, 28]
        :param audio_feature:  [batch, 17, 128]
        :return: [batch, 10, 512]
        '''
        # audio = audio.transpose(1, 0)
        video = video.permute(0, 2, 3, 4, 1)    # (B, T, H, W, C)
        batch, t_size, h, w, v_dim = video.shape 
        a_dim = audio.size(-1)  
        audio_feature = audio.view(batch*t_size, a_dim)             # (B*T, 128)
        visual_feature = video.reshape(batch, t_size, -1, v_dim)    # (B, T, H*W, C)
        raw_visual_feature = visual_feature.reshape(batch*t_size, -1, v_dim)    # # (B*T, H*W, C)

        # ============================== Channel Attention ====================================
        """
        audio and video features are independently processed by a fully-connected layer with ReLu as an activation function
        """
        audio_query_1 = self.relu(self.affine_audio_1(audio_feature)).unsqueeze(-2)                     # (B*T, 1, C)
        video_query_1 = self.relu(self.affine_video_1(visual_feature)).reshape(batch*t_size, h*w, -1)   # (B*T, H*W, C)
        """
        We perform element-wise multiplication on the processed audio and video features
        further process them through two fully-connected layers with ReLu as an activation function 
        """
        audio_video_query_raw = (audio_query_1 * video_query_1).mean(-2)                # (B*T, C)
        audio_video_query = self.relu(self.affine_bottleneck(audio_video_query_raw))    # (B*T, D)
        # audio_video_query_raw = (audio_query_1 * video_query_1).permute(0, 2, 1)           # (B*T, C, H*W)
        # audio_video_query = self.affine_bottleneck_2(self.relu(self.affine_bottleneck_1(audio_video_query_raw))).permute(0, 2, 1)   # (B*T, 1, C)
        """
        Finally, The Sigmoid activation function is used to normalize.
        """
        channel_att_maps = self.affine_v_c_att(audio_video_query).sigmoid().reshape(batch*t_size, -1, v_dim)   # (B*T, 1, C)
        c_att_visual_feat = (raw_visual_feature * (channel_att_maps + 1)).reshape(batch*t_size, h*w, -1)       # (B*T, H*W, C)

        # ============================== Spatial Attention =====================================
        # channel attended visual feature: [batch * 10, 49, v_dim]
        # c_att_visual_feat = c_att_visual_feat.reshape(batch*t_size, -1, v_dim)          # (B*T, H*W, 512)
        audio_query_2 = self.relu(self.affine_audio_2(audio_feature)).unsqueeze(-2)     # (B*T, 1, D)
        c_att_visual_query = self.relu(self.affine_video_2(c_att_visual_feat))          # (B*T, H*W, D)
        audio_video_query_2 = c_att_visual_query * audio_query_2                        # (B*T, H*W, D)
        # spatial_att_maps = self.softmax(self.tanh(self.affine_v_s_att(audio_video_query_2)).transpose(2, 1))    # (B*T, 1, H*W)
        # c_s_att_visual_feat = torch.bmm(spatial_att_maps, c_att_visual_feat).squeeze().reshape(batch, t_size, v_dim)    # (B, T, 512)
        spatial_att_maps = self.softmax(self.tanh(self.affine_v_s_att(audio_video_query_2)))    # (B*T, H*W, 1)
        # c_s_att_visual_feat = torch.bmm(spatial_att_maps, c_att_visual_feat)    # (B, T, 512)
        c_s_att_visual_feat = c_att_visual_feat * spatial_att_maps                # (B, T, 512)
        # print(f'c_s_att_visual_feat: {c_s_att_visual_feat.shape}')

        return c_s_att_visual_feat
    

class ASVFI(nn.Module):
    def __init__(self):
        super(ASVFI, self).__init__()
        
        self.audio_encoder = AudioNet()
        self.video_encoder = UNet3DEncoder()
        self.video_decoder = UNet3DDecoder()
        nf = [512 , 512 , 256, 128 , 128 , 64]
        self.fusion_layers = [
            New_Audio_Guided_Attention(n).cuda() for n in nf
        ]
        # self.linear_layers = [nn.Linear(nf[i], nf[i+1]).cuda() for i in range(len(nf)-1)] + [nn.Linear(64, 64).cuda()]

    def fuse(self, lth, vid_feat, aud_feat):
        """
        Fuse vid_feat w/ aud_feat with 6-lth(1~5) layer, becomes input to the lth layer of decoder
        """
        # video_feats[-i]: torch.Size([1, 512, 17, 28, 28])
        batch, v_dim, t_size, h, w = vid_feat.shape
        enhanced_vid_feat = self.fusion_layers[lth](vid_feat, aud_feat)    # (B*T, H*W, v_dim)
        enhanced_vid_feat = enhanced_vid_feat.reshape(batch, t_size, h, w, v_dim)
        enhanced_vid_feat = enhanced_vid_feat.permute(0, -1, 1, 2, 3)
        
        return enhanced_vid_feat
    
    def decode(self, lth, prev_vid_feat, enhanced_vid_feat):
        """
        Decode the fused output with the lth layer of video_decoder
        Concat prev_vid_feat w/ decoder_out on v_dim
        """
        decoder_out = self.video_decoder(lth, enhanced_vid_feat)                   # (B, vid_i-1_dim, T, h, w)
        # batch, v_dim, t_size, h, w = decoder_out.shape
        # cat_vid_feats = torch.cat((prev_vid_feat, v_t_lth), dim=1)              # (B, vid_i_dim, T, h, w)
        if prev_vid_feat != None:
            cat_vid_feats = torch.cat((prev_vid_feat, decoder_out), dim=1)            # (B, vid_i_dim, T, h, w)
            # print(f'v_{5-lth+1}: {cat_vid_feats.shape}')
            return cat_vid_feats
        # cat_vid_feats = cat_vid_feats.permute(0, 2, 3, 4, 1)
        # v_t = self.linear_layers[lth](cat_vid_feats)
        # v_t = v_t.permute(0, -1, 1, 2, 3)
        # v_t = torch.cat((prev_vid_feat, decoder_out), dim=1)            # (B, vid_i_dim, T, h, w)
        # print(f'v_{6-lth-1}: {v_t.shape}')
        # return v_t
        
        return decoder_out


    def forward(self, vid, mfcc):
        
        """
        vid:    (B, 3, T, H, W)
        mfcc:   (B, T, 1, 96, 64)
        gt:     (B, 3, H, W)
        """
        
        ## Batch mean normalization works slightly better than global mean normalization, thanks to https://github.com/myungsub/CAIN
        mean_ = vid.mean(2, keepdim=True).mean(3, keepdim=True).mean(4,keepdim=True)
        vid -= mean_

        """
        We feed fused v5 into the first decoder layer, and through 3D transpose convolution 
        and Feature Gating, we get 3D feature maps.
        """
        batch, t_size, _, n_frames, n_mels = mfcc.shape
        reshaped_audio = mfcc.view(batch*t_size, 1, n_frames, n_mels)
        aud_feat = self.audio_encoder(reshaped_audio).view(batch, t_size, -1)
        
        video_feats = self.video_encoder(vid)
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
        vid_feat = video_feats[-1]  # First input to fusion network is the output of the last encoded vid feature
        for lth in range(4):
            # print(f'************************{1-4}th layer************************')
            enhanced_vid_feat = self.fuse(lth, vid_feat, aud_feat)
            # print(f'enhanced_vid_feat: {enhanced_vid_feat.shape}')
            prev_vid_feat = video_feats[-2-lth]
            # print(f'prev_vid_feat: video_feats[{6-lth-2}]: {prev_vid_feat.shape}')
            vid_feat = self.decode(lth, prev_vid_feat, enhanced_vid_feat)
            # print(f'vid_feat: {vid_feat.shape}')
            # break
        # print(f'************************{5}th layer************************')
        enhanced_vid_feat = self.fuse(4, vid_feat, aud_feat)
        # print(f'enhanced_vid_feat: {enhanced_vid_feat.shape}')
        vid_feat = self.decode(4, None, enhanced_vid_feat)
        # print(f'vid_feat: {vid_feat.shape}')
        # print(f'************************Last layer************************')
        enhanced_vid_feat = self.fuse(5, vid_feat, aud_feat)
        # print(f'enhanced_vid_feat: {enhanced_vid_feat.shape}')
        out = self.decode(5, None, enhanced_vid_feat)  # dim=1 -> 3 * n_outputs
        mean_ = mean_.squeeze(2)
        out = [o+mean_ for o in out]
        # print(len(out), out[0], out[0].shape)
        # print(f'n_imgs: {len(I_T)}')
        return out      # Single output image for now, later train w/ more gt imgs

# model = ASVFI()
# print(model)