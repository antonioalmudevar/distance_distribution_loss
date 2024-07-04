import os

import torch.nn as nn
import torch

import timm
from timm.models.layers import to_2tuple


PRETRAINED_MODELS = {
    'audioset_10_10_0.4593.pth': {'fstride': 10, 'tstride': 10}, 
    'audioset_10_10_0.4495.pth': {'fstride': 10, 'tstride': 10},
}


def get_shape(embedding_dim, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
    test_input = torch.randn(1, 1, input_fdim, input_tdim)
    test_proj = nn.Conv2d(
        1, embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
    test_out = test_proj(test_input)
    f_dim = test_out.shape[2]
    t_dim = test_out.shape[3]
    return f_dim, t_dim


# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# This class is only to load the pretrained models
class ASTModel(nn.Module):
    
    def __init__(
            self, 
            fstride: int=10, 
            tstride: int=10,
            input_fdim: int=128, 
            input_tdim: int=1024,
        ):
        super(ASTModel, self).__init__()

        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
        self.cls_token_num = 2

        self.embed_dim = self.v.pos_embed.shape[2]
        self.f_dim, self.t_dim = get_shape(
            self.embed_dim, fstride, tstride, input_fdim, input_tdim, 16, 16
        )
        num_patches = self.f_dim * self.t_dim

        # Linear Projection
        new_proj = torch.nn.Conv2d(
            1, self.embed_dim, kernel_size=(16, 16), stride=(fstride, tstride)
        )
        self.v.patch_embed.proj = new_proj

        # Positional embedding
        new_pos_embed = nn.Parameter(torch.zeros(
            1, num_patches + 2, self.embed_dim
        ))
        self.v.pos_embed = new_pos_embed


class PretrainedASTModel(nn.Module):

    def __init__(
            self, 
            path: str,
            fstride: int=None, 
            tstride: int=None, 
            input_fdim: int=128, 
            input_tdim: int=1024,
            embedding: str='cls',
        ):
        super(PretrainedASTModel, self).__init__()

        pretrained = os.path.basename(path)
        assert pretrained in PRETRAINED_MODELS.keys(), "Selected model is not available"
        sd = torch.load(path)
        audio_model = ASTModel(
            fstride=PRETRAINED_MODELS[pretrained]['fstride'], 
            tstride=PRETRAINED_MODELS[pretrained]['tstride'],
        )
        audio_model = torch.nn.DataParallel(audio_model)
        audio_model.load_state_dict(sd, strict=False)
        audio_model = audio_model.module

        fstride = PRETRAINED_MODELS[pretrained]['fstride'] if fstride is None else fstride
        tstride = PRETRAINED_MODELS[pretrained]['tstride'] if tstride is None else tstride

        # patch array dimension during finetuning
        original_pos_embed = audio_model.v.pos_embed
        self.embed_dim = original_pos_embed.shape[2]
        f_dim, t_dim = get_shape(
            self.embed_dim, fstride, tstride, input_fdim, input_tdim, 16, 16
        )
        num_patches = f_dim * t_dim

        # patch array dimension during pretraining
        p_f_dim, p_t_dim = audio_model.f_dim, audio_model.t_dim
        p_num_patches = p_f_dim * p_t_dim

        # Class and Distillation Tokens
        self.cls_token = audio_model.v.cls_token
        self.dist_token = audio_model.v.dist_token
        self.cls_token_num = audio_model.cls_token_num

        # Positional embedding
        new_pos_embed = original_pos_embed[:, self.cls_token_num:, :].detach().\
            reshape(1, p_num_patches, self.embed_dim).transpose(1, 2).\
                reshape(1, self.embed_dim, p_f_dim, p_t_dim)
        if t_dim < p_t_dim:
            new_pos_embed = new_pos_embed[
                :, :, :, int(p_t_dim/2)-int(t_dim/2):int(p_t_dim/2)-int(t_dim/2)+t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(
                new_pos_embed, size=(8, t_dim), mode='bilinear')
        if f_dim < p_f_dim:
            new_pos_embed = new_pos_embed[
                :, :, int(p_f_dim/2)-int(f_dim/2):int(p_f_dim/2)-int(f_dim/2)+f_dim, :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(
                new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
        new_pos_embed = new_pos_embed.reshape(
            1, self.embed_dim, num_patches).transpose(1, 2)
        self.pos_embed = nn.Parameter(torch.cat(
            [original_pos_embed[:, :self.cls_token_num, :].detach(), new_pos_embed], dim=1))

        # Patch embedding
        self.patch_embed = audio_model.v.patch_embed
        self.patch_embed.num_patches = num_patches

        # Positional Dropout
        self.pos_drop = audio_model.v.pos_drop

        # Transformer Blocks
        self.blocks = audio_model.v.blocks

        # MLP head
        self.norm = audio_model.v.norm

        self.embedding = embedding

        #self.last_norm = nn.LayerNorm(self.embed_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expect input x = (batch_size, time_frames, frequency_bins)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.embedding == 'avg':
            x = torch.mean(x[:, self.cls_token_num:, :], dim=1)
        elif self.embedding == 'cls':
            x = (x[:, 0] + x[:, 1]) / 2 if self.cls_token_num == 2 else x[:, 0]
        #x = self.last_norm(x)
        return x