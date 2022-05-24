# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import sys
sys.path.append('..')
import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

class MAE_Encoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # 通过卷积使得图片变为14*14的featuremap，一共有embed_dim个channel，展开后变为196*embed_dim，对196个patch进行mask操作即可
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.cls_token, std=.02)
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_keep):
        # embed patches
        x = self.patch_embed(x)

        # print(f'embed_size:{x.size()}')
        # add pos embed w/o cls token
        # 第一个留给cls_token
        x = x + self.pos_embed[:, 1:, :]

        # print(f'after pos_embed:{x.size()}')
        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)
        N, L, D = x.shape
        
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # print(f'after masking:{x.size()}')
        # append cls token
        # cls_token 加上pos_embedding
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # expand 成[B,1,embeding_size]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # print(f'after cls_tokens:{x.size()}')
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # print(f'after encoder:{x.size()}')
        return x

class MAE_Decoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, bsp=False):
        super().__init__()
        self.num_patches = PatchEmbed(img_size, patch_size, in_chans, embed_dim).num_patches
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        if bsp:
            self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True) # decoder to patch
        else:
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore):
        # embed tokens
        # print(f'decoder input size:{x.size()}')
        x = self.decoder_embed(x)
        
        # append mask tokens to sequence
        # [B, 14*14 + 1 - , 1]
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, bsp=False):
        super().__init__()
        self.encoder = MAE_Encoder(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                                    mlp_ratio, norm_layer)
        
        self.decoder = MAE_Decoder(img_size, patch_size, in_chans, embed_dim, decoder_embed_dim, decoder_depth, decoder_num_heads,
                                    mlp_ratio, norm_layer, bsp=bsp)
        # bootstrap
        if bsp:
            self.proxy_encoder = MAE_Encoder(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                                    mlp_ratio, norm_layer)
            self.proxy_encoder.requires_grad=False
        self.bsp = bsp
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.norm_pix_loss = norm_pix_loss

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        # print(f'patch_size={p}')
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        
        # shuffle后被保留下来的patch的值，形状是[B, L*(1-mask_ratio), D]
        # x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        # 形状是[B,14*14], 0表示原图中保留下来的patch, 1表示原图中被mask的patch,也就是记录被保留下来的patch都在原图中什么位置
        # 若想要知道x_mask中的每一个元素对应原图中的什么位置，只需要类似于torch.gather[x_mask, dim=1, index=ids_restore]即可将图片恢复
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, len_keep, ids_shuffle, ids_restore

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # print(f'imgs:{imgs.size()}')
        if not self.bsp:
            target = self.patchify(imgs)
        else:
            target = imgs
        # print(f'target:{target.size()}')
        # print(f'pred:{pred.size()}')
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # print(f'loss:{loss.size()}')
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        random_imgs = torch.rand(imgs.shape[0], (self.img_size//self.patch_size)**2, self.embed_dim, device=imgs.device)
        # print(f'random_imgs:{random_imgs.size()}')
        # ids_restore[ids_shuffle] = [0,1,2,3,...] 也就是恢复顺序
        
        ids_keep, mask, len_keep, ids_shuffle, ids_restore = self.random_masking(random_imgs, mask_ratio)
        # print(f'mask:{mask}')
        if self.bsp:
            ids_keep_for_proxy = ids_shuffle[:, len_keep:]
            proxy_latent = self.proxy_encoder(imgs, ids_keep_for_proxy)
            
            proxy_mask_tokens = torch.zeros(1, 1, self.embed_dim, device=imgs.device).repeat(proxy_latent.shape[0], ids_restore.shape[1] + 1 - proxy_latent.shape[1], 1)
            proxy_output = torch.cat([proxy_mask_tokens, proxy_latent[:, 1:, :]], dim=1)  # no cls token
            proxy_output = torch.gather(proxy_output, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, proxy_latent.shape[2]))  # unshuffle
            # print(f'proxy_out:{proxy_output[:,:,0]}')
        latent = self.encoder(imgs, ids_keep)
        
        pred = self.decoder(latent, ids_restore)  # [N, L, p*p*3]
        
        if self.bsp:
            loss = self.forward_loss(proxy_output, pred, mask)
        else:
            loss = self.forward_loss(imgs, pred, mask)
        
        return loss, pred, mask

# decoder_num_heads的decoder_depth numheads和embeddim保持不变，
def mae_deit_tiny_patch4_dec512d(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=32, patch_size=4, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

if __name__=='__main__':
    model = mae_deit_tiny_patch4_dec512d(bsp=False)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'n_param:{n_parameters}')
    x = torch.rand(2,3,32,32)
    model(x)