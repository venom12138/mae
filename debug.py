
import torch
from timm.models.vision_transformer import PatchEmbed, Block
import torch.nn as nn
from networks import models_bspmae
from networks import models_vit
from copy import deepcopy
# def random_masking(x, mask_ratio):
#     """
#     Perform per-sample random masking by per-sample shuffling.
#     Per-sample shuffling is done by argsort random noise.
#     x: [N, L, D], sequence
#     """
#     N, L, D = x.shape  # batch, length, dim
#     len_keep = int(L * (1 - mask_ratio))
    
#     noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
#     # sort noise for each sample
#     ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
#     print(f'idxshuffle:{ids_shuffle}')
#     ids_restore = torch.argsort(ids_shuffle, dim=1)
#     print(f'idx_restore:{ids_restore}')
#     # keep the first subset
#     ids_keep = ids_shuffle[:, :len_keep]
#     print(f'idx_keep:{ids_keep.unsqueeze(-1).repeat(1, 1, D).size()}')
#     x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
#     print(f'xmask:{x_masked.size()}')
#     # generate the binary mask: 0 is keep, 1 is remove
#     mask = torch.ones([N, L], device=x.device)
#     mask[:, :len_keep] = 0
#     # unshuffle to get the binary mask
#     mask = torch.gather(mask, dim=1, index=ids_restore)

#     return x_masked, mask, ids_restore

# x = torch.rand(1,3,224,224)
# patch_embed = PatchEmbed(224, 16, 3, 192)
# x = patch_embed(x)

# pos_embed = nn.Parameter(torch.zeros(1, patch_embed.num_patches + 1, 192), requires_grad=False)  # fixed sin-cos embedding
# x = x + pos_embed[:, 1:, :]

# x, mask, ids_restore = random_masking(x, 0.75)
# print(f'mask shape:{mask.size()}')
# print(f'ids_restore shape:{ids_restore.size()}')

# noise = torch.rand(10)  # noise in [0, 1]
# x = torch.range(10)

# # sort noise for each sample
# ids_shuffle = torch.argsort(noise)  # ascend: small is keep, large is remove
# ids_restore = torch.argsort(ids_shuffle)
# ids_keep = ids_shuffle[:5] # mask掉5个
# print(f'ids_shuffle:{ids_shuffle}')
# print(f'ids_restore:{ids_restore}')

# x_mask = x_mask[ids_keep]
# print(f'x_mask:{x_mask}')

# mask_token = torch.zeros(5)
# x_ = torch.cat((x_mask, mask_token))
# x_ = x_[ids_restore]
# print(f'restore_x:{x_}')

# x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))


# model = models_bspmae.__dict__['mae_deit_tiny_patch4_dec512d'](norm_pix_loss=True)
# resume = '/home/jwyu/venom/bootstrap-mae/ckpt.pth'
# checkpoint = torch.load(resume, map_location='cpu')
# encoder_ckpt = {}
# decoder_ckpt = {}
# for k,v in checkpoint['model'].items():
#     if 'encoder' in k:
#         if 'head' in k:
#             print(k)
#         if k.startswith('encoder.'):
#             newkey = k[len('encoder.') :]
#             encoder_ckpt.update({newkey:v})
#     if 'decoder' in k:
#         if k.startswith('decoder.'):
#             newkey = k[len('decoder.') :]
#             decoder_ckpt.update({newkey:v})

# sd_before = deepcopy(model.state_dict())
# model.encoder.load_state_dict(encoder_ckpt)
# model.decoder.load_state_dict(decoder_ckpt)
# # model.proxy_encoder.load_state_dict(encoder_ckpt)

# sd_after = model.state_dict()
# diff_keys = [k for k in sd_before if not torch.equal(sd_before[k], sd_after[k])]

# # print(set(diff_keys)^set(sd_before.keys()))
# # assert set(diff_keys) == set(sd_before.keys())

# print(encoder_ckpt.keys())
# print(decoder_ckpt.keys())
# print(len(encoder_ckpt.keys()))
# print(len(decoder_ckpt.keys()))

# model = models_vit.__dict__['deit_tiny_patch4_32'](
#         num_classes=10,
#         global_pool=False,
#     )

# resume = '/home/jwyu/venom/bootstrap-mae/ckpt.pth'
# checkpoint = torch.load(resume, map_location='cpu')
# encoder_ckpt = {}
# decoder_ckpt = {}
# for k,v in checkpoint['model'].items():
#     if 'encoder' in k:
#         if k.startswith('encoder.'):
#             newkey = k[len('encoder.') :]
#             encoder_ckpt.update({newkey:v})

# sd_before = deepcopy(model.state_dict())

# for k in ['head.weight', 'head.bias']:
#     if k in encoder_ckpt and encoder_ckpt[k].shape != sd_before[k].shape:
#         print(f"Removing key {k} from pretrained checkpoint")
#         del encoder_ckpt[k]

# model.load_state_dict(encoder_ckpt,strict=False)
# sd_after = model.state_dict()
# diff_keys = [k for k in sd_before if not torch.equal(sd_before[k], sd_after[k])]
# print(set(diff_keys)^set(sd_before.keys()))
import timm.optim.optim_factory as optim_factory

model = models_bspmae.__dict__['mae_deit_tiny_patch4_dec512d'](norm_pix_loss=True, bsp=True)
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(name)
def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            # print(f'frozen weight name:{name}')
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
