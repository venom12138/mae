
import torch
from timm.models.vision_transformer import PatchEmbed, Block
import torch.nn as nn

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

noise = torch.rand(10)  # noise in [0, 1]
    
# sort noise for each sample
ids_shuffle = torch.argsort(noise)  # ascend: small is keep, large is remove
ids_restore = torch.argsort(ids_shuffle)
print(f'ids_shuffle:{ids_shuffle}')
print(f'ids_restore:{ids_restore}')