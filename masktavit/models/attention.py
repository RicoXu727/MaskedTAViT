'''
Code referenced from https://github.com/wilson1yan/VideoGPT
'''


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .utils import shift_dim, view_range, tensor_slice

import itertools

class AttentionStack(nn.Module):
    def __init__(
        self, shape, embd_dim, n_head, n_layer, dropout, dist_mask,
        attn_type, attn_dropout, frame_cond_shape,
    ):
        super().__init__()
        self.shape = shape
        self.embd_dim = embd_dim
        self.use_frame_cond = frame_cond_shape is not None

        self.right_shift = RightShift(embd_dim)
        self.pos_embd = AddBroadcastPosEmbed(
            shape=shape, embd_dim=embd_dim
        )

        self.attn_nets = nn.ModuleList(
            [
                AttentionBlock(
                    shape=shape,
                    embd_dim=embd_dim,
                    n_head=n_head,
                    n_layer=n_layer,
                    dropout=dropout,
                    dist_mask = dist_mask,
                    attn_type=attn_type,
                    attn_dropout=attn_dropout,
                    frame_cond_shape=frame_cond_shape
                )
                for i in range(n_layer)
            ]
        )

    def forward(self, x, frame_cond, decode_step, decode_idx):
        """
        Args
        ------
            x: (b, d1, d2, ..., dn, embd_dim)
            frame_cond: conditioning tensor

            (below is used only when sampling for fast decoding)
            decode: the enumerated rasterscan order of the current idx being sampled
            decode_step: a tuple representing the current idx being sampled
        """
        x = self.right_shift(x, decode_step)
        x = self.pos_embd(x, decode_step, decode_idx)
        for net in self.attn_nets:
            x = net(x, frame_cond, decode_step, decode_idx)

        return x


class AttentionBlock(nn.Module):
    def __init__(self, shape, embd_dim, n_head, n_layer, dropout,
                 attn_type, dist_mask, attn_dropout, frame_cond_shape):
        super().__init__()
        self.use_frame_cond = frame_cond_shape is not None

        self.pre_attn_norm = nn.LayerNorm(embd_dim)
        self.post_attn_dp = nn.Dropout(dropout)
        self.attn = MultiHeadAttention(shape, embd_dim, embd_dim, n_head,
                                       n_layer, causal=True, attn_type=attn_type,
                                       dist_mask = dist_mask,
                                       attn_kwargs=dict(attn_dropout=attn_dropout))

        if frame_cond_shape is not None:
            enc_len = np.prod(frame_cond_shape[:-1])
            self.pre_enc_norm = nn.LayerNorm(embd_dim)
            self.post_enc_dp = nn.Dropout(dropout)
            self.enc_attn = MultiHeadAttention(shape, embd_dim, frame_cond_shape[-1],
                                               n_head, n_layer, attn_type='full',
                                               dist_mask = dist_mask,
                                               attn_kwargs=dict(attn_dropout=0.), causal=False)

        self.pre_fc_norm = nn.LayerNorm(embd_dim)
        self.post_fc_dp = nn.Dropout(dropout)
        self.fc_block = nn.Sequential(
            nn.Linear(in_features=embd_dim, out_features=embd_dim * 4),
            GeLU2(),
            nn.Linear(in_features=embd_dim * 4, out_features=embd_dim),
        )

    def forward(self, x, frame_cond, decode_step, decode_idx):
        h = self.pre_attn_norm(x)
        if self.training:
            h = checkpoint(self.attn, h, h, h, decode_step, decode_idx)
        else:
            h = self.attn(h, h, h, decode_step, decode_idx)
        h = self.post_attn_dp(h)
        x = x + h

        if self.use_frame_cond:
            h = self.pre_enc_norm(x)
            if self.training:
                h = checkpoint(self.enc_attn, h, frame_cond, frame_cond,
                               decode_step, decode_idx)
            else:
                h = self.enc_attn(h, frame_cond, frame_cond,
                                  decode_step, decode_idx)
            h = self.post_enc_dp(h)
            x = x + h

        h = self.pre_fc_norm(x)
        if self.training:
            h = checkpoint(self.fc_block, h)
        else:
            h = self.fc_block(h)
        h = self.post_fc_dp(h)
        x = x + h

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, shape, dim_q, dim_kv, n_head, n_layer,
                 causal, attn_type, dist_mask, attn_kwargs):
        super().__init__()
        self.causal = causal
        self.shape = shape

        self.d_k = dim_q // n_head
        self.d_v = dim_kv // n_head
        self.n_head = n_head

        self.w_qs = nn.Linear(dim_q, n_head * self.d_k, bias=False) # q
        self.w_qs.weight.data.normal_(std=1.0 / np.sqrt(dim_q))

        self.w_ks = nn.Linear(dim_kv, n_head * self.d_k, bias=False) # k
        self.w_ks.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.w_vs = nn.Linear(dim_kv, n_head * self.d_v, bias=False) # v
        self.w_vs.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.fc = nn.Linear(n_head * self.d_v, dim_q, bias=True) # c
        self.fc.weight.data.normal_(std=1.0 / np.sqrt(dim_q * n_layer))

        if attn_type == 'full':
            self.attn = FullAttention(shape, causal, dist_mask, **attn_kwargs)
        elif attn_type == 'axial':
            assert not causal, 'causal axial attention is not supported'
            self.attn = AxialAttention(len(shape), **attn_kwargs)
        elif attn_type == 'sparse':
            raise Exception('sparse attention is not supported')
        self.cache = None

    def forward(self, q, k, v, decode_step=None, decode_idx=None):
        """ Compute multi-head attention
        Args
            q, k, v: a [b, d1, ..., dn, c] tensor or
                     a [b, 1, ..., 1, c] tensor if decode_step is not None

        Returns
            The output after performing attention
        """

        # compute k, q, v
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = view_range(self.w_qs(q), -1, None, (n_head, d_k))
        k = view_range(self.w_ks(k), -1, None, (n_head, d_k))
        v = view_range(self.w_vs(v), -1, None, (n_head, d_v))

        # b x n_head x seq_len x d
        # (b, *d_shape, n_head, d) -> (b, n_head, *d_shape, d)
        q = shift_dim(q, -2, 1)
        k = shift_dim(k, -2, 1)
        v = shift_dim(v, -2, 1)

        # fast decoding
        if decode_step is not None:
            if decode_step == 0:
                if self.causal:
                    k_shape = (q.shape[0], n_head, *self.shape, self.d_k)
                    v_shape = (q.shape[0], n_head, *self.shape, self.d_v)
                    self.cache = dict(k=torch.zeros(k_shape, dtype=k.dtype, device=q.device),
                                    v=torch.zeros(v_shape, dtype=v.dtype, device=q.device))
                else:
                    # cache only once in the non-causal case
                    self.cache = dict(k=k.clone(), v=v.clone())
            if self.causal:
                idx = (slice(None, None), slice(None, None), *[slice(i, i+ 1) for i in decode_idx])
                self.cache['k'][idx] = k
                self.cache['v'][idx] = v
            k, v = self.cache['k'], self.cache['v']

        a = self.attn(q, k, v, decode_step, decode_idx)

        # (b, *d_shape, n_head, d) -> (b, *d_shape, n_head * d)
        a = shift_dim(a, 1, -2).flatten(start_dim=-2)
        a = self.fc(a) # (b x seq_len x embd_dim)

        return a

############## Attention #######################
class FullAttention(nn.Module):
    def __init__(self, shape, causal, dist_mask, attn_dropout):
        super().__init__()
        self.causal = causal
        self.attn_dropout = attn_dropout

        seq_len = np.prod(shape)
        if self.causal:
            self.register_buffer('mask', torch.tril(torch.ones(seq_len, seq_len)))
        
        self.distance_mask = None
        if dist_mask:
            self.distance = torch.zeros(seq_len, seq_len)
            idxs = list(itertools.product(*[range(s) for s in shape]))
            
            for i in range(0,seq_len):
                for j in range(i,seq_len):
                    self.distance[i][j] = sum(abs(idxs[i][k]-idxs[j][k]) for k in range(0,len(shape))) #l1 distance
                    self.distance[j][i] = self.distance[i][j]
            max_distance = self.distance[0][seq_len-1]
            self.distance_mask = torch.exp(-self.distance/max_distance)

    def forward(self, q, k, v, decode_step, decode_idx):
        mask = self.mask if self.causal else None
        if decode_step is not None and mask is not None:
            mask = mask[[decode_step]]

        old_shape = q.shape[2:-1]
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        out = scaled_dot_product_attention(q, k, v, mask=mask,
                                           attn_dropout=self.attn_dropout,
                                           training=self.training,
                                           distance_mask=self.distance_mask)

        return view_range(out, 2, 3, old_shape)

class AxialAttention(nn.Module):
    def __init__(self, n_dim, axial_dim):
        super().__init__()
        if axial_dim < 0:
            axial_dim = 2 + n_dim + 1 + axial_dim
        else:
            axial_dim += 2 # account for batch, head, dim
        self.axial_dim = axial_dim

    def forward(self, q, k, v, decode_step, decode_idx):
        q = shift_dim(q, self.axial_dim, -2).flatten(end_dim=-3)
        k = shift_dim(k, self.axial_dim, -2).flatten(end_dim=-3)
        v = shift_dim(v, self.axial_dim, -2)
        old_shape = list(v.shape)
        v = v.flatten(end_dim=-3)

        out = scaled_dot_product_attention(q, k, v, training=self.training)
        out = out.view(*old_shape)
        out = shift_dim(out, -2, self.axial_dim)
        return out


################ Spatiotemporal broadcasted positional embeddings ###############
class AddBroadcastPosEmbed(nn.Module):
    def __init__(self, shape, embd_dim, dim=-1):
        super().__init__()
        assert dim in [-1, 1] # only first or last dim supported
        self.shape = shape
        self.n_dim = n_dim = len(shape)
        self.embd_dim = embd_dim
        self.dim = dim

        assert embd_dim % n_dim == 0, f"{embd_dim} % {n_dim} != 0"
        self.emb = nn.ParameterDict({
             f'd_{i}': nn.Parameter(torch.randn(shape[i], embd_dim // n_dim) * 0.01
                                    if dim == -1 else
                                    torch.randn(embd_dim // n_dim, shape[i]) * 0.01)
             for i in range(n_dim)
        })

    def forward(self, x, decode_step=None, decode_idx=None):
        embs = []
        for i in range(self.n_dim):
            e = self.emb[f'd_{i}']
            if self.dim == -1:
                # (1, 1, ..., 1, self.shape[i], 1, ..., -1)
                e = e.view(1, *((1,) * i), self.shape[i], *((1,) * (self.n_dim - i - 1)), -1)
                e = e.expand(1, *self.shape, -1)
            else:
                e = e.view(1, -1, *((1,) * i), self.shape[i], *((1,) * (self.n_dim - i - 1)))
                e = e.expand(1, -1, *self.shape)
            embs.append(e)

        embs = torch.cat(embs, dim=self.dim)
        if decode_step is not None:
            embs = tensor_slice(embs, [0, *decode_idx, 0],
                                [x.shape[0], *(1,) * self.n_dim, x.shape[-1]])
        return x + embs

################# Helper Functions ###################################
def scaled_dot_product_attention(q, k, v, mask=None, attn_dropout=0., training=True, distance_mask=None):
    # Performs scaled dot-product attention over the second to last dimension dn

    # (b, n_head, d1, ..., dn, d)
    attn = torch.matmul(q, k.transpose(-1, -2))
    attn = attn / np.sqrt(q.shape[-1])
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float('-inf'))

    if distance_mask is not None:
        if distance_mask.shape == attn.shape[-2:]:
            attn = torch.mul(distance_mask.type_as(attn),attn)
        else:
            t = int(distance_mask.shape[0]/ attn.shape[-2])
            hw =  attn.shape[-2]
            
            distance_mask_modified = distance_mask[0:attn.shape[-2],0:attn.shape[-1]]
            attn = torch.mul(distance_mask_modified.type_as(attn),attn)

    attn_float = F.softmax(attn, dim=-1)
    attn = attn_float.type_as(attn) # b x n_head x d1 x ... x dn x d
    attn = F.dropout(attn, p=attn_dropout, training=training)

    a = torch.matmul(attn, v) # b x n_head x d1 x ... x dn x d

    return a


class RightShift(nn.Module):
    def __init__(self, embd_dim):
        super().__init__()
        self.embd_dim = embd_dim
        self.sos = nn.Parameter(torch.FloatTensor(embd_dim).normal_(std=0.02), requires_grad=True)

    def forward(self, x, decode_step):
        if decode_step is not None and decode_step > 0:
            return x

        x_shape = list(x.shape)
        x = x.flatten(start_dim=1, end_dim=-2) # (b, seq_len, embd_dim)
        sos = torch.ones(x_shape[0], 1, self.embd_dim, dtype=torch.float32).to(self.sos) * self.sos
        sos = sos.type_as(x)
        x = torch.cat([sos, x[:, :-1, :]], axis=1)
        x = x.view(*x_shape)

        return x


class GeLU2(nn.Module):
    def forward(self, x):
        return (1.702 * x).sigmoid() * x

