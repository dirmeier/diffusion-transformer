import numpy as np
from einops import rearrange
from flax import linen as nn
from jax import numpy as jnp

from dit.nn.embedding import timestep_embedding


def _modulate(inputs, shift, scale):
    return inputs * (1.0 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    hidden_size: int
    n_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, context, is_training, **kwargs):
        hidden = inputs
        adaln_norm = nn.Dense(self.hidden_size * 6)(context)
        attn, gate = jnp.split(adaln_norm, 2, axis=-1)

        pre_shift, pre_scale, post_scale = jnp.split(attn, 3, -1)
        intermediate = nn.LayerNorm(use_scale=False, use_bias=False)(hidden)
        intermediate = _modulate(intermediate, pre_shift, pre_scale)
        intermediate = nn.SelfAttention(num_heads=self.n_heads)(intermediate)
        hidden = hidden + post_scale[:, None] * intermediate

        pre_shift, pre_scale, post_scale = jnp.split(gate, 3, -1)
        intermediate = nn.LayerNorm(use_scale=False, use_bias=False)(hidden)
        intermediate = _modulate(intermediate, pre_shift, pre_scale)
        intermediate = nn.Sequential(
            [
                nn.Dense(self.hidden_size * 4),
                nn.gelu,
                lambda x: nn.Dropout(self.dropout_rate)(
                    x, deterministic=not is_training
                ),
                nn.Dense(self.hidden_size),
                lambda x: nn.Dropout(self.dropout_rate)(
                    x, deterministic=not is_training
                ),
            ]
        )(intermediate)
        outputs = hidden + post_scale[:, None] * intermediate

        return outputs


class DiT(nn.Module):
    n_channels: int
    n_out_channels: int
    patch_size: int
    n_blocks: int
    n_heads: int
    dropout_rate: float = 0.1

    def _time_embedding(self, times):
        times = timestep_embedding(times, self.n_channels * 2)
        times = nn.Sequential(
            [
                nn.Dense(self.n_channels),
                nn.swish,
                nn.Dense(self.n_channels),
                nn.swish,
            ]
        )(times)
        return times

    def _patchify(self, inputs):
        B, H, W, C = inputs.shape
        patch_tuple = (self.patch_size, self.patch_size)
        n_patches = H // self.patch_size
        hidden = nn.Conv(
            self.n_channels,
            patch_tuple,
            patch_tuple,
            padding="VALID",
            kernel_init=nn.initializers.xavier_uniform(),
        )(inputs)
        outputs = rearrange(
            hidden, "b h w c -> b (h w) c", h=n_patches, w=n_patches
        )
        return outputs

    def _unpatchify(self, inputs):
        B, HW, *_ = inputs.shape
        h = w = int(np.sqrt(HW))
        p = q = self.patch_size
        hidden = jnp.reshape(
            inputs,
            (B, h, w, p, q, self.n_out_channels),
        )
        outputs = rearrange(
            hidden, "b h w p q c -> b (h p) (w q) c", h=h, w=w, p=q, q=q
        )
        return outputs

    def _embed(self, inputs):
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        patch_embedding = self.param(
            "patch_embedding",
            nn.initializers.normal(stddev=1.0),
            pos_emb_shape,
            inputs.dtype,
        )
        return inputs + patch_embedding

    @nn.compact
    def __call__(self, inputs, times, is_training=True):
        hidden = self._patchify(inputs)
        hidden = self._embed(hidden)
        times = self._time_embedding(times)

        for i in range(self.n_blocks):
            hidden = DiTBlock(self.n_channels, self.n_heads, self.dropout_rate)(
                hidden, context=times, is_training=is_training
            )

        # final layer
        times = nn.Dense(
            self.n_channels * 2, kernel_init=nn.initializers.zeros
        )(times)
        times_shift, times_scale = jnp.split(times, 2, -1)
        hidden = nn.Sequential(
            [
                nn.LayerNorm(use_scale=False, use_bias=False),
                lambda x: _modulate(x, times_shift, times_scale),
                nn.Dense(
                    self.patch_size * self.patch_size * self.n_out_channels,
                    kernel_init=nn.initializers.zeros,
                ),
            ]
        )(hidden)
        outputs = self._unpatchify(hidden)
        return outputs
