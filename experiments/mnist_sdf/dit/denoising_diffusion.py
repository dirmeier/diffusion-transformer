from typing import Any

import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from jax import random as jr


class DenoisingDiffusion(nn.Module):
    score_model: nn.Module
    parameterization: Any
    n_sampling_steps: int = 25

    def __call__(self, method, **kwargs):
        return getattr(self, method)(**kwargs)

    def loss(self, inputs, is_training, **kwargs):
        new_shape = (-1,) + tuple(
            np.ones(inputs.ndim - 1, dtype=np.int32).tolist()
        )

        epsilon = jr.normal(self.make_rng("sample"), (inputs.shape[0],))
        sigma = self.parameterization.sigma(epsilon)

        noise = jr.normal(self.make_rng("sample"), inputs.shape)
        noise = noise * sigma.reshape(new_shape)
        target_hat = self._denoise(
            inputs + noise,
            sigma=sigma,
            is_training=is_training,
        )

        loss = jnp.sum(jnp.square(inputs - target_hat), axis=(1, 2, 3))
        loss_weight = self.parameterization.loss_weight(sigma)
        return loss * loss_weight

    def sample(self, sample_shape, is_training=False, **kwargs):
        n = sample_shape[0]
        sigmas = self.parameterization.sampling_sigmas(self.n_sampling_steps)
        noise = jr.normal(self.make_rng("sample"), sample_shape) * sigmas[0]

        sample_next = noise
        for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            sample_curr = sample_next
            pred_curr = self._denoise(
                sample_curr,
                sigma=jnp.repeat(sigma, n),
                is_training=is_training,
            )
            d_cur = (sample_curr - pred_curr) / sigma
            sample_next = sample_curr + d_cur * (sigma_next - sigma)

            # second order correction
            if i < self.n_sampling_steps - 1:
                pred_next = self._denoise(
                    sample_next,
                    sigma=jnp.repeat(sigma_next, n),
                    is_training=is_training,
                )
                d_prime = (sample_next - pred_next) / sigma_next
                sample_next = sample_curr + (sigma_next - sigma) * (
                    0.5 * d_cur + 0.5 * d_prime
                )
        return sample_next

    def _denoise(self, sample, sigma, is_training):
        # taken from https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/generate.py#L69
        new_shape = (-1,) + tuple(
            np.ones(sample.ndim - 1, dtype=np.int32).tolist()
        )
        # apply scaling
        inputs = sample * self.parameterization.in_scaling(sigma).reshape(
            new_shape
        )
        context = self.parameterization.noise_conditioning(sigma)
        out = self.score_model(inputs, context, is_training=is_training)
        # output scaling and skip connection
        skip = sample * self.parameterization.skip_scaling(sigma).reshape(
            new_shape
        )
        outputs = out * self.parameterization.out_scaling(sigma).reshape(
            new_shape
        )
        outputs = skip + outputs
        return outputs
