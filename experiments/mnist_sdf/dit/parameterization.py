from jax import numpy as jnp


class EDMParameterization:
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    sigma_data: float = 0.5
    P_mean: float = -1.2
    P_std: float = 1.2
    S_churn: float = 40
    S_min: float = 0.05
    S_max: float = 50
    S_noise: float = 1.003

    def sigma(self, eps):
        return jnp.exp(eps * self.P_std + self.P_mean)

    def loss_weight(self, sigma):
        return (jnp.square(sigma) + jnp.square(self.sigma_data)) / jnp.square(
            sigma * self.sigma_data
        )

    def skip_scaling(self, sigma):
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def out_scaling(self, sigma):
        return sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5

    def in_scaling(self, sigma):
        return 1 / (sigma**2 + self.sigma_data**2) ** 0.5

    def noise_conditioning(self, sigma):
        return 0.25 * jnp.log(sigma + 1e-30)

    def sampling_sigmas(self, num_steps):
        rho_inv = 1 / self.rho
        step_idxs = jnp.arange(num_steps, dtype=jnp.float32)
        sigmas = (
            self.sigma_max**rho_inv
            + step_idxs
            / (num_steps - 1)
            * (self.sigma_min**rho_inv - self.sigma_max**rho_inv)
        ) ** self.rho
        return jnp.concatenate([sigmas, jnp.zeros_like(sigmas[:1])])

    def sigma_hat(self, sigma, num_steps):
        gamma = (
            jnp.minimum(self.S_churn / num_steps, 2**0.5 - 1)
            if self.S_min <= sigma <= self.S_max
            else 0
        )
        return sigma + gamma * sigma
