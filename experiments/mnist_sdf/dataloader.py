import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from jax import numpy as jnp
from jax import random as jr
from scipy.ndimage import distance_transform_edt


def distance_transform(data):
    data[data < 0.5] = 0.0
    data[data >= 0.5] = 1.0

    neg_distances = distance_transform_edt(data)
    sd_img = data - 1.0
    sd_img = sd_img.astype(np.uint8)
    signed_distances = distance_transform_edt(sd_img) - neg_distances
    signed_distances /= float(data.shape[1])
    return signed_distances


def resize(data, size):
    data = jax.image.resize(
        data,
        (data.shape[0], size, size, data.shape[-1]),
        method="bicubic",
        antialias=True,
    )
    return np.copy(data)


def data_loaders(rng_key, config, split="train", outpath: str = None):
    datasets = tfds.load(
        "mnist", try_gcs=False, split=split, data_dir=outpath, batch_size=-1
    )
    if isinstance(split, str):
        datasets = [datasets]
    itrs = []
    for dataset in datasets:
        itr_key, rng_key = jr.split(rng_key)
        ds = np.float32(dataset["image"]) / 255.0
        ds = resize(ds, 32)
        for i in range(ds.shape[0]):
            ds[i] = distance_transform(ds[i])
        itr = _as_batched_numpy_iter(itr_key, ds, config)
        itrs.append(itr)
    return itrs


def _as_batched_numpy_iter(rng_key, itr, config):
    itr = tf.data.Dataset.from_tensor_slices(itr)
    max_int32 = jnp.iinfo(jnp.int32).max
    seed = jr.randint(rng_key, shape=(), minval=0, maxval=max_int32)
    return tfds.as_numpy(
        itr.shuffle(
            config.buffer_size,
            reshuffle_each_iteration=config.do_reshuffle,
            seed=int(seed),
        )
        .batch(config.batch_size, drop_remainder=True)
        .prefetch(config.batch_size * 5)
    )
