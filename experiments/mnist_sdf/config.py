import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.rng_key = 1
    config.model = new_dict(
        dit_score_model=new_dict(
            n_channels=384,
            n_out_channels=1,
            patch_size=4,
            n_blocks=12,
            n_heads=12,
        )
    )

    config.training = new_dict(
        n_epochs=200,
        batch_size=64,
        buffer_size=32 * 100,
        prefetch_size=32 * 4,
        do_reshuffle=True,
        early_stopping=new_dict(n_patience=10, min_delta=0.001),
        checkpoints=new_dict(
            max_to_keep=5,
            save_interval_steps=5,
        ),
        ema_rate=0.999,
    )

    config.optimizer = new_dict(
        name="adamw",
        params=new_dict(
            learning_rate=1e-4,
            weight_decay=1e-6,
            do_warmup=False,
            warmup_steps=2_000,
            do_decay=True,
            decay_steps=100_000,
            end_learning_rate=1e-5,
            do_gradient_clipping=False,
            gradient_clipping=1.0,
        ),
    )

    return config
