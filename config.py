exp_cfg = dict(
    tag='debug',
    description='',
    seed_list=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    device_id=0,
    evaluate_cfg=dict(
        svm_cfg=dict(
            train_ratio_list=[0.8],
        )
    ),
)

train_cfg = dict(
    epoch=100,
    patience=10,
    # 64 for DBLP, 4 for IMDB
    batch_size=4,
    # 512 for DBLP, 128 for IMDB
    sample_limit=128,
    optim_cfg=dict(
        # 5e-3 for DBLP, 2e-4 for IMDB
        lr=2e-4,
        weight_decay=0.001,
    ),
    scheduler_cfg=dict(
        gamma=0.95,
    ),
    sft_cfg=dict(
        apply_filtering=False,
        filtering_cfg=dict(
            memory=1,
            warmup=2,
        ),
        apply_loss=False,
        loss_cfg=dict(
            threshold=0.5,
            weight=[0.5, 0.1],
        ),
        apply_fixmatch=False,
        fixmatch_cfg=dict(),
    )
)

data_cfg = dict(
    # 'DBLP', 'IMDB'
    dataset='IMDB',
    split_cfg=dict(
        apply=True,
        seed=0,
        split_ratio=[0.1, 0.1, 0.8],
    ),
    noise_cfg=dict(
        apply=True,
        seed=0,
        pair_flip_rate=0.4,
        uniform_flip_rate=0.0,
    ),
)

model_cfg = dict(
    type='HAN',
    node_feature_dim=64,
    node_feature_dropout_rate=0.5,
    num_attention_heads=8,
    semantic_attention_dim=128,
    type_aware_semantic=True,
)

# model_cfg = dict(
#     type='MLP',
#     node_feature_dim=64,
#     node_feature_dropout_rate=0.5,
#     num_attention_heads=8,
#     hidden_dims=[256, 256],
# )