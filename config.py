exp_cfg = dict(
    tag='debug',
    description='',
    seed_list=[100, 200],
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
    # 64 for DBLP, 8 for IMDB
    batch_size=8,
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
        apply=False,
        mb_cfg=dict(
            memory=1,
            warm_up=2,
        ),
        loss_cfg=dict(
            threshold=0.2,
            penalty_weight=[0.5, 0.05],
        ),
    )
)

data_cfg = dict(
    # 'DBLP', 'IMDB'
    dataset='IMDB',
    split_cfg=dict(
        split_seed=-1,
        split=[400, 400, 3257],
    ),
    noise_cfg=dict(
        apply=False,
        pair_flip_rate=0.0,
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