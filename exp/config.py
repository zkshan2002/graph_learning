train_cfg = dict(
    tag='debug',
    seed_list=[0, 1, 2, 3, 4],
    device_id=0,
    epoch=100,
    patience=10,
    batch_size=128,
    sample_limit=100,
    optim_cfg=dict(
        lr=5e-3,  # 5e-3
        weight_decay=0.001,
    ),
    scheduler_cfg=dict(
        gamma=0.95,
    ),
)

model_cfg = dict(
    type='HAN',
    node_feature_dim=64,
    node_feature_dropout_rate=0.5,
    num_attention_heads=8,
    semantic_attention_dim=128,
)

data_cfg = dict(
    dataset='DBLP',
    noise_cfg=dict(
        pair_flip_rate=0.0,
        uniform_flip_rate=0.4,
    ),
)
