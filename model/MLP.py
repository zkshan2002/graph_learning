import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.functional as dgl_F

from typing import List


# todo: support multiple control type
class MLP(nn.Module):
    def __init__(
            self, node_raw_feature_dim_list: List[int], node_feature_dim, node_feature_dropout_rate,
            num_attention_heads, hidden_dims: List[int], num_cls,
            device, **kwargs
    ):
        super(MLP, self).__init__()
        self.device = device

        # node feature transform
        self.node_feature_dim = node_feature_dim

        node_feature_projector_list = []
        for node_raw_feature_dim in node_raw_feature_dim_list:
            node_feature_projector = nn.Linear(node_raw_feature_dim, node_feature_dim)
            nn.init.xavier_normal_(node_feature_projector.weight, gain=np.sqrt(2))
            node_feature_projector_list.append(node_feature_projector)
        self.node_feature_projector_list = nn.ModuleList(node_feature_projector_list)

        if node_feature_dropout_rate > 0:
            self.node_feature_dropout = nn.Dropout(node_feature_dropout_rate)
        else:
            self.node_feature_dropout = lambda x: x

        mlp = []
        for in_dim, out_dim in zip(
                [self.node_feature_dim] + hidden_dims,
                hidden_dims[:-1] + [node_feature_dim * num_attention_heads]
        ):
            fc = nn.Linear(in_dim, out_dim)
            nn.init.xavier_normal_(fc.weight, gain=np.sqrt(2))
            mlp.append(fc)
            mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp[:-1])

        # cls head
        self.cls_head = nn.Linear(node_feature_dim * num_attention_heads, num_cls, bias=True)
        nn.init.xavier_normal_(self.cls_head.weight, gain=np.sqrt(2))

        self.to(device)
        return

    def forward(
            self, target_nodes: torch.tensor, metapath_list: List[torch.tensor],
            node_type_mapping: torch.tensor, node_feature_list: List[torch.tensor]
    ):
        # node feature transform
        num_nodes = node_type_mapping.shape[0]
        node_features = torch.zeros((num_nodes, self.node_feature_dim), device=self.device)
        for node_type, fc in enumerate(self.node_feature_projector_list):
            node_indices = torch.where(node_type_mapping == node_type)[0]
            node_features[node_indices] = fc(node_feature_list[node_type])
        node_features = self.node_feature_dropout(node_features)

        embeddings = self.mlp(node_features)[target_nodes]

        cls_logits = self.cls_head(embeddings)

        return cls_logits, embeddings
