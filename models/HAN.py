import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.functional as dgl_F

from typing import List


# todo: support multiple control type
class HAN(nn.Module):
    def __init__(
            self, node_feature_dim_list: List[int], node_feature_dim, node_feature_dropout_rate,
            num_attention_heads, num_metapaths, semantic_attention_dim, num_cls,
            device, type_aware_semantic=False, leaky_relu_slope=0.01,
    ):
        super(HAN, self).__init__()
        self.device = device

        # node feature transform
        self.node_feature_dim = node_feature_dim

        node_feature_projector_list = []
        for node_raw_feature_dim in node_feature_dim_list:
            node_feature_projector = nn.Linear(node_raw_feature_dim, node_feature_dim)
            # nn.init.xavier_normal_(node_feature_projector.weight, gain=np.sqrt(2))
            node_feature_projector_list.append(node_feature_projector)
        self.node_feature_projector_list = nn.ModuleList(node_feature_projector_list)

        if node_feature_dropout_rate > 0:
            self.node_feature_dropout = nn.Dropout(node_feature_dropout_rate)
        else:
            self.node_feature_dropout = lambda x: x

        # node level attention
        self.num_metapaths = num_metapaths
        self.num_attention_heads = num_attention_heads

        node_attention_list = []
        for _ in range(num_metapaths):
            node_attention = nn.Parameter(torch.empty((1, num_attention_heads, node_feature_dim * 2), device=device))
            node_attention_list.append(node_attention)
        self.node_attention_list = nn.ParameterList(node_attention_list)
        self.activation_func = nn.LeakyReLU(leaky_relu_slope)

        # semantic level attention
        self.semantic_attention_dim = semantic_attention_dim
        self.type_aware_semantic = type_aware_semantic
        if type_aware_semantic:
            # node type-aware semantic projection
            num_node_types = len(node_feature_dim_list)
            semantic_projector_list = []
            semantic_attention_list = []
            for _ in range(num_node_types):
                semantic_projector = nn.Linear(node_feature_dim * num_attention_heads, semantic_attention_dim)
                semantic_projector_list.append(semantic_projector)

                semantic_attention = nn.Parameter(torch.empty((1, semantic_attention_dim), device=device))
                semantic_attention_list.append(semantic_attention)
            self.semantic_projector_list = nn.ModuleList(semantic_projector_list)
            self.semantic_attention_list = nn.ParameterList(semantic_attention_list)

        else:
            self.semantic_projector = nn.Linear(node_feature_dim * num_attention_heads, semantic_attention_dim)
            self.semantic_attention = nn.Parameter(torch.empty((1, semantic_attention_dim)))

        # cls head
        self.cls_head = nn.Linear(node_feature_dim * num_attention_heads, num_cls, bias=True)

        self.to(device)
        return

    def init_weights(self):

        # node feature transform
        for node_feature_projector in self.node_feature_projector_list:
            nn.init.xavier_normal_(node_feature_projector.weight, gain=np.sqrt(2))

        for node_attention in self.node_attention_list:
            nn.init.xavier_normal_(node_attention.data, gain=np.sqrt(2))

        # semantic level attention
        if self.type_aware_semantic:
            for semantic_projector in self.semantic_projector_list:
                nn.init.xavier_normal_(semantic_projector.weight, gain=np.sqrt(2))

            for semantic_attention in self.semantic_attention_list:
                nn.init.xavier_normal_(semantic_attention.data, gain=np.sqrt(2))
        else:
            nn.init.xavier_normal_(self.semantic_projector.weight, gain=np.sqrt(2))
            nn.init.xavier_normal_(self.semantic_attention.data, gain=np.sqrt(2))

        # cls head
        nn.init.xavier_normal_(self.cls_head.weight, gain=np.sqrt(2))

        return

    def forward(
            self, target_nodes: torch.tensor, metapath_list: List[torch.tensor],
            node_type_mapping: torch.tensor, node_feature_list: List[torch.tensor]
    ):
        # node feature transform
        num_nodes = node_type_mapping.shape[0]
        num_batch = target_nodes.shape[0]
        node_features = torch.zeros((num_nodes, self.node_feature_dim), device=self.device)
        for node_type, fc in enumerate(self.node_feature_projector_list):
            node_indices = torch.where(node_type_mapping == node_type)[0]
            node_features[node_indices] = fc(node_feature_list[node_type])
        node_features = self.node_feature_dropout(node_features)

        # node level attention
        h_metapath = []
        for metapath_id in range(self.num_metapaths):
            neighbor_node_indices = metapath_list[metapath_id][:, 0]
            current_node_indices = metapath_list[metapath_id][:, -1]

            # (batch, feat)
            edge_attention = torch.cat([
                node_features[current_node_indices], node_features[neighbor_node_indices]
            ], dim=-1)
            # -> (batch, head, 1)
            edge_attention = torch.sum(
                self.node_attention_list[metapath_id] * edge_attention.unsqueeze(dim=1), dim=-1, keepdim=True
            )
            edge_attention = self.activation_func(edge_attention)
            graph = dgl.graph((neighbor_node_indices, current_node_indices))
            edge_attention = dgl_F.edge_softmax(graph, edge_attention)
            graph.edata['src_node_feature'] = \
                node_features[neighbor_node_indices].unsqueeze(dim=1).expand(-1, self.num_attention_heads, -1)
            graph.edata['attention'] = edge_attention

            def message(edges):
                node_feature = edges.data['src_node_feature'] * edges.data['attention']
                return {'src_node_information': node_feature}

            graph.update_all(message, dgl.function.sum('src_node_information', 'node_feature'))
            node_feature = graph.ndata['node_feature'][target_nodes]
            node_feature = self.activation_func(node_feature)
            h_metapath.append(node_feature)
        # (path, batch, head, feat)
        h_metapath = torch.stack(h_metapath)

        # semantic level attention
        # all metapath schemes share the same target_node set(guaranteed by sample policy)
        beta = []
        for metapath_id in range(self.num_metapaths):
            if self.type_aware_semantic:
                # (batch, 1)
                metapath_attention = torch.zeros((num_batch, 1), device=self.device)
                for node_type, fc in enumerate(self.semantic_projector_list):
                    node_indices = torch.where(node_type_mapping[target_nodes] == node_type)[0]
                    # (mask, feat)
                    aux = fc(
                        h_metapath[metapath_id, node_indices].view(-1, self.node_feature_dim * self.num_attention_heads)
                    )
                    aux = torch.tanh(aux)
                    # -> (mask, 1)
                    metapath_attention[node_indices] = torch.sum(
                        self.semantic_attention_list[node_type] * aux, dim=-1, keepdim=True
                    )
            else:
                # (batch, feat)
                metapath_attention = self.semantic_projector(
                    h_metapath[metapath_id].view(-1, self.node_feature_dim * self.num_attention_heads)
                )
                metapath_attention = torch.tanh(metapath_attention)
                # -> (batch, 1)
                metapath_attention = torch.sum(self.semantic_attention * metapath_attention, dim=-1, keepdim=True)
            # -> (1, 1)
            metapath_attention = torch.mean(metapath_attention, dim=0, keepdim=True)
            beta.append(metapath_attention)
        # (path, 1, 1)
        beta = torch.stack(beta)
        beta = F.softmax(beta, dim=0)
        # (batch, head * feat)
        embeddings = torch.sum(
            beta.unsqueeze(dim=-1) * h_metapath, dim=0
        ).view(-1, self.node_feature_dim * self.num_attention_heads)

        cls_logits = self.cls_head(embeddings)

        return cls_logits, embeddings
