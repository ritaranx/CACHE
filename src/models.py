#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

"""
This script contains all models in our paper.
"""

import torch

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv
from layers import *

import math

from torch_scatter import scatter
from torch_geometric.utils import softmax

import numpy as np


class SetGNN(nn.Module):
    def __init__(self, args, norm=None):
        super(SetGNN, self).__init__()
        """
        args should contain the following:
        V_in_dim, V_enc_hid_dim, V_dec_hid_dim, V_out_dim, V_enc_num_layers, V_dec_num_layers
        E_in_dim, E_enc_hid_dim, E_dec_hid_dim, E_out_dim, E_enc_num_layers, E_dec_num_layers
        All_num_layers,dropout
        !!! V_in_dim should be the dimension of node features
        !!! E_out_dim should be the number of classes (for classification)
        """

        self.All_num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.aggr = args.aggregate
        self.NormLayer = args.normalization
        self.InputNorm = True

        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()
        self.bnV2Es = nn.ModuleList()
        self.bnE2Vs = nn.ModuleList()


        if self.All_num_layers == 0:
            self.classifier = MLP(in_channels=args.num_features,
                                  hidden_channels=args.Classifier_hidden,
                                  out_channels=args.num_labels,
                                  num_layers=args.Classifier_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False)
        else:
            self.V2EConvs.append(HalfNLHconv(in_dim=args.feature_dim,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA))
            self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
            for i in range(self.All_num_layers):
                self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))
                self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
                self.V2EConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))
                if i < self.All_num_layers-1:
                    self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
            self.classifier = MLP(
                                  # in_channels=args.MLP_hidden,
                                  in_channels=args.MLP_hidden * (args.All_num_layers + 1),
                                  hidden_channels=args.Classifier_hidden,
                                  out_channels=args.num_labels,
                                  num_layers=args.Classifier_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False)

    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, data, edge_weight=None):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
        #             The data should contain the follows
        #             data.x: node features
        #             data.V2Eedge_index:  edge list (of size (2,|E|)) where
        #             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges

        x, edge_index, norm = data.x, data.edge_index, data.norm

        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack(
            [edge_index[1], edge_index[0]], dim=0)

        vec = []
        x = F.dropout(x, p=0.2, training=self.training)  # Input dropout

        scale = 1
        eps = 1e-5
        for i, _ in enumerate(self.E2VConvs):
            x, weight_tuple = self.V2EConvs[i](x, edge_index, norm, self.aggr, edge_weight=edge_weight)
            # PairNorm
            x = x - x.mean(dim=0, keepdim=True)
            x = scale * x / (eps + x.pow(2).sum(-1).mean()).sqrt()
            # Jumping Knowledge
            vec.append(x)
            x = self.bnV2Es[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x, weight_tuple = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr, edge_weight=edge_weight)
            # PairNorm
            x = x - x.mean(dim=0, keepdim=True)
            x = scale * x / (eps + x.pow(2).sum(-1).mean()).sqrt()
            node_feat = x
            x = self.bnE2Vs[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x, weight_tuple = self.V2EConvs[-1](x, edge_index, norm, self.aggr, edge_weight=edge_weight)
        # PairNorm
        x = x - x.mean(dim=0, keepdim=True)
        x = scale * x / (eps + x.pow(2).sum(-1).mean()).sqrt()
        edge_feat = x
        # Jumping Knowledge
        vec.append(x)

        x = torch.cat(vec, dim=1)
        x = x[:data.y.shape[0], :]
        edge_score = self.classifier(x)

        return edge_score, edge_feat, node_feat, weight_tuple


class ViewLearner(torch.nn.Module):
    def __init__(self, encoder, input_dim, viewer_hidden_dim=64):
        super(ViewLearner, self).__init__()

        self.encoder = encoder
        self.input_dim = input_dim

        self.mlp_edge_model = nn.Sequential(
            Linear(self.input_dim * 2, viewer_hidden_dim),
            nn.ReLU(),
            Linear(viewer_hidden_dim, 1)
        )
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data, device):

        _, edge_feat, node_feat, _ = self.encoder(data.clone())

        totedges = data.totedges
        num_hyperedges = data.num_hyperedges[0]
        num_self_loop = totedges - num_hyperedges
        edge_index = data.edge_index.clone()
        num_self_loop_clone = num_self_loop.clone()
        node, edge = edge_index[:, :-num_self_loop_clone][0], edge_index[:, :-num_self_loop_clone][1]
        emb_node = node_feat[node]
        emb_edge = edge_feat[edge]

        total_emb = torch.cat([emb_node, emb_edge], 1)
        edge_weight = self.mlp_edge_model(total_emb)

        self_loop_weight = np.ones(shape=(num_self_loop_clone, 1)) * 10.0
        self_loop_weight = torch.FloatTensor(self_loop_weight).to(device)
        weight_logits = torch.cat([edge_weight, self_loop_weight], 0)

        return weight_logits
