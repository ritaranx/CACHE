#!/usr/bin/env python
# coding: utf-8

# In[45]:


import torch
import pickle
import os
import ipdb

import os.path as osp
import numpy as np
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_sparse import coalesce


def load_dataset(path='../data/raw_data/', dataset='mimic3',
                       node_feature_path="../data/mimic3/node-embeddings-mimic3",
                       num_node=7423):
    '''
    this will read the yelp dataset from source files, and convert it edge_list to
    [[ -V- | -E- ]
     [ -E- | -V- ]]

    each node is a restaurant, a hyperedge represent a set of restaurants one user had been to.

    node features:
        - add gaussian noise with sigma = nosie, mean = one hot coded label.

    node label:
        - average stars from 2-10, converted from original stars which is binned in x.5, min stars = 1
    '''
    print(f'Loading hypergraph dataset from: {dataset}')

    # first load edge labels
    df_labels = pd.read_csv(osp.join(path, dataset, f'edge-labels-{dataset}.txt'), sep=',', header=None)
    num_edges = df_labels.shape[0]
    labels = df_labels.values

    # then create node features.
    with open(node_feature_path, 'r') as f:
        line = f.readline()
        print(line)
        n_node, embedding_dim = line.split(" ")
        features = np.random.rand(num_node, int(embedding_dim))
        for lines in f.readlines():
            values = list(map(float, lines.split(" ")))
            features[int(values[0]) - 1] = np.array(values[1:])

    num_nodes = features.shape[0]


    print(f'number of nodes:{num_nodes}, feature dimension: {features.shape[1]}')

    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)

    p2hyperedge_list = osp.join(path, dataset, f'hyperedges-{dataset}.txt')
    node_list = []
    he_list = []
    he_id = num_nodes

    with open(p2hyperedge_list, 'r') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            cur_set = line.split(',')
            cur_set = [int(x) for x in cur_set]

            node_list += cur_set
            he_list += [he_id] * len(cur_set)
            he_id += 1
    # shift node_idx to start with 0.
    node_idx_min = np.min(node_list)
    node_list = [x - node_idx_min for x in node_list]

    edge_index = [node_list + he_list,
                  he_list + node_list]

    edge_index = torch.LongTensor(edge_index)

    data = Data(x=features,
                edge_index=edge_index,
                y=labels)
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
                                               None,
                                               total_num_node_id_he_id,
                                               total_num_node_id_he_id)

    n_x = num_nodes
    data.n_x = n_x
    data.num_hyperedges = he_id - num_nodes

    return data


def save_data_to_pickle(data, p2root = '../data/', file_name = None):
    '''
    if file name not specified, use time stamp.
    '''
#     now = datetime.now()
#     surfix = now.strftime('%b_%d_%Y-%H:%M')
    surfix = 'star_expansion_dataset'
    if file_name is None:
        tmp_data_name = '_'.join(['Hypergraph', surfix])
    else:
        tmp_data_name = file_name
    p2he_StarExpan = osp.join(p2root, tmp_data_name)
    if not osp.isdir(p2root):
        os.makedirs(p2root)
    with open(p2he_StarExpan, 'bw') as f:
        pickle.dump(data, f)
    return p2he_StarExpan


class dataset_Hypergraph(InMemoryDataset):
    def __init__(self, root='../data/pyg_data/hypergraph_dataset/', name=None,
                 p2raw=None, transform=None, pre_transform=None, num_nodes=7423):
        
        existing_dataset = ['mimic3', 'cradle']
        if name not in existing_dataset:
            raise ValueError(f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name
        
        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(f'path to raw hypergraph dataset "{p2raw}" does not exist!')
        
        if not osp.isdir(root):
            os.makedirs(root)
            
        self.root = root
        self.myraw_dir = osp.join(root, self.name, 'raw')
        self.myprocessed_dir = osp.join(root, self.name, 'processed')
        self.num_nodes = num_nodes
        super(dataset_Hypergraph, self).__init__(osp.join(root, name), transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        
    # @property
    # def raw_dir(self):
    #     return osp.join(self.root, self.name, 'raw')

    # @property
    # def processed_dir(self):
    #     return osp.join(self.root, self.name, 'processed')


    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        file_names = ['data.pt']
        return file_names

    @property
    def num_features(self):
        return self.data.num_node_features


    def download(self):
        for name in self.raw_file_names:
            p2f = osp.join(self.myraw_dir, name)
            if not osp.isfile(p2f):
                # file not exist, so we create it and save it there.
                print(p2f)
                print(self.p2raw)
                print(self.name)

                if self.name in ['mimic3']:
                    tmp_data = load_dataset(path=self.p2raw,
                        dataset=self.name,
                        node_feature_path="../data/raw_data/mimic3/node-embeddings-mimic3", num_node=self.num_nodes)

                elif self.name in ['cradle']:
                    tmp_data = load_dataset(path=self.p2raw,
                        dataset=self.name,
                        node_feature_path="../data/raw_data/cradle/node-embeddings-cradle", num_node=self.num_nodes)
                    
                _ = save_data_to_pickle(tmp_data, 
                                          p2root = self.myraw_dir,
                                          file_name = self.raw_file_names[0])
            else:
                # file exists already. Do nothing.
                pass

    def process(self):
        p2f = osp.join(self.myraw_dir, self.raw_file_names[0])
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)
