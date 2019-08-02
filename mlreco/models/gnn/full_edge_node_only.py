# GNN that attempts to match clusters to groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout, BatchNorm1d
from torch_geometric.nn import MetaLayer, GATConv

class FullEdgeNodeOnlyModel(torch.nn.Module):
    """
    Model that runs edge weights + node weights through a MLP for predictions
    
    """
    def __init__(self, cfg):
        super(FullEdgeNodeOnlyModel, self).__init__()
        
        if 'modules' in cfg:
                self.model_config = cfg['modules']['edge_only']
        else:
            self.model_config = cfg

        if 'leak' in self.model_config:
            self.leak = self.model_config['leak']
        else:
            self.leak = 0.1
        
        n_nf = int(self.model_config['n_node_features'])
        n_ef = int(self.model_config['n_edge_features'])
        
        class EdgeModel(torch.nn.Module):
            def __init__(self, leak):
                super(EdgeModel, self).__init__()
                
                self.leak = leak
                self.bn_node = BatchNorm1d(n_nf)
                self.bn_edge = BatchNorm1d(n_ef)

                self.edge_pred_mlp = Seq(
                    Lin(n_nf*2 + n_ef, 64),
                    LeakyReLU(self.leak),
                    Lin(64, 64),
                    LeakyReLU(self.leak),
                    Lin(64,32),
                    LeakyReLU(self.leak),
                    Lin(32,16),
                    LeakyReLU(self.leak),
                    Lin(16,8),
                    LeakyReLU(self.leak),
                    Lin(8,2)
                )

            def edge_pred_model(self, source, target, edge_attr, u, batch):
                out = torch.cat([source, target, edge_attr], dim=1)
                out = self.edge_pred_mlp(out)
                return out
            
            def forward(self, x, edge_index, e, u, batch):
                e = self.bn_edge(e)
                x = self.bn_node(x)
                return self.edge_pred_model(x, edge_index, e, u=None, batch=batch)

        self.edge_predictor = MetaLayer(EdgeModel(self.leak))
    
    def forward(self, x, edge_index, e, xbatch):
        x, e, u = self.edge_predictor(x, edge_index, e, u=None, batch=xbatch)
        return [[e]]
#         return {
#             'edge_pred': e
#         }