# GNN that attempts to match clusters to groups
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, LeakyReLU, Dropout, BatchNorm1d
from torch_geometric.nn import MetaLayer, NNConv

class FullNNConvModel(torch.nn.Module):
    """
    Simple GNN with several edge convolutions, followed by MetLayer for edge prediction
    
    for use in config
    model:
        modules:
            edge_model:
              name: nnconv
    """
    def __init__(self, cfg):
        super(FullNNConvModel, self).__init__()
        
        
        if 'modules' in cfg:
            self.model_config = cfg['modules']['edge_model']
        else:
            self.model_config = cfg
            
        
        self.aggr = self.model_config.get('aggr', 'add')
        self.leak = self.model_config.get('leak', 0.1)
        
        n_nf = int(self.model_config['n_node_features'])
        n_ef = int(self.model_config['n_edge_features'])
        
        self.bn_node = BatchNorm1d(n_nf)
        self.bn_edge = BatchNorm1d(n_ef)

        # go from 16 to 32 node features
        ninput = n_nf
        noutput = n_nf*2
        self.nn1 = Seq(
            Lin(n_ef, ninput),
            LeakyReLU(self.leak),
            Lin(ninput, ninput*noutput),
            LeakyReLU(self.leak)
        )
        self.layer1 = NNConv(ninput, noutput, self.nn1, aggr=self.aggr)

#         # go from 32 to 64 node features
#         ninput = n_nf*2
#         noutput = n_nf*4
#         self.nn2 = Seq(
#             Lin(n_ef, ninput),
#             LeakyReLU(self.leak),
#             Lin(ninput, ninput*noutput),
#             LeakyReLU(self.leak)
#         )
#         self.layer2 = NNConv(ninput, noutput, self.nn2, aggr=self.aggr)
        
        class EdgeModel(torch.nn.Module):
            def __init__(self, leak, aggr):
                super(EdgeModel, self).__init__()
                
                self.leak = leak
                self.aggr = aggr
                

                # final prediction layer
                self.edge_pred_mlp = Seq(Lin(2*noutput + n_ef, 64),
                                         LeakyReLU(self.leak),
                                         Lin(64, 32),
                                         LeakyReLU(self.leak),
                                         Lin(32, 16),
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
                
                return self.edge_pred_model(x, edge_index, e, u=None, batch=batch)

        self.edge_predictor = MetaLayer(EdgeModel(self.leak, self.aggr))
        
        
    def forward(self, x, edge_index, e, xbatch):
        """
        inputs data:
            x - vertex features
            edge_index - graph edge list
            e - edge features
            xbatch - node batchid
        """
        e = self.bn_edge(e)
        x = self.bn_node(x)
        x = self.layer1(x, edge_index, e)
#         x = self.layer2(x, edge_index, e)
        x, e, u = self.edge_predictor(x, edge_index, e, u=None, batch=xbatch)
        return [[e]]