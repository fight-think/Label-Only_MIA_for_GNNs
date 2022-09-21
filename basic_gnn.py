#references
#https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/basic_gnn.py
#https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/gin.py
#https://github.com/mujm/CCS21_GNNattack/blob/master/Gin.py
import copy
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList

from torch_geometric.nn.conv import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    MessagePassing,
    PNAConv,
    SAGEConv,
)
from torch_geometric.data import Batch
from torch_geometric.nn.models import MLP
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
#from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj
from torch_geometric.nn import global_add_pool,global_mean_pool,global_max_pool


class BasicGNN(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models.
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        readout (str, optioinal): The type of aggregating features of node for the graph (pooling). 'add','mean','max'
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        norm: Optional[torch.nn.Module] = None,
        jk: Optional[str] = None,
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        readout: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.dropout = dropout
        #self.act = activation_resolver(act, **(act_kwargs or {}))
        self.act = F.relu
        self.jk_mode = jk
        self.act_first = act_first
        self.readout=readout

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels
        #no jk
        # self.convs = ModuleList()
        # self.convs.append(self.init_conv(in_channels, hidden_channels, **kwargs))
        # for _ in range(num_layers - 2):
        #     self.convs.append(self.init_conv(hidden_channels, hidden_channels, **kwargs))
        # self.convs.append(self.init_conv(hidden_channels, out_channels, **kwargs))

        # self.norms = None
        # if norm is not None:
        #     self.norms = ModuleList()
        #     for _ in range(num_layers-1):
        #         self.norms.append(copy.deepcopy(norm))
  
        self.convs = ModuleList()
        self.convs.append(
            self.init_conv(in_channels, hidden_channels, **kwargs))
        for _ in range(num_layers - 2):
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, **kwargs))
        if out_channels is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(
                self.init_conv(hidden_channels, out_channels, **kwargs))
        else:
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, **kwargs))

        self.norms = None
        if norm is not None:
            self.norms = ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(copy.deepcopy(norm))
            if jk is not None:
                self.norms.append(copy.deepcopy(norm))
  
        # self.linears_prediction = torch.nn.ModuleList()#maps the output of different layers into a prediction score
        # for layer in range(num_layers): 
        #     if layer == 0:
        #         self.linears_prediction.append(nn.Linear(in_channels, out_channels))
        #     else:
        #         self.linears_prediction.append(nn.Linear(hidden_channels, out_channels))
        #add one more layer for linears_prediction
        #self.linears_prediction.append(nn.Linear(hidden_channels, out_channels))

        self.jk=None
        self.lin=None

        if jk is not None and jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)
          
        if jk is not None:
            if jk == 'cat':
                self.lin=Linear(num_layers * hidden_channels,self.out_channels)
            else:
                self.lin = Linear(hidden_channels, self.out_channels)
            # self.lin = Linear(hidden_channels, self.out_channels)

        #for graph classification
        if self.readout is not None: #aggregate nodes in one graph
          if self.readout=='add':
            self.pool=global_add_pool
          elif self.readout=='mean':
            self.pool=global_mean_pool
          elif self.readout=='max':
            self.pool=global_max_pool
          else:
            raise NotImplementedError

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        raise NotImplementedError
    
    # def init_weight(self):
    #   for item in self.convs:
    #     torch.nn.init.xavier_normal_(item.weight, gain=1.0)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(self,data, *args, **kwargs) -> Tensor:  #x: Tensor, edge_index: Adj, *args, **kwargs
        """"""
        #x, edge_index, batch = data.x, data.edge_index, data.batch
        x, edge_index = data.x, data.edge_index

        if self.readout is not None: #for graph classification
          for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, *args, **kwargs)
            if self.act is not None and self.act_first and i<(self.num_layers-1):
                x = self.act(x)
            if self.norms is not None and i<(self.num_layers-1):
                x = self.norms[i](x)
            if self.act is not None and not self.act_first and i<(self.num_layers-1):
                x = self.act(x)
            if i<(self.num_layers-1):
              x = F.dropout(x, p=self.dropout, training=self.training)
          
          #not connect the output of all the layers
          batch=data.batch
          x = self.pool(x, batch)
          return F.log_softmax(x,dim=1)
          
        else:
          
          xs: List[Tensor] = []
          for i in range(self.num_layers):
              x = self.convs[i](x, edge_index, *args, **kwargs)
              if i == self.num_layers - 1 and self.jk_mode is None:
                  break
              if self.act is not None and self.act_first:
                  x = self.act(x)
              if self.norms is not None:
                  x = self.norms[i](x)
              if self.act is not None and not self.act_first:
                  x = self.act(x)
              x = F.dropout(x, p=self.dropout, training=self.training)
              if hasattr(self, 'jk'):
                  xs.append(x)
          if self.jk != None:
            x = self.jk(xs) if hasattr(self, 'jk') else x
          if self.lin != None:
            x = self.lin(x) if hasattr(self, 'lin') else x
          #return x
          return F.log_softmax(x, dim=1)

    
    def predict(self, data, device):
        #this is the prediction for single graph, for graph classification, not node classification
        self.eval() 
        graph = Batch.from_data_list([data]).to(device)
        output = self(graph)  #logits of graph: [[0.2,0.3,0.5]]
        pred = output.max(1, keepdim = True)[1] #final predicted label: [[1]]
        return pred[0][0]
        
    def predict_vector(self, data, device):
        #just for graph classification, not node classification
        self.eval()
        graph = Batch.from_data_list([data]).to(device)
        output = self(graph)
        vector = output[0]
        return torch.nn.functional.softmax(vector, dim=0)
        #return vector

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')


class GCN(BasicGNN):
    r"""The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, using the
    :class:`~torch_geometric.nn.conv.GCNConv` operator for message passing.
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"lstm"`).
            (default: :obj:`"last"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GCNConv`.
    """
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return GCNConv(in_channels, out_channels, **kwargs)


class GraphSAGE(BasicGNN):
    r"""The Graph Neural Network from the `"Inductive Representation Learning
    on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, using the
    :class:`~torch_geometric.nn.SAGEConv` operator for message passing.
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"lstm"`).
            (default: :obj:`"last"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.SAGEConv`.
    """
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return SAGEConv(in_channels, out_channels, **kwargs)


class GIN(BasicGNN):
    r"""The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, using the
    :class:`~torch_geometric.nn.GINConv` operator for message passing.
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :obj:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"lstm"`).
            (default: :obj:`"last"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GINConv`.
    """
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP([in_channels, out_channels, out_channels], batch_norm=True)
        return GINConv(mlp, **kwargs)


class GAT(BasicGNN):
    r"""The Graph Neural Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
    :class:`~torch_geometric.nn.GATConv` or
    :class:`~torch_geometric.nn.GATv2Conv` operator for message passing,
    respectively.
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        v2 (bool, optional): If set to :obj:`True`, will make use of
            :class:`~torch_geometric.nn.conv.GATv2Conv` rather than
            :class:`~torch_geometric.nn.conv.GATConv`. (default: :obj:`False`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"lstm"`).
            (default: :obj:`"last"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.GATv2Conv`.
    """
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:

        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(in_channels, out_channels, heads=heads, concat=concat,
                    dropout=self.dropout, **kwargs)


class PNA(BasicGNN):
    r"""The Graph Neural Network from the `"Principal Neighbourhood Aggregation
    for Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper, using the
    :class:`~torch_geometric.nn.conv.PNAConv` operator for message passing.
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"lstm"`).
            (default: :obj:`"last"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.PNAConv`.
    """
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return PNAConv(in_channels, out_channels, **kwargs)


__all__ = ['GCN', 'GraphSAGE', 'GIN', 'GAT', 'PNA']
