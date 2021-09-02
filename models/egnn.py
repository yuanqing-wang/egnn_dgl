import torch
import dgl
from typing import Callable

class EGNNLayer(torch.nn.Module):
    """ Layer of E(n) Equivariant Graph Neural Networks.

    Parameters
    ----------
    in_features : int
        Input features.

    out_features : int
        Output features.

    edge_features : int
        Edge features.

    References
    ----------
    [1] Satorras, E.G. et al. "E(n) Equivariant Graph Neural Networks"
    https://arxiv.org/abs/2102.09844

    [2] https://github.com/vgsatorras/egnn

    """
    def __init__(
        self,
        in_features : int,
        hidden_features: int,
        out_features : int,
        edge_features : int=0,
        activation : Callable=torch.nn.SiLU(),
    ):
        super(EGNNLayer, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.edge_features = edge_features
        self.activation = activation

        self.coordinate_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, 1)
        )

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(
                in_features * 2 + 1 + edge_features,
                hidden_features
            ),
            activation,
            torch.nn.Linear(hidden_features, hidden_features),
            activation,
        )

        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_features + in_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, out_features)
        )

    def _edge_model(self, edge):
        return {"h_e":
            self.edge_mlp(
                torch.cat(
                    [
                        edge.src["h_v"],
                        edge.dst["h_v"],
                        (edge.src["x"] - edge.dst["x"]) ** 2,
                    ],
                    dim=-1
                )
            )
        }

    def _node_model(self, node):
        return {"h_v":
            self.node_mlp(
                torch.cat(
                    [
                        node.data["h_v"],
                        node.data["h_agg"],
                    ]
                )
            )
        }

    def _coordinate_edge_model(self, edge):
        return {
            "x_e": (edge.src["x"] - edge.dst["x"])
            * self.coordinate_mlp(edge.data["h_e"])
        }

    def _coordinate_node_model(self, node):
        return {
            "x": node["x"] + node.data["x_v_agg"],
        }

    def forward(self, graph, feat, coordinate):
        """ Forward pass.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Input graph.

        feat : torch.Tensor
            Input features.

        coordinate : torch.Tensor
            Input coordinates.

        Returns
        -------
        torch.Tensor : Output features.

        torch.Tensor : Output coordinates.
        """
        # get local copy of the graph
        graph = graph.local_var()

        # put features and coordinates into graph
        graph.ndata["h_v"], graph.ndata["x"] = feat, coordinate

        # apply representation update on edge
        # Eq. 3 in "E(n) Equivariant Graph Neural Networks"
        graph.apply_edges(func=self._edge_model)

        # apply coordinate update on edge
        graph.apply_edges(func=self._coordinate_edge_model)

        # aggregate coordinate update
        graph.update_all(
            dgl.function.copy_e("x_e", "x_msg"),
            dgl.function.sum("x_msg", "x_v_agg"),
        )

        # apply coordinate update on nodes
        graph.apply_nodes(func=self._coordinate_node_model)

        # apply representation update on nodes
        graph.apply_nodes(func=self._node_model)

        # pull features
        feat = graph.ndata["x"]
        coordinate = graph.ndata["h_v"]

        return feat, coordinate

class EGNN(torch.nn.Module):
    """ E(n) Equivariant Graph Neural Networks.

    Parameters
    ----------

    References
    ----------
    [1] Satorras, E.G. et al. "E(n) Equivariant Graph Neural Networks"
    https://arxiv.org/abs/2102.09844

    """
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        depth=4,
        edge_features=0,
        activation=torch.nn.SiLU(),
    ):
        super(EGNN, self).__init__()
