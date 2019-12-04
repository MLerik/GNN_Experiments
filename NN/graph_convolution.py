import torch
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class MsgPassLayer(MessagePassing):
    def __init__(self, in_dim, out_dim):
        # Message passing with max aggregation.
        super(MsgPassLayer, self).__init__(aggr='add')
        self.mlp = MLP(in_dim, out_dim)
        self.update_mlp = MLP(2 * in_dim, out_dim)

    def forward(self, x, edge_index):
        """
        Forward pass of MsgPassLayer

        Parameters
        ----------
        x: Features on Nodes in Graph
        edge_index: Edge indexes of graph

        Returns
        -------
        Returns the state of the graph after message passing
        """
        # Add self loops to nodes
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Start propagating messages.
        return self.propagate(edge_index=edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, x_i, flow="source_to_target"):
        """
        Generate message to be passed between node. Messages are generated using a MLP
        Parameters
        ----------
        x_j : Feature vector of source node
        x_i: Feature vector of target node
        flow: Flow direction of message

        Returns
        -------
        Output from MLP
        """
        #tmp_x = torch.cat([x_j, x_i], dim=1)
        tmp_x = x_j+x_i
        return self.mlp(tmp_x)  # 0.5 *  x_j

    def update(self, aggr_out, x):
        # Todo: Understand how to properly use this
        # aggr_out has shape [N, out_channels]
        # new_embedding = torch.cat([aggr_out, x], dim=1)
        # new_embedding = self.update_mlp(new_embedding)
        # Step 5: Return new node embeddings.
        return aggr_out

class Net(torch.nn.Module):
    """
    Message passing neural network.

    Currently we fixed number of message passes to 3. This will change and become a hyperparameter
    """

    def __init__(self, n_trains, n_nodes):
        super(Net, self).__init__()
        self.feature_size = n_trains+n_nodes
        self.nodesize = n_nodes
        self.msg_passing_1 = MsgPassLayer(self.feature_size, self.feature_size)
        self.msg_passing_2 = MsgPassLayer(self.feature_size, self.feature_size)
        self.readout_mlp = MLP(self.feature_size * n_nodes, n_nodes * n_trains, hidsize=64)

    def forward(self, data):
        # TODO: Introduce time as a concept in the model
        # Todo:Investigate on better propagation models for infromation propagation
        x, edge_index = data.x, data.edge_index
        x = self.msg_passing_1(x, edge_index)
        x = self.msg_passing_2(x, edge_index)

        # We ask an oracle to look at the hidden messages to deduce the locations of the trains
        new_shape = int(x.shape[0] / self.nodesize)
        y = x.view(new_shape, -1)
        y = self.readout_mlp(y)
        return y



class MLP(Module):
    """
    Multi Layer Perceptron for message passing

    This MLP takes the features of a node as an input and outputs a feature vector of the same size.

    We currently use Relu as non-linearity. Once we have updated the feature vector correctly we can work with softmax
    """

    def __init__(self, input_size, output_size, hidsize=16):
        super(MLP, self).__init__()

        self.fc1 = Linear(input_size, hidsize)
        self.fc2 = Linear(hidsize, output_size)

    def forward(self, x):
        """
        Forward pass of MLP
        Parameters
        ----------
        x: Feature vector containing all features of a node

        Returns
        -------
        Feature vector of the same dimensions as input vector
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x
