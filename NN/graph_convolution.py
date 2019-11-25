import torch
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class MsgPassLayer(MessagePassing):
    def __init__(self, in_dim, out_dim):
        # Message passing with max aggregation.
        super(MsgPassLayer, self).__init__(aggr='add')
        self.mlp = MLP(1, 1)
        self.update_mlp = MLP(2, 2)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Start propagating messages.
        return self.propagate(edge_index=edge_index, size=(x.size(0), x.size(0)), x=x)

    # How to do proper message passing ?
    def message(self, x_j, x_i, flow="source_to_target"):
        # Generate messages.
        return self.mlp(x_j)  # 0.5 *  x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        new_embedding = torch.cat([aggr_out, x], dim=1)
        new_embedding = self.update_mlp(new_embedding)
        # Step 5: Return new node embeddings.
        return new_embedding


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.msg_passing_1 = MsgPassLayer(4, 4)
        self.msg_passing_2 = MsgPassLayer(4, 4)
        self.msg_passing_3 = MsgPassLayer(4, 4)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.msg_passing_1(x, edge_index)
        x = F.relu(x)
        x = self.msg_passing_2(x, edge_index)
        x = F.relu(x)
        x = self.msg_passing_3(x, edge_index)

        return F.relu(x)


class MLP(Module):

    def __init__(self, input_size, output_size, hidsize=16):
        super(MLP, self).__init__()

        self.fc1 = Linear(input_size, hidsize)
        self.fc2 = Linear(hidsize, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x
