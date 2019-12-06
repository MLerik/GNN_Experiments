import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1],
                           [2, 3],
                           [3, 2],
                           [3, 4],
                           [4, 3],
                           [4, 5],
                           [5, 4],
                           [5, 6],
                           [6, 5],
                           [6, 7],
                           [7, 6],
                           [7, 8],
                           [8, 7],
                           [2, 9],
                           [9, 2],
                           [9, 10],
                           [10, 9],
                           [10, 11],
                           [11, 10],
                           [11, 6],
                           [6, 11]
                           ], dtype=torch.long)


class TrainScheduleDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None):
        super(TrainScheduleDataset, self).__init__(root, transform, pre_transform)

        # Loading data if already present
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.edge_index = edge_index
    @property
    def raw_file_names(self):
        return ['some_file_1']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        """
        Dummy method, not used as we don't download data
        Returns
        -------

        """
        pass

    def process(self):
        # Read data into huge `Data` list.
        print("Reading in")
        data_list = []
        self.edge_index = edge_index
        # Loop over all possible positions (5 Nodes in this example)
        for node in range(1000):
            tmp_data = self.generate_data_point(position=None)

            data_list.append(tmp_data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def generate_data_point(self, data_x=None, data_y=None, position=None, train=None):
        """
        Own implementation, not part of Framework!
        Simple training data generator for simple schedule. Generates tuples of input output data.
        Input is graph at time t and output is graph at time t+1

        Parameters
        ----------
        position: Generate specific data point. If not provided random pair of input output is generated

        Returns
        -------
        Data point with graph at t and t+1
        """
        self.nr_nodes = 12
        self.nr_trains = 2

        if position is None:
            current_node = np.random.randint(self.nr_nodes)
        else:
            current_node = position

        if train is None:
            current_train = np.random.randint(self.nr_trains)
        else:
            current_train = train
        current_train = 0
        # Start by only moving around at the bottom rail thus clipping between 0 and 8
        next_node = np.clip(current_node + int(1 - 2 * current_train), 0, 8)

        # Load data from previous graph
        if data_x is None:
            input_data = np.zeros(shape=(self.nr_nodes, self.nr_trains + self.nr_nodes), dtype=float)
            input_data[current_node][current_train] = 1
            for i in range(self.nr_nodes):
                input_data[i, i + self.nr_trains] = 1
            input_tensor = torch.tensor(input_data, dtype=torch.float)
        else:
            input_tensor = torch.tensor(data_x, dtype=torch.float)

        if data_y is None:
            output_data = np.zeros(shape=(self.nr_nodes * self.nr_trains,), dtype=float)
            output_data[next_node + (self.nr_nodes*current_train)] = 1
            output_tensor = torch.tensor([output_data], dtype=torch.float)

        else:
            output_tensor = torch.tensor([data_y], dtype=torch.float)

        # Create graph data point
        data = Data(x=input_tensor, y=output_tensor, edge_index=self.edge_index.t().contiguous())

        return data
