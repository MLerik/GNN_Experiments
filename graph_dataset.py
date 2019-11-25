import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


class TrainScheduleDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None):
        super(TrainScheduleDataset, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        print("Reading in")
        data_list = []
        for i in range(500):
            tmp_data = self.generate_data_point()
            data_list.append(tmp_data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def generate_data_point(self, position=None):
        """
        Simple training data generator for simple schedule. Generates tuples of input output data.
        Input is graph at time t and output is graph at time t+1

        Parameters
        ----------
        position: Generate specific data point. If not provided random pair of input output is generated

        Returns
        -------
        Data point with graph at t and t+1
        """
        if position is None:
            current_position = np.random.randint(4)
        else:
            current_position = position

        if current_position == 2:
            next_position = np.random.choice([3, 4])
        elif current_position == 3:
            next_position = 3
        elif current_position == 4:
            next_position = 4
        else:
            next_position = int(np.clip(current_position + 1, 0, 3))
        input_data = np.zeros(shape=(5, 1 + 5), dtype=float)
        output_data = np.zeros(shape=(5, 1 + 5), dtype=float)
        input_data[current_position][0] = 1
        output_data[next_position][0] = 1
        for i in range(5):
            input_data[i, i + 1] = 1
            output_data[i, i + 1] = 1
        input_tensor = torch.tensor(input_data, dtype=torch.float)
        output_tensor = torch.tensor(output_data, dtype=torch.float)
        edge_index = torch.tensor([[0, 1],
                                   [1, 0],
                                   [1, 2],
                                   [2, 1],
                                   [2, 3],
                                   [3, 2],
                                   [2, 4],
                                   [4, 2]
                                   ], dtype=torch.long)
        positions = torch.tensor([[0, 0],
                                  [0, 1],
                                  [0, 2],
                                  [0, 3],
                                  [1, 3]
                                  ], dtype=torch.long)
        data = Data(x=input_tensor, y=output_tensor, edge_index=edge_index.t().contiguous(), pos=positions)
        return data


    def generate_data_point(self, position=None):
        """
        Simple training data generator for simple schedule. Generates tuples of input output data.
        Input is graph at time t and output is graph at time t+1

        Parameters
        ----------
        position: Generate specific data point. If not provided random pair of input output is generated

        Returns
        -------
        Data point with graph at t and t+1
        """
        if position is None:
            current_position = np.random.randint(4)
        else:
            current_position = position

        if current_position == 2:
            next_position = np.random.choice([3, 4])
        elif current_position == 3:
            next_position = 3
        elif current_position == 4:
            next_position = 4
        else:
            next_position = int(np.clip(current_position + 1, 0, 3))
        input_data = np.zeros(shape=(5, 1 + 5), dtype=float)
        output_data = np.zeros(shape=(5, 1 + 5), dtype=float)
        input_data[current_position][0] = 1
        output_data[next_position][0] = 1
        for i in range(5):
            input_data[i, i + 1] = 1
            output_data[i, i + 1] = 1
        input_tensor = torch.tensor(input_data, dtype=torch.float)
        output_tensor = torch.tensor(output_data, dtype=torch.float)
        edge_index = torch.tensor([[0, 1],
                                   [1, 0],
                                   [1, 2],
                                   [2, 1],
                                   [2, 3],
                                   [3, 2],
                                   [2, 4],
                                   [4, 2]
                                   ], dtype=torch.long)
        positions = torch.tensor([[0, 0],
                                  [0, 1],
                                  [0, 2],
                                  [0, 3],
                                  [1, 3]
                                  ], dtype=torch.long)
        data = Data(x=input_tensor, y=output_tensor, edge_index=edge_index.t().contiguous(), pos=positions)
        return data