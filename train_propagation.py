import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader

from NN.graph_convolution import Net
from graph_dataset import TrainScheduleDataset

device = torch.device('cpu')

schedule_data = TrainScheduleDataset("./tmp/train_schedules")
schedule_loader = DataLoader(schedule_data, batch_size=6, shuffle=True)
model = Net(features=6, n_nodes=5).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

a = np.arange(80)
a = torch.tensor(a)


def train():
    model.train()
    for data in schedule_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = F.mse_loss(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss


for epoch in range(150):
    loss = train()
    if epoch % 100 == 0:
        print("Running epoch {} with a loss of {}".format(epoch, loss))

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
current_graph = schedule_data.generate_data_point(0)

model.eval()
for t in range(4):
    print("=============================================")
    print("Time step number {}, agent should be at position {}".format(t, t + 1))
    print("=============================================")
    if t == 0:
        nex_step = current_graph
    else:
        nex_step = Data(x=input_data, y=input_data, edge_index=edge_index.t().contiguous(), pos=positions)
    output = model(nex_step)
    input_data = np.zeros(shape=(5, 1 + 5), dtype=float)
    input_data[:, 0] = output.detach().numpy()
    for i in range(5):
        input_data[i, i + 1] = 1
    input_data = torch.tensor(input_data, dtype=torch.float)
    print(input_data)
