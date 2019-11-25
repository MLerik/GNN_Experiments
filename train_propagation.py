import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from NN.graph_convolution import Net
from graph_dataset import TrainScheduleDataset

device = torch.device('cpu')

schedule_data = TrainScheduleDataset("./tmp/train_schedules")
schedule_loader = schedule_data  # DataLoader(schedule_data, batch_size=128)
model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

schedule_data.generate_data_point()


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


for epoch in range(100):
    loss = train()
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
current_grid = schedule_data.generate_data_point(0)

for fun in range(4):
    print("=============================================")
    if fun == 0:
        nex_step = current_grid
    else:
        nex_step = Data(x=output, y=output, edge_index=edge_index.t().contiguous(), pos=positions)
    output = model(nex_step)
    print(output)
