import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader

from NN.graph_convolution import Net
from graph_dataset import MultiTrainScheduleDataset

device = torch.device('cpu')

schedule_data = MultiTrainScheduleDataset("./tmp/train_schedules_multi")
schedule_loader = DataLoader(schedule_data, batch_size=128, shuffle=True)
model = Net(12+2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

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


for epoch in range(5000):
    loss = train()
    print("Running epoch {} with a loss of {}".format(epoch, loss))

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1],
                           [2, 3],
                           [3, 2],
                           [3, 4],
                           [4, 3],
                           [4, 5],
                           [5, 6],
                           [6, 5],
                           [6, 7],
                           [7, 6],
                           [7, 8],
                           [2, 9],
                           [9, 2],
                           [9, 10],
                           [10, 9],
                           [10, 11],
                           [11, 10],
                           [11, 6],
                           [6, 11]
                           ], dtype=torch.long)
current_graph = schedule_data.generate_data_point(0, 0)

for t in range(8):
    print("=============================================")
    print("Time step number {}, agent should be at position {}".format(t,t+1))
    print("=============================================")
    if t == 0:
        nex_step = current_graph
    else:
        nex_step = Data(x=output, y=output, edge_index=edge_index.t().contiguous())
    output = model(nex_step)
    print(output[:,0])

current_graph = schedule_data.generate_data_point(1, 8)

for t in range(4):
    print("=============================================")
    print("Time step number {}, agent should be at position {}".format(t,t+1))
    print("=============================================")
    if t == 0:
        nex_step = current_graph
    else:
        nex_step = Data(x=output, y=output, edge_index=edge_index.t().contiguous())
    output = model(nex_step)
    print(output[:,0])
