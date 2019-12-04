import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader

from NN.graph_convolution import Net
from graph_dataset import TrainScheduleDataset

device = torch.device('cpu')

schedule_data = TrainScheduleDataset('tmp/')
schedule_loader = DataLoader(schedule_data, batch_size=128, shuffle=True)
n_nodes = 12
n_trains = 2
#TODO better implementation of magic number ;)
model = Net(n_trains=n_trains, n_nodes=n_nodes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-6)

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


for epoch in range(250):
    loss = train()
    if epoch % 100 == 0:
        print("Running epoch {} with a loss of {}".format(epoch, loss))

# Todo get this out of the data
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

# Move train to the right
for t in range(8):
    print("=============================================")
    print("Time step number {}, agent 1 should be at position {}".format(t,t+1))
    print("=============================================")
    if t == 0:
        nex_step = current_graph
    else:
        nex_step = Data(x=output, y=output, edge_index=edge_index.t().contiguous())
    output = model(nex_step)
    print(output[:,0])

current_graph = schedule_data.generate_data_point(1, 8)

# Move other train to the left
for t in range(8):
    print("=============================================")
    print("Time step number {}, agent 2 should be at position {}".format(t,t+1))
    print("=============================================")
    if t == 0:
        nex_step = current_graph
    else:
        nex_step = Data(x=output, y=output, edge_index=edge_index.t().contiguous())
    output = model(nex_step)
    print(output[:,0])
