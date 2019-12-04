import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader

from NN.graph_convolution import Net
from graph_dataset import TrainScheduleDataset

device = torch.device('cpu')


schedule_data = TrainScheduleDataset('tmp/')
schedule_loader = DataLoader(schedule_data, batch_size=6, shuffle=True)
n_nodes = 12
n_trains = 2
model = Net(n_trains=n_trains, n_nodes=n_nodes).to(device)

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


for epoch in range(200):
    loss = train()
    if epoch % 100 == 0:
        print("Running epoch {} with a loss of {}".format(epoch, loss))


current_graph = schedule_data.generate_data_point(train=1, position=7)

model.eval()
for t in range(10):
    print("====================================================")
    print("Time step number {}, agent should be at position {}".format(t, t + 1))
    print("====================================================")
    if t == 0:
        nex_step = current_graph
    else:
        nex_step =schedule_data.generate_data_point(data_x=input_data, data_y=input_data)
    output = model(nex_step)
    input_data = np.zeros(shape=(n_nodes, n_trains + n_nodes), dtype=float)
    for i in range(n_nodes):
        input_data[i, i + n_trains] = 1
    output_np = output.detach().numpy()
    # All still very hacky
    input_data[:, 0] = output_np[0][:n_nodes]
    input_data[:, 1] = output_np[0][n_nodes:]

    print(output.detach().numpy()[0][:n_nodes])

    print(output[:,0])

