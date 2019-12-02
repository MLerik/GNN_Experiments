import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader

from NN.graph_convolution import Net
from graph_dataset import TrainScheduleDataset

device = torch.device('cpu')


schedule_data = TrainScheduleDataset('tmp/')
schedule_loader = DataLoader(schedule_data, batch_size=6, shuffle=True)
model = Net(n_trains=2, n_nodes=12).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)



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


current_graph = schedule_data.generate_data_point(train=0, position=0)

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
    print(output)
    input_data = np.zeros(shape=(5, 1 + 5), dtype=float)
    input_data[:, 0] = output.detach().numpy()
    for i in range(5):
        input_data[i, i + 1] = 1

    print(output.detach().numpy())
