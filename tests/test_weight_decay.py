import torch
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc2 = torch.nn.Linear(120, 84)

        self.fc3 = torch.nn.Linear(84, 10)
        # self.relu = CallWrapper(F.relu, node_tracing_name='relu')
        self.add = torch.add

    def forward(self, x):
        x = self.fc2(x)
        x1 = F.relu(x)
        # x1.node_tracing_name = self.fc2.node_tracing_name
        x = self.add(x, x1)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.0, 0.99))

    a = net(torch.zeros(5, 120))
    loss = (1. - a**2).mean()
    loss.backward()

    # a = net(torch.zeros(5, 120)) If un commented it runs
    loss = (1. - a).mean()
    loss.backward()

