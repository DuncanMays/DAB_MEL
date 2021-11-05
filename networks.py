import torch
F = torch.nn.functional


class TwoNN(torch.nn.Module):

    def __init__(self):
        super(TwoNN, self).__init__()
        
        self.fc1 = torch.nn.Linear(784, 200)
        self.fc2 = torch.nn.Linear(200, 200)
        self.fc3 = torch.nn.Linear(200, 10)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x

class FFnet(torch.nn.Module):

    def __init__(self):
        super(FFnet, self).__init__()
        
        self.fc1 = torch.nn.Linear(784, 300)
        self.fc2 = torch.nn.Linear(300, 124)
        self.fc3 = torch.nn.Linear(124, 60)
        self.fc4 = torch.nn.Linear(60, 10)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)

        return x

class convNet(torch.nn.Module):

    def __init__(self):
        super(convNet, self).__init__()

        self.loss = torch.nn.CrossEntropyLoss()
        self.device = 'cpu'

        # 28*28*1
        self.conv1 = torch.nn.Conv2d(1, 6, (3, 3))
        # 26*26*1

        # 26*26*1
        self.conv2 = torch.nn.Conv2d(6, 16, (3, 3))
        # 24*24*1

        
        self.fc1 = torch.nn.Linear(400, 200)
        self.fc2 = torch.nn.Linear(200, 100)
        self.fc3 = torch.nn.Linear(100, 10)

    def forward(self, x):
        
        x = x.to(self.device)

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, 400)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x