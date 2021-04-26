import torch
import torchvision
import torch.nn as nn
import torch.optim as O
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc2(x)
        return x

custom_net = Net()
# custom_net.fc1.weight.data.fill_(2)
# custom_net.fc1.bias.data.fill_(0)
criterion = nn.MSELoss()
optimizer = O.SGD(custom_net.parameters(), lr=0.0001, momentum=0.9)

input_tensors = []
target_tensors = []
for data_index in range(200):
    output_tensor = torch.tensor([10.0]).unsqueeze(1)
    random_tensor = torch.rand((1,1))
    input_tensor = torch.cat((random_tensor, random_tensor*output_tensor), 1)
    input_tensors.append(input_tensor)
    target_tensors.append(output_tensor)

epoch_loss_list = []
iter_loss_list = []
epoch_list = []
iter_list = []
for epoch in range(100):
    epoch_list.append(epoch)
    avg_loss = 0
    for idx, (inputs, target) in enumerate(list(zip(input_tensors, target_tensors))):
        optimizer.zero_grad()
        pred = custom_net(inputs)
        loss = criterion(pred, target)
        avg_loss += loss.item()
        iter_list.append((epoch*1000) + idx)
        iter_loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
    epoch_loss_list.append(avg_loss/len(input_tensors))

print(min(epoch_loss_list))
print(target_tensors[0], custom_net(input_tensors[0]))
print(custom_net.fc1.weight)
print(custom_net.fc1.bias.data)
# plt.scatter(epoch_list, epoch_loss_list)
plt.scatter(iter_list, iter_loss_list)
plt.show()