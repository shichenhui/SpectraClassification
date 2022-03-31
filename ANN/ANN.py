import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import MyUtils
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self):
        feature_num = 3131
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=16), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=16),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=16),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )

        self.fc = nn.Sequential(
            nn.Linear(5920, 1024),
            nn.ReLU(),

            nn.Linear(1024, 4),
            #nn.Softmax()
        )

    def forward(self, X):
        feature = self.conv(X)
        output = self.fc(feature.view(X.shape[0], -1))
        return output

class NetLinear(nn.Module):
    def __init__(self):
        super(NetLinear, self).__init__()
        self.Linear =nn.Sequential(
            nn.Linear(3121, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200,4),
            #nn.Softmax(),
        )
    def forward(self, X):
        output = self.Linear(X.view(X.shape[0], -1))
        return output

class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)


X_train, y_train, X_test, y_test = MyUtils.load_data(r'C:\Users\panda\Desktop\光谱数据样例\star_AFGK_2kx4.csv', class_num=4,
                                                     norm=True, shuffle=True, split=0.8, one_hot=False, dtype=np.float32)
X_train = torch.from_numpy(X_train[:,np.newaxis,:])
y_train = torch.from_numpy(y_train).long()

model = NetLinear()
epoch_num = 200
batch_size = 100
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                             weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=40, gamma=0.1)
train_data = DataLoader(GetLoader(X_train, y_train), batch_size=batch_size, shuffle=False)
print(len(train_data))
for epoch in range(epoch_num):
    sum_loss = 0
    sum_accuracy = 0
    for X, y in train_data:
        out = model(X)
        loss = criterion(out, y)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        sum_loss = sum_loss + loss
        #print(loss)
        optimizer.step()
        # =========calculate current accuracy============
        y_pred = torch.max(out, dim=1)[1]   # get predict class
        sum_accuracy += torch.true_divide((y_pred == y).sum(), len(y))
    scheduler.step()
    print('epoch: ',epoch, sum_loss.item()/len(train_data), 'accuracy: ', torch.true_divide(sum_accuracy, len(train_data)))

    #scheduler.step()

# print(X_train.shape)
# X_train = torch.from_numpy(X_train)
# model = Net()
# print(model)
# print(model(X_train[:1][:,np.newaxis,:]))
