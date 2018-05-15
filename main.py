import torch
import os

from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from data.dataset import MyDataset
from models import NetWork

# if cuda is available
use_cuda = torch.cuda.is_available()

# print model architecture
model_org = NetWork.Net()
# print(model_org)

# load the model
if use_cuda:
    model = model_org.cuda()
else:
    model = model_org

# data root
root = os.path.abspath('../test_projects/data_test/pre_process_data/')
print(root)

# 利用自己的MyDataset创建数据集
train_data = MyDataset(root, datatxt='/train.txt', transform=transforms.ToTensor())
test_data = MyDataset(root, datatxt='/test.txt', transform=transforms.ToTensor())

# 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=64)

# define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)

        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()

        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.data[0]
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data)), train_acc / (len(train_data))))

    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)

        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()

        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.data[0]
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.data[0]
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data)), eval_acc / (len(test_data))))
