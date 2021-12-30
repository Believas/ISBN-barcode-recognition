import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
from mydataset import MyDataset
from torch.optim import sgd, adam, adagrad, rmsprop, adadelta, adamax, adamw, sparse_adam, asgd
from Net import Net


if __name__ == '__main__':
    batch_size = 128
    # 加载本地数据集
    data_path = r"./data/Mydataset_01"
    # save_params = "./save_model/para_version_2.pth"
    save_net = "./save_model/net_version_2.pth"

    train_data = MyDataset(data_path, True)
    test_data = MyDataset(data_path, False)

    train_loader = data.DataLoader(train_data, batch_size, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    net = Net().to(device)
    net = torch.load(save_net).to(device)
    # net.load_state_dict(torch.load(save_params))


    loss_function = nn.MSELoss()

    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.5, dampening=0,
    #                             weight_decay=0,  nesterov=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                                 weight_decay=0, amsgrad=False)  # betas:0.9越大越平滑， 默认效果最好
    # weight_decay:表示L2正则化系数

    # optimizer = adagrad.Adagrad(net.parameters())
    # optimizer = adadelta.Adadelta(net.parameters())
    # optimizer = rmsprop.RMSprop(net.parameters())
    # optimizer = sgd.SGD(net.parameters(), 1e-3)
    # optimizer = adam.Adam(net.parameters())

    a = []
    b = []
    plt.ion()
    net.train()
    for epoch in range(10):
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            output = net(x)
    
            # print(output)
            # print(output[0])  # 一张图片经过神经网络输出的十个值
            # print(output.shape)  # torch.Size([100, 10])
            # print(y)
            # 在1轴里面填1， 同时将标签形状变为（N, 1）
            y = torch.zeros(y.cpu().size(0), 12).scatter_(1, y.cpu().reshape(-1, 1), 1).to(device)  # 根据数字确定索引位置
            # print(y)
            # print(y.size(0))
    
            # 加正则化
            # L1 = 0
            # L2 = 0
            # for params in net.parameters():
            #     L1 += torch.sum(torch.abs(params))
            #     L2 += torch.sum(torch.pow(params, 2))
            # loss = loss_function(output, y)
            # loss1 = loss + 0.001*L1
            # loss2 = loss + 0.001*L2
            # loss = 0.2*loss1 + 0.8*loss2
    
            loss = loss_function(output, y)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                a.append(i + (epoch*(len(train_data) / batch_size)))
                b.append(loss.item())
                # plt.clf()
                # plt.plot(a, b)
                # plt.pause(1)
                print("Epoch:{}, batch:{}/600, loss:{:.3f}".format(epoch, int(i), loss.item()))
    
        # print(a)
        torch.save(net.state_dict(), "./save_model/para_version_2.pth")
        torch.save(net, "./save_model/net_version_2.pth")

    net.eval()
    eval_loss = 0
    eval_acc = 0
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        out = net(x)

        y = torch.zeros(y.cpu().size(0), 12).scatter_(1, y.cpu().reshape(-1, 1), 1).to(device)  # 标签y不能在GPU上运算，需要转CPU
        loss = loss_function(out, y)
        # print("Test_Loss:{:.3f}".format(loss.item()))

        # print(y.size(0))
        # print(loss.item())
        # print("====")

        eval_loss += loss.item()*y.size(0)  # 一张图片的损失值乘以批次就是这一批的损失，循环一轮，就是总损失。
        arg_max = torch.argmax(out, 1)

        # y = torch.argmax(y, 1)
        y = y.argmax(1)  # 根据索引将数字取出来
        # print(arg_max)
        # print(y)
        # exit()

        eval_acc += (arg_max==y).sum().item()

    mean_loss = eval_loss / len(test_data)  # 算完所有轮次的总损失除以总的测试数据长度
    mean_acc = eval_acc / len(test_data)

    print(y)
    print(torch.argmax(out, 1))
    print("loss:{:.3f}, Acc:{:.3f}".format(mean_loss, mean_acc))

