import torch
from torch import nn
from torchsummary import summary

class LeNet(nn.Module):
    # 进行初始化
    def __init__(self):
        super(LeNet,self).__init__()
        #定义第一次卷积层c1
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6,kernel_size=5,stride=1,padding=2)
        #激活函数
        self.sig = nn.Sigmoid()
        #池化层
        self.s2 = nn.AvgPool2d(kernel_size=2,stride=2)
        #卷积
        self.c3 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        #池化
        self.s4 = nn.AvgPool2d(kernel_size=2,stride=2)
        #平展操作，定义平展层
        self.flatten = nn.Flatten()
        #全连接层
        self.f5 = nn.Linear(400,120)
        self.f6 = nn.Linear(120,84)
        self.f7 = nn.Linear(84,10)

        #定义前向传播的函数forward
    def forward(self,x):
        x = self.sig(self.c1(x))
        x = self.s2(x)
        x = self.sig(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f7(self.f6(self.f5(x)))
        return x

#主函数
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    print(summary(model,(1,28,28)))