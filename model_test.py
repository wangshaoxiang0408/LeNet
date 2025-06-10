import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet


def test_data_process():
    test_data = FashionMNIST(root='./data',  # 指定数据集下载或保存的路径，这里为当前目录下的 './data'
                             train=False,  # 加载测试集。如果设置为 False，则加载测试集
                             transform=transforms.Compose([  # 进行数据预处理和转换
                                 transforms.Resize(size=28),
                                 transforms.ToTensor()]),  # 将图像转换为 PyTorch 张量
                             download=True)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=0)

    return test_dataloader


def test_model_process(model, test_dataloader):
    # 设定训练所用到的设备，有GPU则使用GPU，否则使用CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 将模型放入到测试设备中
    model = model.to(device)

    # 初始化参数
    test_correct = 0.0
    test_num = 0

    # 只进行前向传播，不进行梯度计算，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 将数据放入到测试设备中
            test_data_x = test_data_x.to(device)
            # 将标签放入到测试设备中
            test_data_y = test_data_y.to(device)
            # 将模型设置为评估模式
            model.eval()
            # 前向传播过程，输入为测试数据集，输出每个样本的预测值
            output = model(test_data_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，则精确度test_correct加1
            test_correct += torch.sum(pre_lab == test_data_y.data)
            # 将所有的测试样本累加
            test_num += test_data_x.size(0)

    # 计算测试集的准确度
    test_acc = test_correct.double().item() / test_num
    print("测试的准确率为：", test_acc)


if __name__ == "__main__":
    # 加载模型
    model = LeNet()
    model.load_state_dict(torch.load("best_model.pth",weights_only=True))
    # 加载测试数据
    test_dataloader = test_data_process()
    # 加载模型测试的函数
    test_model_process(model, test_dataloader)

    # 设定测试所用到的设备，有GPU则使用GPU，否则使用CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
            label = b_y.item()
            print("预测值：", classes[result], "-------", "真实值：", classes[label])