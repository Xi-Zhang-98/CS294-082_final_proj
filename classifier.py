import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import argparse
from data import ODIRDataset
from data import ISBI2021Dataset
from data import STAREDataset
from torchvision import models
import os
import sklearn
from sklearn import metrics
import shutil
from efficientnet_pytorch import EfficientNet




# 参数设置
parser = argparse.ArgumentParser(description='PyTorch STARE Training')
parser.add_argument('--data_dir', default='./data/processed_STARE/', type=str)
parser.add_argument('--outf', default='./exp000/checkpoint/', type=str, help='path of the model checkpoints')
parser.add_argument('--code_dir', default='./exp000/code/', type=str, help='path of the code')
parser.add_argument('--result_dir', default='./exp000/result/', type=str, help='path of the result')
parser.add_argument('--model', default='vgg19', type=str, help='choice of the model')
parser.add_argument('--ngpu', default=0, type=int, help='gpu id(s)')
args = parser.parse_args()

# 定义是否使用GPU
torch.cuda.set_device(args.ngpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
EPOCH = 200   
pre_epoch = 0  
batch_size = 32    
LR = 0.0002        
data_dir = args.data_dir
img_size = 256
NUMCLASS = 3
beta1 = 0.5

trainset = STAREDataset(root_dir=data_dir, img_size=img_size, loader='training')
testset = STAREDataset(root_dir=data_dir, img_size=img_size, loader='testing')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=32,shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=32,shuffle=True)

# define the classification model
model_name = args.model
if model_name == 'ResNet':
    # 模型定义-ResNet
    net = models.resnet50(pretrained=True) 
    num_fits = net.fc.in_features
    # net.fc = nn.Linear(num_fits, NUMCLASS) # 替换最后一个全连接层
    net.fc = nn.Sequential(
                nn.Linear(in_features=num_fits,out_features=512,bias=True),
                nn.Linear(in_features=512,out_features=128,bias=True),
                nn.Linear(in_features=128,out_features=32, bias=True),
                nn.Linear(in_features=32,out_features=3, bias=True)
            )
elif model_name == 'Effinet':
    # 模型定义-Effinet-b5
    effinet_ckpt = torch.load("./code/efficientnet-b5-b6417697.pth")
    net = EfficientNet.from_name('efficientnet-b5')
    net.load_state_dict(effinet_ckpt)
    feature = net._fc.in_features
    # 替换最后的全连接层，以适用我们的应用
    net._fc = nn.Sequential(
    nn.Linear(in_features=feature,out_features=256,bias=True),
    nn.Linear(in_features=256,out_features=64,bias=True),
    nn.Linear(in_features=64,out_features=3, bias=True)
    )
else:
    net = models.vgg19(pretrained=True)
    net._classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, NUMCLASS),
        )
net = net.to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.Adam(net.parameters(), lr=LR, betas=(beta1, 0.999)) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

# save the code in result dir
code_dir = args.code_dir
if not os.path.exists(code_dir):
    os.makedirs(code_dir)
shutil.copy(__file__, code_dir)
shutil.copy('./code/data.py', code_dir)

model_dir = args.outf
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


result_dir = args.result_dir
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
acc_txt = os.path.join(result_dir,"acc.txt")
best_acc_txt = os.path.join(result_dir,"best_acc.txt")
log_txt = os.path.join(result_dir,"log.txt")

# 训练
if __name__ == "__main__":
    best_acc = 0  #2 初始化best test accuracy
    print("Start training, ResNet50")  # 定义遍历数据集的次数
    with open(acc_txt, "w") as f:
        with open(log_txt, "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Waiting for test")
                with torch.no_grad():
                    all_labels = []
                    predictions = []
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)

                        _, predicted = torch.max(outputs.data, 1)
                        all_labels.extend(labels.cpu().numpy())
                        predictions.extend(predicted.cpu().numpy())
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    # calcualte AUC and kappa
                    one_hot_all_labels = sklearn.preprocessing.label_binarize(all_labels, classes = [0, 1, 2])
                    one_hot_predictions = sklearn.preprocessing.label_binarize(predictions, classes = [0, 1, 2])
                    auc_score = metrics.roc_auc_score(one_hot_all_labels, one_hot_predictions)
                    kappa = sklearn.metrics.cohen_kappa_score(all_labels, predictions, labels=None, weights=None)
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    # write the results in acc.txt
                    f.write("EPOCH=%03d,Accuracy= %.3f%%，AUC=%.3f, kappa=%.3f" % (epoch + 1, acc, auc_score, kappa))
                    f.write('\n')
                    f.flush() 
                    # write the best results in best_acc.txt
                    if acc > best_acc:
                        print('Saving model......')
                        torch.save(net.state_dict(), '%s/net.pth' % model_dir)
                        f3 = open(best_acc_txt, "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Finished training, TotalEPOCH=%d" % EPOCH)
    
