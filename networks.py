import torch
from torch import nn
import torch.nn.functional as F


def weights_init(mod):

    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)


class get_generator(nn.Module): # 提取特征工具
    def __init__(self, submodule, extracted_layers):
        super(get_generator, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
        self.pooling = F.adaptive_avg_pool2d
        self.fc = nn.Sequential(
            nn.Linear(in_features=2048,out_features=512,bias=True),
            nn.Linear(in_features=512,out_features=256,bias=True),
            nn.Linear(in_features=256,out_features=64,bias=True),
        )

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": 
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        # 进行自适应全局平均池化
        outputs = self.pooling(outputs[0], (1, 1))
        outputs = torch.squeeze(outputs, 3)
        outputs = torch.squeeze(outputs, 2)

        # 将输出特征降维度到64
        outputs = self.fc(outputs)
        return outputs

class get_generator_conv(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(get_generator_conv, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.fc = nn.Linear(in_features=256, out_features=64)
    
    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": 
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        # 使用卷积层和全连接层将维度降低到64
        outputs = self.conv1(outputs[0])
        outputs = self.conv2(outputs)
        outputs = outputs.view(outputs.size(0),-1)
        outputs = self.fc(outputs)


        return outputs


class get_disc_latent(nn.Module):
    """
    DISCRIMINATOR latent NETWORK
    """

    def __init__(self):
        super(get_disc_latent, self).__init__()
        # self.dense_1 = nn.Linear(2048, 512)
        # self.batch_norm_1 = nn.BatchNorm1d(512)

        # self.dense_2 = nn.Linear(512, 128)
        # self.batch_norm_2 = nn.BatchNorm1d(128)

        # self.dense_3 = nn.Linear(128, 64)
        # self.batch_norm_3 = nn.BatchNorm1d(64)

        # self.dense_4 = nn.Linear(64, 32)
        # self.batch_norm_4 = nn.BatchNorm1d(32)

        # self.dense_5 = nn.Linear(32, 16)
        # self.batch_norm_5 = nn.BatchNorm1d(16)

        # self.dense_6 = nn.Linear(16, 1)

        # self.dense_1 = nn.Linear(256, 64)
        # self.batch_norm_1 = nn.BatchNorm1d(64)

        self.dense_2 = nn.Linear(64, 16)
        self.batch_norm_2 = nn.BatchNorm1d(16)

        self.dense_3 = nn.Linear(16, 4)
        self.batch_norm_3 = nn.BatchNorm1d(4)

        self.dense_4 = nn.Linear(4, 1)


    def forward(self, input):
        output = input.view(input.size(0),-1)

        # output = self.batch_norm_1(self.dense_1(output))
        # output = F.relu(output)
        
        # output = self.batch_norm_2(self.dense_2(output))
        # output = F.relu(output)

        # output = self.batch_norm_3(self.dense_3(output))
        # output = F.relu(output)

        # output = self.batch_norm_4(self.dense_4(output))
        # output = F.relu(output)

        # output = self.batch_norm_5(self.dense_5(output))
        # output = F.relu(output)

        # output = self.dense_6(output)
        # output = torch.sigmoid(output)

        # output = self.batch_norm_1(self.dense_1(output))
        # output = F.relu(output)
        
        output = self.batch_norm_2(self.dense_2(output))
        output = F.relu(output)

        output = self.batch_norm_3(self.dense_3(output))
        output = F.relu(output)

        output = self.dense_4(output)
        
        output = torch.sigmoid(output)

        return output

class get_fc(nn.Module):
    def __init__(self):
        super(get_fc, self).__init__()
        # self.dense_1 = nn.Linear(2048, 512)

        # self.dense_2 = nn.Linear(512, 128)

        # self.dense_3 = nn.Linear(128, 32)

        # self.dense_4 = nn.Linear(32, 3)

        # self.dense_1 = nn.Linear(256, 64)

        self.dense_2 = nn.Linear(64, 16)

        self.dense_3 = nn.Linear(16, 3)

    def forward(self, input):
        output = input.view(input.size(0), -1)

        # output = self.dense_1(output)

        output = self.dense_2(output)
        output0 = output

        output = self.dense_3(output)

        # output = self.dense_4(output)

        return output0, output

class classifier(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(classifier, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
        self.pooling = F.adaptive_avg_pool2d
        self.fc = nn.Sequential(
            nn.Linear(in_features=2048,out_features=512,bias=True),
            nn.Linear(in_features=512,out_features=256,bias=True),
            nn.Linear(in_features=256,out_features=64,bias=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64,out_features=16,bias=True),
            nn.Linear(in_features=16,out_features=2,bias=True)
        )

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
        # for name, module in self._modules.items():
            if name is "fc": 
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        # 进行自适应全局平均池化
        outputs = self.pooling(outputs[0], (1, 1))
        outputs = torch.squeeze(outputs, 3)
        outputs = torch.squeeze(outputs, 2)

        # 将输出特征降维度到64
        outputs = self.fc(outputs)
        outputs_cls = self.fc1(outputs)
        return outputs_cls, outputs

