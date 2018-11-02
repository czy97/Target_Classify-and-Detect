import torch.nn as nn
from torchvision import models


class VggBn_Net(nn.Module):
    def __init__(self,num_classes = 5):
        super(VggBn_Net, self).__init__()
        vgg_model = models.vgg16_bn(pretrained=True)

        self.features = vgg_model.features
        self.classifier = vgg_model.classifier
        self.classifier[6] = nn.Linear(4096, num_classes)


        self.regressor = nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace = True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=4, bias=True),
        )

        for index in (0,3):
            l1 = self.classifier[index]
            l2 = self.regressor[index]
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))


    def forward(self, x):

        x = self.features(x)
        x = x.view(-1,512*7*7)

        x_classifier = self.classifier(x)
        x_regressor = self.regressor(x)

        return (x_classifier,x_regressor)


class Alex_Net(nn.Module):
    def __init__(self,num_classes = 5):
        super(Alex_Net, self).__init__()
        alex_model = models.alexnet(pretrained=True)

        self.features = alex_model.features
        self.classifier = alex_model.classifier
        self.classifier[6] = nn.Linear(4096, num_classes)


        self.regressor = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=9216, out_features=4096, bias=True),
        nn.ReLU(inplace = True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=4096, out_features=4, bias=True),
        )

        for index in (1,4):
            l1 = self.classifier[index]
            l2 = self.regressor[index]
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))


    def forward(self, x):

        x = self.features(x)
        x = x.view(-1,9216)

        x_classifier = self.classifier(x)
        x_regressor = self.regressor(x)

        return (x_classifier,x_regressor)

