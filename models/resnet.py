import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50


class ResNetModel(nn.Module):
    def __init__(self, pretrained=True, num_classes=2, backbone_name="resnet18", freeze_backbone=False):
        super(ResNetModel, self).__init__()
        if backbone_name == "resnet50":
            self.backbone = resnet50(pretrained=pretrained)
            last_layer = 2048
        else:
            # self.resnet18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=pretrained)
            self.backbone = resnet18(pretrained=pretrained)
            last_layer = 512

        # create MPL head as seen in the code in: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        # TODO: check if this is really the right architecture
        sequential_layers = []
        head_layers = [512, 512, 128]
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1d(num_neurons))
            sequential_layers.append(nn.ReLU(inplace=True))
            last_layer = num_neurons

        # the last layer without activation
        head = nn.Sequential(
            *sequential_layers
        )
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            head,
            nn.Linear(last_layer, num_classes)
        )
        # self.head = head
        # self.out = nn.Linear(last_layer, num_classes)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        embeds = self.backbone(x)
        # tmp = self.head(embeds)
        # logits = self.out(tmp)
        logits = self.head(embeds)
        return logits, embeds

    def forward_features(self, x):
        embeds = self.backbone(x)
        return embeds

    def forward_features_layer(self, x, layer_idx):
        if layer_idx not in (1, 2, 3, 4):
            raise ValueError("resnet_layer_idx must be 1..4")
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        if layer_idx >= 1:
            x = self.backbone.layer1(x)
        if layer_idx >= 2:
            x = self.backbone.layer2(x)
        if layer_idx >= 3:
            x = self.backbone.layer3(x)
        if layer_idx >= 4:
            x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def freeze_resnet(self):
        # freez full resnet18
        for param in self.backbone.parameters():
            param.requires_grad = False
        # unfreeze head:
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        # unfreeze all:
        for param in self.parameters():
            param.requires_grad = True
