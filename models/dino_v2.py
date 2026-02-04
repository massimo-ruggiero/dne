import torch
import torch.nn as nn
import timm


class DinoV2Timm(nn.Module):
    def __init__(self, 
                 model_name="vit_base_patch14_dinov2", 
                 pretrained=True,
                 freeze_backbone=True): 
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0 # no head
        )
        self.embed_dim = self.backbone.num_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def forward_features(self, x):
        with torch.no_grad():
            features = self.backbone(x)
            
        return features

    def forward(self, x):
        return self.forward_features(x)
