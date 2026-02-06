import torch
import torch.nn as nn
import timm


class DinoV2Timm(nn.Module):
    def __init__(self, 
                 model_name="vit_base_patch14_dinov2", 
                 pretrained=True,
                 img_size=224,
                 freeze_backbone=True): 
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0, # no head
            img_size=img_size,
        )
        self.embed_dim = self.backbone.num_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def forward_features(self, x, layer_idx=None):
        with torch.no_grad():
            if layer_idx is None or layer_idx < 0:
                return self.backbone(x)
            if layer_idx >= len(self.backbone.blocks):
                raise ValueError(
                    f"dino_layer_idx {layer_idx} out of range (0..{len(self.backbone.blocks) - 1})."
                )
            captured = {}

            def _hook(_module, _input, output):
                captured["x"] = output

            handle = self.backbone.blocks[layer_idx].register_forward_hook(_hook)
            _ = self.backbone(x)
            handle.remove()

            x = captured.get("x")
            if x is None:
                raise RuntimeError("Failed to capture intermediate features for dino_v2.")
            if x.dim() == 3:
                x = self.backbone.norm(x)
                x = x[:, 0]
            return x

    def forward(self, x, layer_idx=None):
        return self.forward_features(x, layer_idx=layer_idx)
