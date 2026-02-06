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

    def forward_features(self, x, layer_idx=None, patch_tokens=False, layer_indices=None):
        with torch.no_grad():
            if layer_indices and len(layer_indices) > 0:
                captured = {}

                def _hook(index):
                    def _fn(_module, _input, output):
                        captured[index] = output
                    return _fn

                handles = []
                for li in layer_indices:
                    if li < 0 or li >= len(self.backbone.blocks):
                        raise ValueError(
                            f"dino_layer_idx {li} out of range (0..{len(self.backbone.blocks) - 1})."
                        )
                    handles.append(self.backbone.blocks[li].register_forward_hook(_hook(li)))
                _ = self.backbone(x)
                for h in handles:
                    h.remove()
                feats = []
                for li in layer_indices:
                    feat = captured.get(li)
                    if feat is None:
                        raise RuntimeError("Failed to capture intermediate features for dino_v2.")
                    if feat.dim() == 3:
                        feat = self.backbone.norm(feat)
                        feat = feat[:, 1:, :] if patch_tokens else feat[:, 0]
                    feats.append(feat)
                return torch.cat(feats, dim=-1)
            if layer_idx is None or layer_idx < 0:
                if patch_tokens:
                    tokens = self.backbone.forward_features(x)
                    tokens = self.backbone.norm(tokens)
                    return tokens[:, 1:, :]
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
                if patch_tokens:
                    return x[:, 1:, :]
                return x[:, 0]
            return x

    def forward(self, x, layer_idx=None, patch_tokens=False, layer_indices=None):
        return self.forward_features(
            x,
            layer_idx=layer_idx,
            patch_tokens=patch_tokens,
            layer_indices=layer_indices,
        )
