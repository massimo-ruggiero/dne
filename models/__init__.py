from .resnet import ResNetModel
from .vit import ViT
from .csflow_net import NetCSFlow
from .revdis_net import NetRevDis
from .dino_v2 import DinoV2Timm

from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils.optimizer import get_optimizer



def get_net_optimizer_scheduler(args):
    if args.model.name == 'resnet':
        net = ResNetModel(
            pretrained=args.model.pretrained,
            num_classes=args.train.num_classes,
            backbone_name="resnet18",
            freeze_backbone=getattr(args, "resnet_freeze_backbone", False),
        )
        optimizer = get_optimizer(args, net)
        scheduler = CosineAnnealingWarmRestarts(optimizer, args.train.num_epochs)
    elif args.model.name == 'resnet50':
        net = ResNetModel(
            pretrained=args.model.pretrained,
            num_classes=args.train.num_classes,
            backbone_name="resnet50",
            freeze_backbone=getattr(args, "resnet_freeze_backbone", True),
        )
        optimizer = get_optimizer(args, net)
        scheduler = CosineAnnealingWarmRestarts(optimizer, args.train.num_epochs)
    elif args.model.name == 'vit':
        net = ViT(num_classes=args.train.num_classes)
        if args.model.pretrained:
            checkpoint_path = './checkpoints/ViT-B_16.npz'
            net.load_pretrained(checkpoint_path)
        optimizer = get_optimizer(args, net)
        scheduler = CosineAnnealingWarmRestarts(optimizer, args.train.num_epochs)
    elif args.model.name == 'dino_v2':
        net = DinoV2Timm(
            model_name=args.model.dino_name,
            pretrained=args.model.pretrained,
            img_size=args.dataset.image_size,
            freeze_backbone=True,
        )
        if args.model.method == 'dne' and args.eval.eval_classifier == 'density':
            optimizer = None
            scheduler = None
        else:
            optimizer = get_optimizer(args, net)
            scheduler = CosineAnnealingWarmRestarts(optimizer, args.train.num_epochs)
    elif args.model.name == 'net_csflow':
        net = NetCSFlow(args)
        optim_modules = nn.ModuleList()
        if args.model.pretrained:
            names_to_update = ["density_estimator"]
            for name, param in net.named_parameters():
                param.requires_grad_(False)
            for name_to_update in names_to_update:
                optim_modules.append(getattr(net, name_to_update))
                for name, param in net.named_parameters():
                    if name_to_update in name:
                        param.requires_grad_(True)
        optimizer = get_optimizer(args, optim_modules)
        scheduler = None
    elif args.model.name == 'net_revdis':
        net = NetRevDis(args)
        optim_modules = nn.ModuleList()
        if args.model.pretrained:
            names_to_update = ["decoder", "bn"]
            for name, param in net.named_parameters():
                param.requires_grad_(False)
            for name_to_update in names_to_update:
                optim_modules.append(getattr(net, name_to_update))
                for name, param in net.named_parameters():
                    if name_to_update in name:
                        param.requires_grad_(True)
        optimizer = get_optimizer(args, optim_modules)
        scheduler = None
    return net, optimizer, scheduler
