from efficientnet_pytorch import EfficientNet
from torchvision import models
from .custommodel import load_custom

def load_model(args):
    if args.model == "efficientnet-b0":
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes = 5)
        return model
    elif args.model == "efficientnet-b1":
        model = EfficientNet.from_pretrained('efficientnet-b1', num_classes = 5)
        return model
    elif args.model == "efficientnet-b2":
        model = EfficientNet.from_pretrained('efficientnet-b2', num_classes = 5)
        return model
    elif args.model == "efficientnet-b3":
        model = EfficientNet.from_pretrained('efficientnet-b3', num_classes = 5)
        return model
    elif args.model == "efficientnet-b4":
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes = 5)
        return model
    elif args.model == "efficientnet-b5":
        model = EfficientNet.from_pretrained('efficientnet-b5', num_classes = 5)
        return model
    elif args.model == "efficientnet-b6": # 512
        model = EfficientNet.from_pretrained('efficientnet-b6', num_classes = 5)
        return model
    elif args.model == "efficientnet-b7":   # 600
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes = 5)
        return model
    elif args.model == "hybridswintransformer":
        model = load_custom(args)
        return model
    else:
        return "Load Model ERROR"