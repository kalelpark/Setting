import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import typing as ty
from torch import Tensor
import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False


def save_model(model, args):
    torch.save(model.module.state_dict(), args.savepath + "/" + str(args.model) + "_" + str(args.epochs) +".pt")

def get_loss(args):
    if args.label:
        return nn.BCELoss()
    elif args.label:
        return nn.CrossEntropyLoss()
    else:
        pass