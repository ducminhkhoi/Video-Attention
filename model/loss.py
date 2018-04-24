import torch.nn.functional as F


def cross_entropy_loss(y_input, y_target):
    return F.cross_entropy(y_input, y_target)
