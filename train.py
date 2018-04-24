import argparse
import logging
import torch.optim as optim
from model.model import MnistModel
from model.loss import cross_entropy_loss
from model.metric import my_metric, my_metric2, acc_metric
from data_loader import MnistDataLoader
import torch
import torchvision
import torchvision.transforms as transforms

from logger import Logger
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR

logging.basicConfig(level=logging.INFO, format='')

parser = argparse.ArgumentParser(description='PyTorch Template')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('-lr', default=1e-4, type=int,
                    help='learning rate for training')
parser.add_argument('-num-classes', default=200, type=int,
                    help='number of classes')
parser.add_argument('-e', '--epochs', default=300, type=int,
                    help='number of total epochs (default: 32)')
parser.add_argument('--dataset', default='moment', type=str,
                    help='dataset to use: mnist, cifar10 or ucf101, ucfsport, moment (default: none)')
parser.add_argument('--model', default='vgg', type=str,
                    help='model to use: fpn, vgg, resnet')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--verbosity', default=1, type=int,
                    help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
parser.add_argument('--save-dir', default='saved', type=str,
                    help='directory of saved model (default: saved)')
parser.add_argument('--optim', default='Adam', type=str,
                    help='Optimizer to use: SGD or Adam')
parser.add_argument('--save-freq', default=1, type=int,
                    help='training checkpoint frequency (default: 1)')
parser.add_argument('--data-dir', default='datasets', type=str,
                    help='directory of training/testing data (default: datasets)')
parser.add_argument('--validation-split', default=0.1, type=float,
                    help='ratio of split validation data, [0.0, 1.0) (default: 0.1)')
parser.add_argument('--visualize', default=False, type=bool,
                    help='visualize in valid or not')
parser.add_argument('--use_video', default=False, type=bool,
                    help='use video as input or not (or use extracted features)')
parser.add_argument('--extract_only', default=True, type=bool,
                    help='The base network to extract feature only, not trainable')
parser.add_argument('--cuda', default=0, type=int,
                    help='Cuda GPU to use')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='Number of workders')
parser.add_argument('--num_frames', type=int, default=16, metavar='N',
                    help='Number of frames taken from each video')
parser.add_argument('--device-ids', type=list, default=[0, 1], metavar='N',
                    help='num of workers to fetch data')
parser.add_argument('--no-cuda', action="store_true",
                    help='use CPU instead of GPU')


def load_cifar_data_loader(args):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

    return trainloader, testloader


def load_moment_data_loader(args):
    from data_loader.moment_dataset import MomentDataset

    trainset = MomentDataset(args, 'data/Moment/', 'train', num_frames=args.num_frames)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)

    testset = MomentDataset(args, 'data/Moment/', 'test', num_frames=args.num_frames)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers)

    return train_loader, test_loader


def main(args):
    # Model
    if args.dataset == 'cifar10':
        if args.model == 'vgg':
            from model.vgg_cifar import VGGAttentionModel
            model = VGGAttentionModel(args, 'VGGOriginal16')
        elif args.model == 'fpn':
            from model.fpn_cifar import FPN
            model = FPN(args)
        elif args.model == 'resnet':
            from model.fpn_cifar import ResNetAttention
            model = ResNetAttention(args)
    elif args.dataset == 'moment':
        from model.moment_model import MomentModel2, MomentModel, MomentModel3
        model = MomentModel3(args)
    else:
        model = MnistModel()

    model.summary()

    # A logger to store training process information
    train_logger = Logger()

    # Specifying loss function, metric(s), and optimizer
    loss = cross_entropy_loss
    metrics = [acc_metric]
    if args.optim == 'Adam':
        # optimizer = optim.Adam([{'params': model.higher_model.parameters()},
        #                         {'params': model.base_model.parameters(), 'lr': args.lr * 0.1}],
        #                        lr=args.lr)
        optimizer = optim.Adam(model.higher_model.parameters()
                                if args.extract_only else model.parameters(),
                               lr=args.lr)
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if args.model == 'vgg':
        scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
    else:
        # scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    # Data loader and validation split

    if args.dataset == 'cifar10':
        data_loader, valid_data_loader = load_cifar_data_loader(args)
    elif args.dataset == 'moment':
        data_loader, valid_data_loader = load_moment_data_loader(args)
    else:
        data_loader = MnistDataLoader(args.data_dir, args.batch_size, shuffle=True)
        valid_data_loader = data_loader.split_validation(args.validation_split)

    # An identifier for this training session
    training_name = type(model).__name__

    # Trainer instance
    from trainer.trainer import Trainer

    trainer = Trainer(model, loss, metrics,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      epochs=args.epochs,
                      train_logger=train_logger,
                      save_dir=args.save_dir,
                      save_freq=args.save_freq,
                      resume=args.resume,
                      verbosity=args.verbosity,
                      args=args,
                      training_name=training_name,
                      with_cuda=not args.no_cuda,
                      monitor='val_acc_metric',
                      monitor_mode='max')

    # Start training!
    trainer.train()

    # See training history
    print(train_logger)


if __name__ == '__main__':
    main(parser.parse_args())
