from .fpn_cifar import ResNetAttention
from .vgg_cifar import VGGModel
from base import BaseModel
from torch import nn
import torch
import torch.nn.functional as F
from torchvision.models import vgg16_bn
from torch.autograd import Variable
import cv2
from utils.util import show_cam_on_image
import matplotlib.pyplot as plt

with open('data/Moment/moments_categories.txt', 'r') as f:
    index_to_action = {}
    for line in f.readlines():
        name, idx = line.split(',')
        index_to_action[int(idx[:-1])] = name


class MomentModel(BaseModel):
    def __init__(self, args):
        super(MomentModel, self).__init__()
        self.base_model = VGGModel(args)
        self.higher_model = TCN(args)
        if len(args.device_ids) > 1:
            self.base_model = torch.nn.DataParallel(self.base_model, device_ids=args.device_ids)
        else:
            self.base_model.cuda(args.device_ids[0])

    def forward(self, x):
        x_ = x.view(-1, *x.shape[2:])
        x_ = self.base_model(x_)
        x = x_.view(*x.shape[:2], -1).permute(0, 2, 1)
        out = self.higher_model(x)

        return out


class TCN(BaseModel):
    def __init__(self, args):
        super(TCN, self).__init__()
        self.args = args
        self.block1 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.MaxPool1d(2, 2),
            nn.Conv1d(512, 1024, kernel_size=5, padding=2),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, kernel_size=5, padding=2),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )

        self.block3 = nn.Sequential(
            nn.MaxPool1d(2, 2),
            nn.Conv1d(1024, 2048, kernel_size=5, padding=2),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, 2048, kernel_size=5, padding=2),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True)
        )

        self.transform1 = nn.Conv1d(512, 1024, 1)
        self.transform2 = nn.Conv1d(1024, 1024, 1)
        self.transform3 = nn.Conv1d(2048, 1024, 1)

        self.linear1 = nn.Linear(2048, 1024)
        self.linear2 = nn.Linear(1024*3, args.num_classes)

    def forward(self, x):

        l1 = self.block1(x)
        l2 = self.block2(l1)
        l3 = self.block3(l2)
        x = F.adaptive_max_pool1d(l3, 1).view(x.size(0), -1)
        g = self.linear1(x)

        l1 = self.transform1(l1)
        l2 = self.transform2(l2)
        l3 = self.transform3(l3)

        c1 = (l1 * g[:, :, None]).sum(1).unsqueeze(1)
        c2 = (l2 * g[:, :, None]).sum(1).unsqueeze(1)
        c3 = (l3 * g[:, :, None]).sum(1).unsqueeze(1)

        a1 = F.softmax(c1, -1)
        a2 = F.softmax(c2, -1)
        a3 = F.softmax(c3, -1)

        g1 = (a1 * l1).sum(-1)
        g2 = (a2 * l2).sum(-1)
        g3 = (a3 * l3).sum(-1)

        g = self.linear2(torch.cat([g1, g2, g3], 1))

        return g


class Conv3D(BaseModel):
    def __init__(self, args):
        super(Conv3D, self).__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Conv3d(512, 512, kernel_size=3, padding=1),  # 0
            nn.BatchNorm3d(512),  # 1
            nn.ReLU(inplace=True),  # 2
            nn.Conv3d(512, 512, kernel_size=3, padding=1),  # 3
            nn.BatchNorm3d(512),  # 4
            nn.ReLU(inplace=True),  # 5
            # nn.MaxPool3d((2, 2, 2)),

            # Block 2
            nn.Conv3d(512, 1024, kernel_size=3, padding=1, stride=2),  # 6
            nn.BatchNorm3d(1024),  # 7
            nn.ReLU(inplace=True),  # 8
            nn.Conv3d(1024, 1024, kernel_size=3, padding=1, stride=2),  # 9
            nn.BatchNorm3d(1024),  # 10
            nn.ReLU(inplace=True),  # 11
            nn.AdaptiveMaxPool3d((1, 1, 1)),  # 12
        )

        self.linear = nn.Linear(1024, args.num_classes)

    def forward(self, x):
        x = self.net(x).squeeze()
        out = self.linear(x)
        return out


class MomentModel2(BaseModel):
    def __init__(self, args):
        super(MomentModel2, self).__init__()
        self.args = args

        self.higher_model = Conv3D(args)

        if args.use_video:
            self.base_model = vgg16_bn(pretrained=True).features

            # if args.extract_only:
            #     for param in self.base_model.parameters():
            #         param.requires_grad = False

            if len(args.device_ids) > 1:
                self.base_model = torch.nn.DataParallel(self.base_model, device_ids=args.device_ids)
        else:
            if len(args.device_ids) > 1:
                self.higher_model = torch.nn.DataParallel(self.higher_model, device_ids=args.device_ids)

    def forward(self, x, indices=None, train=True):
        if self.args.use_video:
            if self.args.extract_only and train:
                x = Variable(x.data, volatile=True)

            x_ = x.view(-1, *x.shape[2:])
            x_ = self.base_model(x_)
            x = x_.view(*x.shape[:2], *x_.shape[1:]).permute(0, 2, 1, 3, 4)

            # mode = 'training' if train else 'validation'
            #
            # for tensor, index in zip(x, indices):
            #     np.save(f'data/Moment/vgg_64/{mode}/{index}.npy', tensor)
            #     torch.save(tensor, f'data/Moment/vgg_64/{mode}/{index}.pth')

            if self.args.extract_only and train:
                x = Variable(x.data)

        out = self.higher_model(x)

        return out


class MomentModel3(BaseModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.base_model = Conv3D(args)
        self.higher_model = Conv3D(args)

        pretrained_dict = torch.load('saved/MomentModel2/extract_feature/model_best_16.pth.tar')['state_dict']
        new_dict_state = {}
        for k, v in pretrained_dict.items():
            new_dict_state['.'.join(k.split('.')[1:])] = v

        self.base_model.load_state_dict(new_dict_state)
        self.higher_model.load_state_dict(new_dict_state)

        for param in self.base_model.parameters():
            param.requires_grad = False

    def get_cam(self, input, output, target, train=True):
        if train:
            indices = target.view(-1, 1)
            one_hot = output.gather(1, indices)
        else:
            one_hot = output.max(-1)[0]

        one_hot.backward(torch.ones_like(one_hot))

        weights = F.adaptive_avg_pool3d(input.grad, 1)
        cam = (weights * input).sum(1, keepdim=True) + 1
        cam = F.relu(cam)

        min_ = -F.adaptive_max_pool3d(-cam, 1)
        max_ = F.adaptive_max_pool3d(cam, 1)
        cam = (cam - min_)/(max_ - min_)

        scale_cam = F.upsample(cam, (cam.size(2), 256, 256), mode='trilinear')

        return cam, scale_cam

    def forward(self, x, y=None, train=True, images=None, indices=None):
        input = Variable(x.data, requires_grad=True)

        output = self.base_model(input)
        A, cams = self.get_cam(input, output, y, train)

        if self.args.visualize:
            for video, cam, index, y_ in zip(images, cams, indices, y):
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('video_output/{}_{}.avi'.format(index, index_to_action[int(y_)]),
                                      fourcc, 16 / 3., (256, 256))

                for i in range(self.args.num_frames):
                    cam_image = show_cam_on_image(video[i].numpy(), cam[0, i].data.cpu().numpy())
                    out.write(cam_image)
                    plt.show()

                out.release()

        Ahat = torch.zeros_like(A)

        for _ in range(3):
            A = torch.max(Ahat, A)
            # Need a threshold here
            Fhat = (1 - A) * input

            Fhat = Variable(Fhat.data, requires_grad=True)

            output = self.higher_model(Fhat)
            Ahat, cams = self.get_cam(Fhat, output, y, train)
            pass

        if self.args.extract_only and train:
            output = Variable(output.data)

        out = None

        return out
