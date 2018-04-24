import torch.utils.data as data
from torchvision.transforms import transforms
import torch
import numpy as np
import pickle
import sys
from tqdm import tqdm
import lmdb
import cv2


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


class MomentDataset(data.Dataset):
    def __init__(self, args, root, mode='train', num_frames=64):
        # TODO
        # 1. Initialize file path or list of file names.
        self.root = root
        mode = 'training' if mode == 'train' else 'validation'
        self.mode = mode
        self.video_folder = root + mode
        self.num_frames = num_frames

        with open(self.video_folder + 'Set.csv', 'r') as f:
            self.list_files = [line.split(',')[0] for line in f.readlines()]

        with open(root + 'moments_categories.txt', 'r') as f:
            self.action_to_index = {x.split(',')[0]: i for i, x in enumerate(f.readlines())}

        self.select_frames = np.linspace(0, 89, num_frames, dtype=int)
        self.args = args

    def __getitem__(self, index):

        filename = self.list_files[index]
        action_class = self.action_to_index[filename.split('/')[0]]

        if self.args.use_video:
            cap = cv2.VideoCapture(self.video_folder + '/' + filename)

            list_frames = []

            for i in self.select_frames:
                cap.set(1, i)
                ret, frame = cap.read()
                if ret:
                    list_frames.append(self.transform(frame))
                else:
                    list_frames.append(list_frames[-1])

            list_frames = torch.stack(list_frames)

        else:
            cap = cv2.VideoCapture(self.video_folder + '/' + filename)

            list_images = []

            for i in self.select_frames:
                cap.set(1, i)
                ret, frame = cap.read()
                if ret:
                    list_images.append(cv2.resize(frame, (256, 256)))
                else:
                    list_images.append(list_images[-1])

            list_images = np.stack(list_images)

            list_frames = torch.from_numpy(np.load(self.root + 'np_arrays/' + self.mode + '/{}.npy'.format(index)))

        return list_frames, action_class, index, list_images

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.list_files)

    @staticmethod
    def transform(x):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.ToTensor(),
            normalize,
        ])(x)


if __name__ == '__main__':
    list_length = []
    dataset = MomentDataset('../data/Moment/', mode='test', num_frames=16)

    with tqdm(total=len(dataset)) as pbar:
        for i, data in enumerate(dataset):
            pbar.update()
            list_length.append(data)

    print(max(list_length), min(list_length))

    # trainset = MomentDataset('../data/Moment/', 'train', num_frames=16)
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)
    #
    # with tqdm(total=len(train_loader)) as pbar:
    #     for i, data in enumerate(train_loader):
    #         pbar.update()

    #     length = len(data[0])
    #     print(i, length)
    #     list_length.append(length)
    #
    # print(min(list_length), max(list_length))
