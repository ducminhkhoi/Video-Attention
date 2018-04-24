import numpy as np
import torch
from torch.autograd import Variable
from base import BaseTrainer
from utils.util import invTrans
import matplotlib.pyplot as plt
from tqdm import tqdm


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        Modify __init__() if you have additional arguments to pass.
    """

    def __init__(self, model, loss, metrics, data_loader, optimizer, scheduler, epochs,
                 save_dir, save_freq, resume, with_cuda, verbosity, args, training_name='',
                 valid_data_loader=None, train_logger=None, monitor='loss', monitor_mode='min'):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, scheduler, epochs,
                                      save_dir, save_freq, resume, verbosity, training_name,
                                      with_cuda, train_logger, monitor, monitor_mode)
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        self.args = args

    def _to_variable(self, tensor, volatile=False):
        tensor = Variable(tensor, volatile=volatile)
        if self.with_cuda:
            tensor = tensor.cuda()
        return tensor

    def _to_tensor(self, variable, numpy=False):
        tensor = variable.data
        if self.with_cuda:
            tensor = tensor.cpu()

        if numpy:
            tensor = tensor.numpy()

        return tensor

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
        """

        with torch.cuda.device(self.args.cuda):

            self.model.base_model.eval()
            self.model.higher_model.train()
            if self.with_cuda:
                self.model.cuda()

            total_loss = 0
            total_metrics = np.zeros(len(self.metrics))

            with tqdm(total=len(self.data_loader)) as pbar:
                for batch_idx, (data, target, index, images) in enumerate(self.data_loader):
                    pbar.update()
                    data, target = self._to_variable(data), self._to_variable(target)

                    self.optimizer.zero_grad()
                    output = self.model(data, y=target, images=images, indices=index)
                    loss = self.loss(output, target)
                    loss.backward()
                    self.optimizer.step()

                    for i, metric in enumerate(self.metrics):
                        y_output = output.data.cpu().numpy()
                        y_output = np.argmax(y_output, axis=1)
                        y_target = target.data.cpu().numpy()
                        total_metrics[i] += metric(y_output, y_target)

                    total_loss += loss.data[0]
                    log_step = int(np.sqrt(self.batch_size))
                    if self.verbosity >= 2 and batch_idx % log_step == 0:
                        self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(self.data_loader) * len(data),
                                   100.0 * batch_idx / len(self.data_loader), loss.data[0]))

                    pbar.set_description(
                        'Loss: {:.6f} Acc: {:.6f}'.format(total_loss / (batch_idx + 1),
                                                          total_metrics[0] / (batch_idx + 1)))

            avg_loss = total_loss / len(self.data_loader)
            avg_metrics = (total_metrics / len(self.data_loader)).tolist()
            log = {'loss': avg_loss, 'metrics': avg_metrics}

            try:
                self.scheduler.step()
            except TypeError:
                self.scheduler.step(avg_loss)

            if self.valid:
                val_log = self._valid_epoch()
                log = {**log, **val_log}

            return log

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """
        self.model.eval()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with tqdm(total=len(self.valid_data_loader)) as pbar:
            for batch_idx, (data_, target, index) in enumerate(self.valid_data_loader):
                pbar.update()
                data, target = self._to_variable(data_, volatile=True), self._to_variable(target, volatile=True)

                output = self.model(data, index, train=False)
                loss = self.loss(output, target)
                total_val_loss += loss.data[0]

                # for visualization
                if self.args.visualize:
                    a1s = self._to_tensor(a1s.squeeze(), True)
                    a2s = self._to_tensor(a2s.squeeze(), True)
                    a3s = self._to_tensor(a3s.squeeze(), True)
                    for image, a1, a2, a3 in zip(data_, a1s, a2s, a3s):
                        image = invTrans(image)

                        plt.imshow(a1, cmap=plt.cm.viridis, interpolation='bilinear')
                        plt.imshow(a2, cmap=plt.cm.viridis, interpolation='bilinear', alpha=.7)
                        plt.imshow(a3, cmap=plt.cm.viridis, interpolation='bilinear', alpha=.5)
                        plt.imshow(image, alpha=0.5)
                        plt.show()

                for i, metric in enumerate(self.metrics):
                    y_output = output.data.cpu().numpy()
                    y_output = np.argmax(y_output, axis=1)
                    y_target = target.data.cpu().numpy()
                    total_val_metrics[i] += metric(y_output, y_target)

            avg_val_loss = total_val_loss / len(self.valid_data_loader)
            avg_val_metrics = (total_val_metrics / len(self.valid_data_loader)).tolist()
            return {'val_loss': avg_val_loss, 'val_metrics': avg_val_metrics}
