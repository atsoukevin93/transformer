import math
import logging
import os.path

from tqdm import tqdm
import numpy as np

import torch
from base.base_trainer import BaseTrainer
from models.losses import cross_entropy_loss
from typing import Callable


# logger = logging.getLogger(__name__)


class ClassicTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, **kwargs):
        super(ClassicTrainer, self).__init__(**kwargs)

    def _train_epoch(self, epoch, criterion: Callable = cross_entropy_loss):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        n = 0
        for batch_idx, (input, target) in enumerate(self.train_data_loader):
            # Place the data on the correct device
            # data, target, pos = data.to(self.model.device), target.to(self.model.device), pos.to(self.model.device)
            data, target = data.to(self.device), target.to(self.device)
            # model_inputs = tuple(map(lambda x: x.to(self.model.device), model_inputs))

            # Forward the blocks
            # torch.autograd.set_detect_anomaly(True)
            with torch.set_grad_enabled(True):
                output = self.model(data)
                loss = criterion(output, target)
                loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                # print(f"the encoder is {print(self.blocks.encoder.weight.grad)}")
                # print(f"the decoder is {print(self.blocks.decoder.weight.grad)}")
                # losses.append(loss.item())

            # backprop and update the parameters
            self.model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            # out = self.blocks.generator(logits)
            for met in self.metric_ftns:
                self.train_metrics.update(
                    met.__name__,
                    met(output, target)
                )

            # Update accuracy metrics
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(logits, target))

            # decay the learning rate based on our progress
            lr = self.update_learning_rate(target)
            # self.update_learning_rate(target)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} Learning rate: {:.8f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    lr
                ))

                self.save_losses_to_csv(n, epoch, loss, output, target)
                n += 1

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        return log

    def update_learning_rate(self, target):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        else:
            # cfg_trainer = self.config['trainers']
            if self.lr_decay:
                self.current_tokens += (target > 0).sum()  # number of tokens processed this step (i.e. label is not 0)
                if self.current_tokens < self.warmup_tokens:
                    # linear warmup
                    lr_mult = float(self.current_tokens) / float(max(1, self.warmup_tokens))
                else:
                    # cosine learning rate decay
                    progress = float(self.current_tokens - self.warmup_tokens) / float(
                        max(1, self.final_tokens - self.warmup_tokens))
                    lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                lr = self.learning_rate * lr_mult
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                return lr

    def _valid_epoch(self, epoch, criterion: Callable = cross_entropy_loss):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        n = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = criterion(output, target)
                loss = loss.mean()

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(
                        met.__name__,
                        met(output, target)
                    )
                if batch_idx % self.log_step == 0:
                    self.save_losses_to_csv(n, epoch, loss, output, target, is_train=False)
                    n += 1

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
