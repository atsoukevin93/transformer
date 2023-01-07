import torch
import os
from abc import abstractmethod
from numpy import inf, sqrt
from logger import TensorboardWriter
from utils import inf_loop, exists
from models.metrics import MetricTracker
from typing import TYPE_CHECKING, Optional, Callable

if TYPE_CHECKING:
    from base.base_data_loader import BaseDataLoader
    from models.transformer_encoder_decoder import Transformer
    from configs.config_parser import ConfigParser


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(
            self,
            model: Optional["Transformer"] = None,
            train_data_loader: Optional["BaseDataLoader"] = None,
            valid_data_loader: Optional["BaseDataLoader"] = None,
            metric_ftns: list[Callable] = None,
            config: Optional["ConfigParser"] = None,
            len_epoch: int = None,
            n_epochs: int = 10,
            learning_rate: float = 1e-4,
            betas: list[float] = [0.9, 0.95],
            grad_norm_clip: float = 1.0,
            weight_decay: float = 0.1,
            lr_decay: bool = True,
            # save_dir: str = 'saved/',
            save_period: int = 1,
            logger_verbosity: int = 2,
            monitor: str = "min val_loss",
            early_stop: int = 10,
            tensorboard: bool = True,
            **kwargs
    ):
        # cfg_trainer = config['trainer']
        self.verbosity = logger_verbosity
        if exists(config):
            self.logger = config.get_logger('trainer', self.verbosity)

        self.grad_norm_clip = grad_norm_clip
        self.lr_decay = lr_decay
        # self.save_dir = save_dir
        #  Model
        self.model = model
        if exists(model):
            self.device = next(self.model.parameters()).device
        if exists(model):
            # raw_model = model.module if hasattr(self.model, "module") else model
            self.optimizer = model.configure_optimizers(
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                betas=betas
            )
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.betas = betas

        self.train_data_loader = train_data_loader
        self.metric_ftns = metric_ftns

        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=2000)
        self.lr_scheduler = None

        self.n_epochs = n_epochs

        if exists(train_data_loader):
            if not exists(len_epoch):
                # epoch-base training
                self.len_epoch = len(self.train_data_loader)
            else:
                # iteration-based training
                self.train_data_loader = inf_loop(train_data_loader)
                self.len_epoch = len_epoch
        else:
            self.len_epoch = None

        self.valid_data_loader = valid_data_loader
        self.do_validation = exists(self.valid_data_loader)
        # self.lr_scheduler = lr_scheduler

        self.current_tokens = 0

        if exists(self.train_data_loader):
            self.log_step = int(sqrt(train_data_loader.batch_size))
        else:
            self.log_step = None

        self.save_period = save_period
        # self.monitor = cfg_trainer.get('monitor', 'off')
        self.monitor = monitor

        # configuration to monitor blocks performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            # self.early_stop = cfg_trainer.get('early_stop', inf)
            self.early_stop = early_stop
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.config = config

        # setup visualization writer instance
        self.tensorboard = tensorboard

        if exists(config):
            self.writer = TensorboardWriter(config.log_dir, self.logger, self.tensorboard)
            self.checkpoint_dir = config.save_dir

        if exists(metric_ftns):
            self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
            self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        if exists(config) and exists(config.resume):
            self._resume_checkpoint(config.resume)

        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        """
        Validation logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def update_learning_rate(self, target):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def save_losses_to_csv(self, current_step, epoch, loss, output, target, is_train: bool = True):
        if is_train:
            modeling_step = "train"
        else:
            modeling_step = "test"
        metrics_dict = {'modeling_step': modeling_step, 'epoch': epoch, 'loss': loss.item()}
        for met in self.metric_ftns:
            metrics_dict[met.__name__] = met(output.contiguous().view(-1, output.size(-1)),
                                             target.contiguous().view(-1))

        self.train_metrics.save_metrics(
            metrics_dict, current_step, os.path.join(self.checkpoint_dir, f"{self.model.__class__.__name__}_metrics.csv")
        )

        del metrics_dict

    def set_model(self, model):
        self.model = model
        if exists(model):
            # raw_model = model.module if hasattr(self.model, "module") else model
            self.optimizer = self.model.configure_optimizers(
                weight_decay=self.weight_decay,
                learning_rate=self.learning_rate,
                betas=self.betas
            )
            self.device = next(self.model.parameters()).device

    def set_train_data_loader(self, train_data_loader):
        self.train_data_loader = train_data_loader
        if exists(train_data_loader):
            if not exists(self.len_epoch):
                # epoch-base training
                self.len_epoch = len(self.train_data_loader)
            else:
                # iteration-based training
                self.train_data_loader = inf_loop(train_data_loader)
            self.log_step = int(sqrt(train_data_loader.batch_size))
        else:
            self.len_epoch = None
            self.log_step = None

    def set_valid_data_loader(self, valid_data_loader):
        self.valid_data_loader = valid_data_loader
        self.do_validation = True

    def set_metric_ftns(self, metric_ftns):
        self.metric_ftns = metric_ftns

    def set_config(self, config):
        self.config = config
        if exists(config):
            self.logger = config.get_logger('trainer', self.verbosity)
            self.checkpoint_dir = config.save_dir

            # setup visualization writer instance
            self.writer = TensorboardWriter(config.log_dir, self.logger, self.tensorboard)
            if exists(self.metric_ftns):
                self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
                self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        if exists(config) and exists(config.resume):
            self._resume_checkpoint(config.resume)

    def setup(
            self,
            model: Optional["Transformer"] = None,
            train_data_loader: Optional["BaseDataLoader"] = None,
            valid_data_loader: Optional["BaseDataLoader"] = None,
            metric_ftns: list[Callable] = None,
            config: Optional["ConfigParser"] = None
    ):
        self.set_model(model)
        self.set_train_data_loader(train_data_loader)
        self.set_valid_data_loader(valid_data_loader)
        self.set_metric_ftns(metric_ftns)
        self.set_config(config)
    # @abstractmethod
    # def _transformer_train_epoch(self, epoch):
    #     """
    #     Training logic for the Encoder Decoder blocks
    #
    #     :param epoch: Current epoch number
    #     """
    #     raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        if not exists(self.model):
            raise RuntimeError("The trainer requires a `model` argument")
        if not exists(self.train_data_loader):
            raise RuntimeError("The trainer requires a `train_data_loader` argument")
        if not exists(self.config):
            raise RuntimeError("The trainer requires a `config` argument")

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.n_epochs + 1):
            result = self._train_epoch(epoch)
            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate blocks performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether blocks performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load blocks params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
