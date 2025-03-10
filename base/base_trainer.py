import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter

#  _******()：类内的私有方法

class BaseTrainer:
    """
    Base class for all trainers
    metric_ftns:性能评估方程
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger()

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']      #模型保存周期
        self.monitor = cfg_trainer.get('monitor','off')
        #从 cfg_trainer 中获取用于监控模型性能的指标设置。get 方法表示如果配置中没有 monitor 这个键，就默认使用 'off'，即不进行性能监控。

        #配置监督模型的性能并保存最优的
        if self.monitor == 'off':   #不进行性能监视
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()   #将moniter拆分为为监控模式 self.mnt_mode（'min' 或 'max'）和监控的指标名称 self.mnt_metric。
            assert self.mnt_mode in ['min', 'max']   #断言为只能是min或者max

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf

            #从配置中获取提前停止训练的条件，即如果模型在多少个 epoch 内性能没有提升就停止训练。
            # 如果配置中没有 early_stop 这个键，则默认值为正无穷大 inf表示不启用提前停止机制。
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)


    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError


    """
    保存检查点
    epoch:当前的轮数
    """
    def _save_checkpoint(self, epoch, save_bast=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__    #将当前模型的名称赋值给arch
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
        self.logger.info('Saving checkpoint: {} ...'.format(filename))
        if save_bast:
            best_path = str(self.checkpoint_dir / 'model_best-epoch {}.pth'.format(epoch))
            torch.save(state, best_path)
            self.logger.info("Saveing currect best: model_best.pth...")


    """
    用于从检查点恢复路径
    """
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

        # load architecture params from checkpoint.
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





    def train(self):
        """
        训练逻辑
        :return:
        """
        not_improved_count = 0   #用于记录模型性能为提升的轮次数量，初始化为0
        for epoch in range(self.start_epoch, self.epochs+1):    #注意从start_epoch开始训练，到
            result = self._train_epoch(epoch)

            #save logged informations into log dict
            log = {'epoch':epoch}
            log.update(result)

            #print logged informations
            for key, value in log.items():
                self.logger.info('  {:15s}: {}'.format(str(key),value))

            #通过配置的指标，评估模型性能，保存最好的检查点作为best_model
            best = False #初始化为false，表示当前不是最优的epoch
            if self.mnt_mode != 'off':  #表示配置了监控指标
                try:   #根据监控模式（min,max）和指定的监控指标（mnt_metric），判断模型新能是否提升
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:   #如果性能提升，更新性能指标，重置未提升轮数，否则，未提升轮数加一
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:   #如果超过阈值就停止
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:               #如果是检查周期整数倍，调用函数打印记录，并判断是否为best
                self._save_checkpoint(epoch, save_best=best)



