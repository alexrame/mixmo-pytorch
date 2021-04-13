"""
Scheduler definitions and factory
"""

from torch.optim.lr_scheduler import Counter, _LRScheduler
from mixmo.utils.logger import get_logger


LOGGER = get_logger(__name__, level="DEBUG")

class MultiGammaStepLR(_LRScheduler):
    """
    Multi step decay scheduler, with decay applied to the learning rate every set milestone
    """

    def __init__(self, optimizer, dict_milestone_to_gamma, last_epoch=-1):
        self.milestones = Counter(dict_milestone_to_gamma.keys())
        self.dict_milestone_to_gamma = dict_milestone_to_gamma
        super(MultiGammaStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        gamma = self.dict_milestone_to_gamma[self.last_epoch]
        LOGGER.warning(f"Decrease lr by gamma: {gamma} at epoch: {self.last_epoch}")
        return [
            group['lr'] * gamma
            for group in self.optimizer.param_groups
        ]


SCHEDULERS = {
    "multigamma_step": MultiGammaStepLR,
}


def get_scheduler(lr_schedule, optimizer, start_epoch):
    """
    Build the scheduler object
    """
    scheduler_name = lr_schedule.pop("name")

    scheduler_params = lr_schedule["params"]
    # Add last epoch
    scheduler_params["last_epoch"] = start_epoch
    LOGGER.info(f"Using {scheduler_name} scheduler with {scheduler_params} params")
    base_scheduler = SCHEDULERS[scheduler_name](optimizer, **scheduler_params)
    return base_scheduler


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_steps: target learning rate is reached at total_steps, gradually
    """

    def __init__(self, optimizer, multiplier, total_steps):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_steps = int(total_steps)
        self.finished = False
        self.last_steps = 0
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr_warmup(self):
        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_steps) / self.total_steps) for base_lr in self.base_lrs
            ]
        else:
            raise NotImplementedError

    def step(self, steps=None):
        if steps is None:
            steps = self.last_steps + 1
        self.last_steps = steps if steps != 0 else 1
        if self.last_steps <= self.total_steps:
            warmup_lr = self.get_lr_warmup()
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
            if self.last_steps == self.total_steps:
                LOGGER.warning(f"This is the end of warmup at lr: {warmup_lr}")



def get_warmup_scheduler(optimizer, warmup_period):
    """
    Build a Scheduler instance with warmup
    """
    return GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_steps=warmup_period,
    )
