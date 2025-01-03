# encoding: utf-8
"""
credit:
https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/train_loop.py
"""

import logging
import time
import weakref
from typing import Dict

import numpy as np
import torch
# from apex import amp
# from apex.parallel import DistributedDataParallel

import fastreid.utils.comm as comm
from fastreid.utils.events import EventStorage, get_event_storage

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer"]

logger = logging.getLogger(__name__)


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.
    Each hook can implement 6 methods. The way they are called is demonstrated
    in the following snippet:
    .. code-block:: python
        hook.before_train()
        for _ in range(start_epoch, max_epoch):
            hook.before_epoch()
            for iter in range(start_iter, max_iter):
                hook.before_step()
                trainer.run_step()
                hook.after_step()
            hook.after_epoch()
        hook.after_train()
    Notes:
        1. In the hook method, users can access `self.trainer` to access more
           properties about the context (e.g., current iteration).
        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.
           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.
    Attributes:
        trainer: A weak reference to the trainer object. Set by the trainer when the hook is
            registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_epoch(self):
        """
        Called before each epoch.
        """
        pass

    def after_epoch(self):
        """
        Called after each epoch.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass


class TrainerBase:
    """
    Base class for iterative trainer with hooks.
    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.
    Attributes:
        iter(int): the current iteration.
        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.
        max_iter(int): The iteration to end training.
        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self):
        self._hooks = []

    def register_hooks(self, hooks):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.
        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_epoch: int, max_epoch: int, iters_per_epoch: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """


        logger = logging.getLogger(__name__)
        logger.info("Starting training from epoch {}".format(start_epoch))

        self.iter = self.start_iter = start_epoch * iters_per_epoch

        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                for self.epoch in range(start_epoch, max_epoch):
                    self.before_epoch()
                    for _ in range(iters_per_epoch):#n iters_per_epoch
                        self.before_step()
                        self.run_step()
                        self.after_step()
                        self.iter += 1
                    self.after_epoch()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        self.storage.iter = self.iter
        for h in self._hooks:
            h.after_train()

    def before_epoch(self):
        self.storage.epoch = self.epoch

        for h in self._hooks:
            h.before_epoch()

    def before_step(self):
        self.storage.iter = self.iter

        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def after_epoch(self):
        for h in self._hooks:
            h.after_epoch()

    def run_step(self):
        raise NotImplementedError


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization.
    It assumes that every step, you:
    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.
    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, cfg,  model, data_loader_list, optimizer):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of heads.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        # self.data_loader = data_loader
        # self._data_loader_iter = iter(data_loader)
        self.data_loader_list = data_loader_list

        if cfg.DATASETS.BIG_to_SMALL:
            self._data_loader_iter_list = [iter(data_loader_list[0]), iter(data_loader_list[1]), iter(data_loader_list[2]), iter(data_loader_list[3])]
            self.src_idx = [0, 1, 2, 3]  ### for big to small 
        else:
            self._data_loader_iter_list = [iter(data_loader_list[0]), iter(data_loader_list[1]), iter(data_loader_list[2])]
            self.src_idx = [0, 1, 2]
        self.optimizer = optimizer

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        # self.model.train()

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """

        cur_idx = np.random.choice(self.src_idx, size=1, replace=True)[0] 
        data = next(self._data_loader_iter_list[cur_idx])
        data_time = time.perf_counter() - start

        """
        If your want to do something with the heads, you can wrap the model.
        """

        # loss_dict = self.model(data)
        loss_dict = self.model(data, cur_idx, self.epoch, use_EN = False)
        losses = sum(loss_dict.values())
        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(loss_dict, data_time)
        self.optimizer.step()



    def _write_metrics(self, loss_dict: Dict[str, torch.Tensor], data_time: float):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        device = next(iter(loss_dict.values())).device

        # Use a new stream so these ops don't wait for DDP or backward
        with torch.cuda.stream(torch.cuda.Stream() if device.type == "cuda" else None):
            metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
            metrics_dict["data_time"] = data_time

            # Gather metrics among all workers for logging
            # This assumes we do DDP-style training, which is currently the only
            # supported method in detectron2.
            all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)







class AMPTrainer(SimpleTrainer):
    """
    Like :class:`SimpleTrainer`, but uses apex automatic mixed precision
    in the training loop.
    """

    def run_step(self):
        """
        Implement the AMP training logic.
        """

        self.model.train()

        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"

        start = time.perf_counter()


        cur_idx = np.random.choice(self.src_idx, size=1, replace=True)[0] 
        data = next(self._data_loader_iter_list[cur_idx])


        data_time = time.perf_counter() - start

        # loss_dict = self.model(data)
        loss_dict = self.model(data, cur_idx, self.epoch, use_IN = False) 
        losses = sum(loss_dict.values())

        self.optimizer.zero_grad()

        with amp.scale_loss(losses, self.optimizer) as scaled_loss:
            scaled_loss.backward()

        self._write_metrics(loss_dict, data_time)

        self.optimizer.step()
