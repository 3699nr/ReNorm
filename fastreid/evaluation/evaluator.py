# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import logging
import time
from contextlib import contextmanager

import torch
import copy
from fastreid.utils.logger import log_every_n_seconds


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.
    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.
    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def preprocess_inputs(self, inputs):
        pass

    def process(self, inputs, outputs):
        """
        Process an input/output pair.
        Args:
            inputs: the inputs that's used to call the model.
            outputs: the return value of `model(input)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.
        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:
                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass

def inference_on_dataset(cfg, model, data_loader, evaluator, flip_test=False):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.
    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.
        flip_test (bool): If get features with flipped images
    Returns:
        The return value of `evaluator.evaluate()`
    """

    logger = logging.getLogger(__name__)

    if cfg.MODEL.META_ARCHITECTURE == 'RENORM':
        logger.info("Start inference on {} images".format(len(data_loader.dataset)))

        total = len(data_loader)
        evaluator.reset()
        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_compute_time = 0

        
        with inference_context(model), torch.no_grad():
            for idx, inputs in enumerate(data_loader):
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0

                start_compute_time = time.perf_counter()




                outputsRN_all, _ = model(inputs, 3, 0, use_EN=False)

                outputsEN_1, _ = model(inputs, 0, 0, use_EN=True)
                outputsEN_2, _ = model(inputs, 1, 0, use_EN=True)
                outputsEN_3, _ = model(inputs, 2, 0, use_EN=True)

                if flip_test:
                    inputs["images"] = inputs["images"].flip(dims=[3])

                    # flip_outputs = model(inputs)

                    flip_outputsRN_all, _ = model(inputs, 3, 0, use_EN=False)

                    flip_outputsEN_1, _ = model(inputs, 0, 0, use_EN=True)
                    flip_outputsEN_2, _ = model(inputs, 1, 0, use_EN=True)
                    flip_outputsEN_3, _ = model(inputs, 2, 0, use_EN=True)


                    outputsRN_allf = (outputsRN_all + flip_outputsRN_all) / 2

                    outputsEN_1f = (outputsEN_1 + flip_outputsEN_1) / 2
                    outputsEN_2f = (outputsEN_2 + flip_outputsEN_2) / 2
                    outputsEN_3f = (outputsEN_3 + flip_outputsEN_3) / 2
                    
                    outputs = outputsRN_allf + (outputsEN_1f + outputsEN_2f + outputsEN_3f) / 3
                    #outputs = outputsRN_allf  ### RN only
                    #outputs = (outputsEN_1f + outputsEN_2f + outputsEN_3f) / 3  ### EN only
                else:
                    outputs = outputsRN_all + (outputsEN_1 + outputsEN_2 + outputsEN_3) / 3


                total_compute_time += time.perf_counter() - start_compute_time
                evaluator.process(inputs, outputs)
                idx += 1
                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                seconds_per_batch = total_compute_time / iters_after_start
                if idx >= num_warmup * 2 or seconds_per_batch > 30:
                    total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                    eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                            idx + 1, total, seconds_per_batch, str(eta)
                        ),
                        n=30,
                    )

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total inference time: {} ({:.6f} s / batch per device)".format(
                total_time_str, total_time / (total - num_warmup)
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / batch per device)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup)
            )
        )
        results = evaluator.evaluate()     

        if results is None:
            results = {}
        return results
    
    else:

        logger.info("Start inference on {} images".format(len(data_loader.dataset)))

        total = len(data_loader)
        evaluator.reset()
        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_compute_time = 0

        
        with inference_context(model), torch.no_grad():
            for idx, inputs in enumerate(data_loader):
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0

                start_compute_time = time.perf_counter()

                outputsRN_all, _ = model(inputs, 3, 0, use_EN=False)

                if flip_test:
                    inputs["images"] = inputs["images"].flip(dims=[3])

                    # flip_outputs = model(inputs)
                    flip_outputsRN_all, _ = model(inputs, 3, 0, use_EN=False)

                    outputsRN_allf = (outputsRN_all + flip_outputsRN_all) / 2

                    
                    outputs = outputsRN_allf 
                else:
                    outputs = outputsRN_all


                total_compute_time += time.perf_counter() - start_compute_time
                evaluator.process(inputs, outputs)
                idx += 1
                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                seconds_per_batch = total_compute_time / iters_after_start
                if idx >= num_warmup * 2 or seconds_per_batch > 30:
                    total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                    eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                            idx + 1, total, seconds_per_batch, str(eta)
                        ),
                        n=30,
                    )

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total inference time: {} ({:.6f} s / batch per device)".format(
                total_time_str, total_time / (total - num_warmup)
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / batch per device)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup)
            )
        )
        results = evaluator.evaluate()   

        if results is None:
            results = {}
        return results
    
@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
