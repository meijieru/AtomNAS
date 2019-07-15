import copy
import importlib
import logging
import math
import os

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from utils import distributed as udist
from utils.model_profiling import model_profiling
from utils.config import FLAGS
from utils.meters import ScalarMeter
from utils.meters import flush_scalar_meters
from utils.common import get_params_by_name

import models.mobilenet_base as mb

summary_writer = None


class SummaryWriterManager(object):
    """Manage `summary_writer`."""

    @udist.master_only
    def __enter__(self):
        global summary_writer
        if summary_writer is not None:
            raise RuntimeError('Should only init `summary_writer` once')
        summary_writer = SummaryWriter(os.path.join(FLAGS.log_dir, 'log'))

    @udist.master_only
    def __exit__(self, exc_type, exc_value, exc_traceback):
        global summary_writer
        if summary_writer is None:
            raise RuntimeError('`summary_writer` lost')
        summary_writer.close()
        summary_writer = None


def setup_ema(model):
    """Setup EMA for model's weights."""
    from utils import optim

    ema = None
    if FLAGS.moving_average_decay > 0.0:
        if FLAGS.moving_average_decay_adjust:
            moving_average_decay = \
                optim.ExponentialMovingAverage.adjust_momentum(
                    FLAGS.moving_average_decay,
                    FLAGS.moving_average_decay_base_batch / FLAGS.batch_size)
        else:
            moving_average_decay = FLAGS.moving_average_decay
        logging.info('Moving average for model parameters: {}'.format(
            moving_average_decay))
        ema = optim.ExponentialMovingAverage(moving_average_decay)
        for name, param in model.named_parameters():
            ema.register(name, param)
        # We maintain mva for batch norm moving mean and variance as well.
        for name, buffer in model.named_buffers():
            if 'running_var' in name or 'running_mean' in name:
                ema.register(name, buffer)
    return ema


def forward_loss(model, criterion, input, target, meter):
    """Forward model and return loss."""
    output = model(input)
    loss = criterion(output, target)
    meter['loss'].cache_list(loss.tolist())
    # topk
    _, pred = output.topk(max(FLAGS.topk))
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    for k in FLAGS.topk:
        correct_k = correct[:k].float().sum(0)
        error_list = list(1. - correct_k.cpu().detach().numpy())
        meter['top{}_error'.format(k)].cache_list(error_list)
    return torch.mean(loss)


def reduce_and_flush_meters(meters, method='avg'):
    """Sync and flush meters."""
    if not FLAGS.use_distributed:
        results = flush_scalar_meters(meters)
    else:
        results = {}
        assert isinstance(meters, dict), "meters should be a dict."
        # NOTE: Ensure same order, otherwise may deadlock
        for name in sorted(meters.keys()):
            meter = meters[name]
            if not isinstance(meter, ScalarMeter):
                continue
            if method == 'avg':
                method_fun = torch.mean
            elif method == 'sum':
                method_fun = torch.sum
            elif method == 'max':
                method_fun = torch.max
            elif method == 'min':
                method_fun = torch.min
            else:
                raise NotImplementedError(
                    'flush method: {} is not yet implemented.'.format(method))
            tensor = torch.tensor(meter.values).cuda()
            gather_tensors = [
                torch.ones_like(tensor) for _ in range(udist.get_world_size())
            ]
            dist.all_gather(gather_tensors, tensor)
            value = method_fun(torch.cat(gather_tensors))
            meter.flush(value)
            results[name] = value
    return results


def get_meters(phase):
    """Util function for meters."""
    meters = {}
    meters['loss'] = ScalarMeter('{}_loss'.format(phase))
    for k in FLAGS.topk:
        meters['top{}_error'.format(k)] = ScalarMeter('{}_top{}_error'.format(
            phase, k))
    return meters


def get_model():
    """Build and init model with wrapper for parallel."""
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(**FLAGS.model_kwparams, input_size=FLAGS.image_size)
    if FLAGS.reset_parameters:
        init_method = FLAGS.get('reset_param_method', None)
        if init_method is None:
            pass  # fall back to model's initialization
        elif init_method == 'slimmable':
            model.apply(mb.init_weights_slimmable)
        elif init_method == 'mnas':
            model.apply(mb.init_weights_mnas)
        else:
            raise ValueError('Unknown init method: {}'.format(init_method))
        logging.info('Init model by: {}'.format(init_method))
    if FLAGS.use_distributed:
        model_wrapper = udist.AllReduceDistributedDataParallel(model.cuda())
    else:
        model_wrapper = torch.nn.DataParallel(model).cuda()
    return model, model_wrapper


def unwrap_model(model_wrapper):
    """Remove model's wrapper."""
    model = model_wrapper.module
    return model


def get_ema_model(ema, model_wrapper):
    """Generate model from ExponentialMovingAverage.

    NOTE: If `ema` is given, generate a new model wrapper. Otherwise directly
        return `model_wrapper`, in this case modifying `model_wrapper` also
        influence the following process.
    FIXME(meijieru): Always return a new model wrapper.
    """
    if ema is not None:
        model_eval_wrapper = copy.deepcopy(model_wrapper)
        model_eval = unwrap_model(model_eval_wrapper)
        names = ema.average_names()
        params = get_params_by_name(model_eval, names)
        for name, param in zip(names, params):
            param.data.copy_(ema.average(name))
    else:
        model_eval_wrapper = model_wrapper
    return model_eval_wrapper


def profiling(model, use_cuda):
    """Profiling on either gpu or cpu."""
    logging.info('Start model profiling, use_cuda:{}.'.format(use_cuda))
    model_profiling(model,
                    FLAGS.image_size,
                    FLAGS.image_size,
                    verbose=getattr(FLAGS, 'model_profiling_verbose', True)
                    and udist.is_master())


def setup_distributed(num_images=None):
    """Setup distributed related parameters."""
    # init distributed
    if FLAGS.use_distributed:
        udist.init_dist()
        FLAGS.batch_size = udist.get_world_size() * FLAGS.per_gpu_batch_size
        FLAGS._loader_batch_size = FLAGS.per_gpu_batch_size
        if FLAGS.bn_calibration:
            FLAGS._loader_batch_size_calib = \
                FLAGS.bn_calibration_per_gpu_batch_size
        FLAGS.data_loader_workers = round(FLAGS.data_loader_workers
                                          / udist.get_local_size())
    else:
        count = torch.cuda.device_count()
        FLAGS.batch_size = count * FLAGS.per_gpu_batch_size
        FLAGS._loader_batch_size = FLAGS.batch_size
        if FLAGS.bn_calibration:
            FLAGS._loader_batch_size_calib = \
                FLAGS.bn_calibration_per_gpu_batch_size * count
    if hasattr(FLAGS, 'base_lr'):
        FLAGS.lr = FLAGS.base_lr * (FLAGS.batch_size / FLAGS.base_total_batch)
    if num_images:
        # NOTE: don't drop last batch, thus must use ceil, otherwise learning
        # rate will be negative
        FLAGS._steps_per_epoch = math.ceil(num_images / FLAGS.batch_size)
