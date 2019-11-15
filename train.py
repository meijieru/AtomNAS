"""Train and val."""
import copy
import importlib
import logging
import os
import time

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from utils.model_profiling import model_profiling
from utils.config import FLAGS, _ENV_EXPAND
from utils.meters import ScalarMeter, flush_scalar_meters
from utils.common import get_params_by_name
from utils.common import unwrap_state_dict
from utils.common import set_random_seed
from utils.common import create_exp_dir
from utils.common import setup_logging
from utils.common import save_status
from utils.common import get_device
from utils.common import extract_item
from utils import dataflow
from utils import optim
from utils import distributed as udist
from utils import prune

import models.mobilenet_base as mb

summary_writer = None
is_root_rank = None

NUM_IMAGENET_TRAIN = 1281167


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


def shrink_model(model_wrapper,
                 ema,
                 optimizer,
                 prune_info,
                 threshold=1e-3,
                 ema_only=False):
    r"""Dynamic network shrinkage to discard dead atomic blocks.

    Args:
        model_wrapper: model to be shrinked.
        ema: An instance of `ExponentialMovingAverage`, could be None.
        optimizer: Global optimizer.
        prune_info: An instance of `PruneInfo`, could be None.
        threshold: A small enough constant.
        ema_only: If `True`, regard an atomic block as dead only when
            `$$\hat{alpha} \le threshold$$`. Otherwise use both current value
            and momentum version.
    """
    model = unwrap_model(model_wrapper)
    for block_name, block in model.get_named_block_list().items():
        assert isinstance(block, mb.InvertedResidualChannels)
        masks = [
            bn.weight.detach().abs() > threshold
            for bn in block.get_depthwise_bn()
        ]
        if ema is not None:
            masks_ema = [
                ema.average('{}.{}.weight'.format(
                    block_name, name)).detach().abs() > threshold
                for name in block.get_named_depthwise_bn().keys()
            ]
            if not ema_only:
                masks = [
                    mask0 | mask1 for mask0, mask1 in zip(masks, masks_ema)
                ]
            else:
                masks = masks_ema
        block.compress_by_mask(masks,
                               ema=ema,
                               optimizer=optimizer,
                               prune_info=prune_info,
                               prefix=block_name,
                               verbose=False)

    if optimizer is not None:
        assert set(optimizer.param_groups[0]['params']) == set(
            model.parameters())

    model_profiling(model,
                    FLAGS.image_size,
                    FLAGS.image_size,
                    num_forwards=0,
                    verbose=False)
    logging.info('Model Shrink to FLOPS: {}'.format(model.n_macs))
    logging.info('Current model: {}'.format(mb.output_network(model)))


def unwrap_model(model_wrapper):
    """Remove model's wrapper."""
    model = model_wrapper.module
    return model


def get_ema_model(ema, model_wrapper):
    """Generate model from ExponentialMovingAverage.

    NOTE: If `ema` is given, generate a new model wrapper. Otherwise directly
        return `model_wrapper`, in this case modifying `model_wrapper` also
        influence the following process.
    FIXME: Always return a new model wrapper.
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


def get_prune_weights(model):
    """Get variables for pruning."""
    return get_params_by_name(unwrap_model(model), FLAGS._bn_to_prune.weight)


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


def get_meters(phase):
    """Util function for meters."""
    meters = {}
    meters['loss'] = ScalarMeter('{}_loss'.format(phase))
    for k in FLAGS.topk:
        meters['top{}_error'.format(k)] = ScalarMeter('{}_top{}_error'.format(
            phase, k))
    return meters


def profiling(model, use_cuda):
    """Profiling on either gpu or cpu."""
    logging.info('Start model profiling, use_cuda:{}.'.format(use_cuda))
    model_profiling(model,
                    FLAGS.image_size,
                    FLAGS.image_size,
                    verbose=getattr(FLAGS, 'model_profiling_verbose', True) and
                    is_root_rank)


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


def bn_calibration(m, cumulative_bn_stats=True):
    """Recalculate BN's running statistics.

    Should be called like `model.apply(bn_calibration)`.
    Args:
        m: sub_module to dealt with.
        cumulative_bn_stats: `True` to usage arithmetic mean instead of EMA.
    """
    if isinstance(m, torch.nn.BatchNorm2d):
        m.reset_running_stats()
        m.train()
        if cumulative_bn_stats:
            m.momentum = None


@udist.master_only
def summary_bn(model, prefix):
    """Summary BN's weights."""
    weights = get_prune_weights(model)
    for name, param in zip(FLAGS._bn_to_prune.weight, weights):
        summary_writer.add_histogram(
            '{}/{}/{}'.format(prefix, 'bn_scale', name), param.detach(),
            FLAGS._global_step)
    if len(FLAGS._bn_to_prune.weight) > 0:
        summary_writer.add_histogram(
            '{}/bn_scale/all'.format(prefix),
            torch.cat([weight.detach() for weight in weights]),
            FLAGS._global_step)


@udist.master_only
def log_pruned_info(model, flops_pruned, infos, prune_threshold):
    """Log pruning-related information."""
    if is_root_rank:
        logging.info('Flops threshold: {}'.format(prune_threshold))
        for info in infos:
            if FLAGS.prune_params['logging_verbose']:
                logging.info(
                    'layer {}, total channel: {}, pruned channel: {}, flops total:'
                    ' {}, flops pruned: {}, pruned rate: {:.3f}'.format(*info))
            summary_writer.add_scalar(
                'prune_ratio/{}/{}'.format(prune_threshold, info[0]), info[-1],
                FLAGS._global_step)
        logging.info('Pruned model: {}'.format(
            prune.output_searched_network(model, infos, FLAGS.prune_params)))

    flops_remain = model.n_macs - flops_pruned
    if is_root_rank:
        logging.info(
            'Prune threshold: {}, flops pruned: {}, flops remain: {}'.format(
                prune_threshold, flops_pruned, flops_remain))
        summary_writer.add_scalar('prune/flops/{}'.format(prune_threshold),
                                  flops_remain, FLAGS._global_step)


def run_one_epoch(epoch,
                  loader,
                  model,
                  criterion,
                  optimizer,
                  lr_scheduler,
                  ema,
                  rho_scheduler,
                  meters,
                  max_iter=None,
                  phase='train'):
    """Run one epoch."""
    assert phase in [
        'train', 'val', 'test', 'bn_calibration'
    ] or phase.startswith(
        'prune'), "phase not be in train/val/test/bn_calibration/prune."
    train = phase == 'train'
    if train:
        model.train()
    else:
        model.eval()
    if phase == 'bn_calibration':
        model.apply(bn_calibration)

    if FLAGS.use_distributed:
        loader.sampler.set_epoch(epoch)

    data_iterator = iter(loader)
    if FLAGS.use_distributed:
        data_fetcher = dataflow.DataPrefetcher(data_iterator)
    else:
        # TODO: prefetch for non distributed
        logging.warning('Not use prefetcher')
        data_fetcher = data_iterator
    for batch_idx, (input, target) in enumerate(data_fetcher):
        # used for bn calibration
        if max_iter is not None:
            assert phase == 'bn_calibration'
            if batch_idx >= max_iter:
                break

        target = target.cuda(non_blocking=True)
        if train:
            optimizer.zero_grad()
            rho = rho_scheduler(FLAGS._global_step)

            loss = forward_loss(model, criterion, input, target, meters)
            loss_l2 = optim.cal_l2_loss(model, FLAGS.weight_decay,
                                        FLAGS.weight_decay_method)
            loss_bn_l1 = prune.cal_bn_l1_loss(get_prune_weights(model),
                                              FLAGS._bn_to_prune.penalty, rho)
            loss = loss + loss_l2 + loss_bn_l1
            loss.backward()
            if FLAGS.use_distributed:
                udist.allreduce_grads(model)

            if FLAGS._global_step % FLAGS.log_interval == 0:
                results = reduce_and_flush_meters(meters)
                if is_root_rank:
                    logging.info('Epoch {}/{} Iter {}/{} {}: '.format(
                        epoch, FLAGS.num_epochs, batch_idx, len(loader), phase)
                                 + ', '.join('{}: {:.4f}'.format(k, v)
                                             for k, v in results.items()))
                    for k, v in results.items():
                        summary_writer.add_scalar('{}/{}'.format(phase, k), v,
                                                  FLAGS._global_step)

            if is_root_rank and FLAGS._global_step % FLAGS.log_interval == 0:
                summary_writer.add_scalar('train/learning_rate',
                                          optimizer.param_groups[0]['lr'],
                                          FLAGS._global_step)
                summary_writer.add_scalar('train/l2_regularize_loss',
                                          extract_item(loss_l2),
                                          FLAGS._global_step)
                summary_writer.add_scalar('train/bn_l1_loss',
                                          extract_item(loss_bn_l1),
                                          FLAGS._global_step)
                summary_writer.add_scalar('prune/rho', rho, FLAGS._global_step)
                summary_writer.add_scalar(
                    'train/current_epoch',
                    FLAGS._global_step / FLAGS._steps_per_epoch,
                    FLAGS._global_step)
                if FLAGS.data_loader_workers > 0:
                    summary_writer.add_scalar('data/train/prefetch_size',
                                              data_iterator.data_queue.qsize(),
                                              FLAGS._global_step)

            if is_root_rank and FLAGS._global_step % FLAGS.log_interval_detail == 0:
                summary_bn(model, 'train')

            optimizer.step()
            lr_scheduler.step()
            if FLAGS.use_distributed and FLAGS.allreduce_bn:
                udist.allreduce_bn(model)
            FLAGS._global_step += 1

            # NOTE: after steps count upate
            if ema is not None:
                model_unwrap = unwrap_model(model)
                ema_names = ema.average_names()
                params = get_params_by_name(model_unwrap, ema_names)
                for name, param in zip(ema_names, params):
                    ema(name, param, FLAGS._global_step)
        else:
            forward_loss(model, criterion, input, target, meters)

    if not train:
        results = reduce_and_flush_meters(meters)
        if is_root_rank:
            logging.info(
                'Epoch {}/{} {}: '.format(epoch, FLAGS.num_epochs, phase) +
                ', '.join(
                    '{}: {:.4f}'.format(k, v) for k, v in results.items()))
            for k, v in results.items():
                summary_writer.add_scalar('{}/{}'.format(phase, k), v,
                                          FLAGS._global_step)
    return results


def train_val_test():
    """Train and val."""
    torch.backends.cudnn.benchmark = True

    # model
    model, model_wrapper = get_model()
    criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    criterion_smooth = optim.CrossEntropyLabelSmooth(
        FLAGS.model_kwparams['num_classes'],
        FLAGS['label_smoothing'],
        reduction='none').cuda()
    # TODO: cal loss on all GPUs instead only `cuda:0` when non
    # distributed

    ema = None
    if FLAGS.moving_average_decay > 0.0:
        if FLAGS.moving_average_decay_adjust:
            moving_average_decay = optim.ExponentialMovingAverage.adjust_momentum(
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

    if FLAGS.get('log_graph_only', False):
        if is_root_rank:
            _input = torch.zeros(1, 3, FLAGS.image_size,
                                 FLAGS.image_size).cuda()
            _input = _input.requires_grad_(True)
            summary_writer.add_graph(model_wrapper, (_input,), verbose=True)
        return

    # check pretrained
    if FLAGS.pretrained:
        checkpoint = torch.load(FLAGS.pretrained,
                                map_location=lambda storage, loc: storage)
        if ema:
            ema.load_state_dict(checkpoint['ema'])
            ema.to(get_device(model))
        # update keys from external models
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        if (hasattr(FLAGS, 'pretrained_model_remap_keys') and
                FLAGS.pretrained_model_remap_keys):
            new_checkpoint = {}
            new_keys = list(model_wrapper.state_dict().keys())
            old_keys = list(checkpoint.keys())
            for key_new, key_old in zip(new_keys, old_keys):
                new_checkpoint[key_new] = checkpoint[key_old]
                logging.info('remap {} to {}'.format(key_new, key_old))
            checkpoint = new_checkpoint
        model_wrapper.load_state_dict(checkpoint)
        logging.info('Loaded model {}.'.format(FLAGS.pretrained))
    optimizer = optim.get_optimizer(model_wrapper, FLAGS)

    # check resume training
    if FLAGS.resume:
        checkpoint = torch.load(os.path.join(FLAGS.resume,
                                             'latest_checkpoint.pt'),
                                map_location=lambda storage, loc: storage)
        model_wrapper.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if ema:
            ema.load_state_dict(checkpoint['ema'])
            ema.to(get_device(model))
        last_epoch = checkpoint['last_epoch']
        lr_scheduler = optim.get_lr_scheduler(optimizer, FLAGS)
        lr_scheduler.last_epoch = (last_epoch + 1) * FLAGS._steps_per_epoch
        best_val = extract_item(checkpoint['best_val'])
        train_meters, val_meters = checkpoint['meters']
        FLAGS._global_step = (last_epoch + 1) * FLAGS._steps_per_epoch
        if is_root_rank:
            logging.info('Loaded checkpoint {} at epoch {}.'.format(
                FLAGS.resume, last_epoch))
    else:
        lr_scheduler = optim.get_lr_scheduler(optimizer, FLAGS)
        # last_epoch = lr_scheduler.last_epoch
        last_epoch = -1
        best_val = 1.
        train_meters = get_meters('train')
        val_meters = get_meters('val')
        FLAGS._global_step = 0

    if not FLAGS.resume and is_root_rank:
        logging.info(model_wrapper)
    assert FLAGS.profiling, '`m.macs` is used for calculating penalty'
    if FLAGS.profiling:
        if 'gpu' in FLAGS.profiling:
            profiling(model, use_cuda=True)
        if 'cpu' in FLAGS.profiling:
            profiling(model, use_cuda=False)

    # data
    (train_transforms, val_transforms,
     test_transforms) = dataflow.data_transforms(FLAGS)
    (train_set, val_set, test_set) = dataflow.dataset(train_transforms,
                                                      val_transforms,
                                                      test_transforms, FLAGS)
    (train_loader, calib_loader, val_loader,
     test_loader) = dataflow.data_loader(train_set, val_set, test_set, FLAGS)

    # get bn's weights
    FLAGS._bn_to_prune = prune.get_bn_to_prune(model, FLAGS.prune_params)
    rho_scheduler = prune.get_rho_scheduler(FLAGS.prune_params,
                                            FLAGS._steps_per_epoch)

    if FLAGS.test_only and (test_loader is not None):
        if is_root_rank:
            logging.info('Start testing.')
        test_meters = get_meters('test')
        validate(last_epoch, calib_loader, test_loader, criterion, test_meters,
                 model_wrapper, ema, 'test')
        return

    # already broadcast by AllReduceDistributedDataParallel
    # optimizer load same checkpoint/same initialization

    if is_root_rank:
        logging.info('Start training.')

    for epoch in range(last_epoch + 1, FLAGS.num_epochs):
        # train
        results = run_one_epoch(epoch,
                                train_loader,
                                model_wrapper,
                                criterion_smooth,
                                optimizer,
                                lr_scheduler,
                                ema,
                                rho_scheduler,
                                train_meters,
                                phase='train')

        # val
        results, model_eval_wrapper = validate(epoch, calib_loader, val_loader,
                                               criterion, val_meters,
                                               model_wrapper, ema, 'val')

        if FLAGS.prune_params['method'] is not None:
            prune_threshold = FLAGS.model_shrink_threshold
            masks = prune.cal_mask_network_slimming_by_threshold(
                get_prune_weights(model_eval_wrapper), prune_threshold)
            FLAGS._bn_to_prune.add_info_list('mask', masks)
            flops_pruned, infos = prune.cal_pruned_flops(FLAGS._bn_to_prune)
            log_pruned_info(unwrap_model(model_eval_wrapper), flops_pruned,
                            infos, prune_threshold)
            if flops_pruned >= FLAGS.model_shrink_delta_flops \
                    or epoch == FLAGS.num_epochs - 1:
                ema_only = (epoch == FLAGS.num_epochs - 1)
                shrink_model(model_wrapper, ema, optimizer, FLAGS._bn_to_prune,
                             prune_threshold, ema_only)
        model_kwparams = mb.output_network(unwrap_model(model_wrapper))

        if results['top1_error'] < best_val:
            best_val = results['top1_error']

            if is_root_rank:
                save_status(model_wrapper, model_kwparams, optimizer, ema,
                            epoch, best_val, (train_meters, val_meters),
                            os.path.join(FLAGS.log_dir, 'best_model'))
                logging.info(
                    'New best validation top1 error: {:.4f}'.format(best_val))

        if is_root_rank:
            # save latest checkpoint
            save_status(model_wrapper, model_kwparams, optimizer, ema, epoch,
                        best_val, (train_meters, val_meters),
                        os.path.join(FLAGS.log_dir, 'latest_checkpoint'))

        # NOTE: from scheduler code, should be called after train/val
        # use stepwise scheduler instead
        # lr_scheduler.step()
    return


def validate(epoch, calib_loader, val_loader, criterion, val_meters,
             model_wrapper, ema, phase):
    """Calibrate and validate."""
    assert phase in ['test', 'val']
    model_eval_wrapper = get_ema_model(ema, model_wrapper)

    # bn_calibration
    if FLAGS.get('bn_calibration', False):
        if not FLAGS.use_distributed:
            logging.warning(
                'Only GPU0 is used when calibration when use DataParallel')
        with torch.no_grad():
            _ = run_one_epoch(epoch,
                              calib_loader,
                              model_eval_wrapper,
                              criterion,
                              None,
                              None,
                              None,
                              None,
                              val_meters,
                              max_iter=FLAGS.bn_calibration_steps,
                              phase='bn_calibration')
        if FLAGS.use_distributed:
            udist.allreduce_bn(model_eval_wrapper)

    # val
    with torch.no_grad():
        results = run_one_epoch(epoch,
                                val_loader,
                                model_eval_wrapper,
                                criterion,
                                None,
                                None,
                                None,
                                None,
                                val_meters,
                                phase=phase)
    summary_bn(model_eval_wrapper, phase)
    return results, model_eval_wrapper


def main():
    """Entry."""
    # init distributed
    global is_root_rank
    if FLAGS.use_distributed:
        udist.init_dist()
        FLAGS.batch_size = udist.get_world_size() * FLAGS.per_gpu_batch_size
        FLAGS._loader_batch_size = FLAGS.per_gpu_batch_size
        if FLAGS.bn_calibration:
            FLAGS._loader_batch_size_calib = FLAGS.bn_calibration_per_gpu_batch_size
        FLAGS.data_loader_workers = round(FLAGS.data_loader_workers /
                                          udist.get_local_size())
        is_root_rank = udist.is_master()
    else:
        count = torch.cuda.device_count()
        FLAGS.batch_size = count * FLAGS.per_gpu_batch_size
        FLAGS._loader_batch_size = FLAGS.batch_size
        if FLAGS.bn_calibration:
            FLAGS._loader_batch_size_calib = FLAGS.bn_calibration_per_gpu_batch_size * count
        is_root_rank = True
    FLAGS.lr = FLAGS.base_lr * (FLAGS.batch_size / FLAGS.base_total_batch)
    # NOTE: don't drop last batch, thus must use ceil, otherwise learning rate
    # will be negative
    FLAGS._steps_per_epoch = int(np.ceil(NUM_IMAGENET_TRAIN / FLAGS.batch_size))

    if is_root_rank:
        FLAGS.log_dir = '{}/{}'.format(FLAGS.log_dir,
                                       time.strftime("%Y%m%d-%H%M%S"))
        create_exp_dir(
            FLAGS.log_dir,
            FLAGS.config_path,
            blacklist_dirs=[
                'exp',
                '.git',
                'pretrained',
                'tmp',
                'deprecated',
                'bak',
            ],
        )
        setup_logging(FLAGS.log_dir)
        for k, v in _ENV_EXPAND.items():
            logging.info('Env var expand: {} to {}'.format(k, v))
        logging.info(FLAGS)

    set_random_seed(FLAGS.get('random_seed', 0))
    with SummaryWriterManager():
        train_val_test()


if __name__ == "__main__":
    main()
