"""Validation."""
import logging
import time

import torch

from utils.config import FLAGS, _ENV_EXPAND
from utils.common import set_random_seed
from utils.common import setup_logging
from utils.common import get_device
from utils.common import bn_calibration
from utils import dataflow
from utils import distributed as udist

import common as mc


def run_one_epoch(epoch,
                  loader,
                  model,
                  criterion,
                  optimizer,
                  lr_scheduler,
                  ema,
                  meters,
                  max_iter=None,
                  phase='train'):
    """Run one epoch."""
    assert phase in ['val', 'test', 'bn_calibration'
                    ], "phase not be in val/test/bn_calibration."
    model.eval()
    if phase == 'bn_calibration':
        model.apply(bn_calibration)

    if FLAGS.use_distributed:
        loader.sampler.set_epoch(epoch)

    data_iterator = iter(loader)
    if FLAGS.use_distributed:
        data_fetcher = dataflow.DataPrefetcher(data_iterator)
    else:
        # TODO(meijieru): prefetch for non distributed
        logging.warning('Not use prefetcher')
        data_fetcher = data_iterator
    for batch_idx, (input, target) in enumerate(data_fetcher):
        # used for bn calibration
        if max_iter is not None:
            assert phase == 'bn_calibration'
            if batch_idx >= max_iter:
                break

        target = target.cuda(non_blocking=True)
        mc.forward_loss(model, criterion, input, target, meters)

    results = mc.reduce_and_flush_meters(meters)
    if udist.is_master():
        logging.info('Epoch {}/{} {}: '.format(epoch, FLAGS.num_epochs, phase)
                     + ', '.join(
                         '{}: {:.4f}'.format(k, v) for k, v in results.items()))
        for k, v in results.items():
            mc.summary_writer.add_scalar('{}/{}'.format(phase, k), v,
                                         FLAGS._global_step)
    return results


def val():
    """Validation."""
    torch.backends.cudnn.benchmark = True

    # model
    model, model_wrapper = mc.get_model()
    ema = mc.setup_ema(model)
    criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    # TODO(meijieru): cal loss on all GPUs instead only `cuda:0` when non
    # distributed

    # check pretrained
    if FLAGS.pretrained:
        checkpoint = torch.load(FLAGS.pretrained,
                                map_location=lambda storage, loc: storage)
        if ema:
            ema.load_state_dict(checkpoint['ema'])
            ema.to(get_device(model))
        model_wrapper.load_state_dict(checkpoint['model'])
        logging.info('Loaded model {}.'.format(FLAGS.pretrained))

    if udist.is_master():
        logging.info(model_wrapper)

    # data
    (train_transforms, val_transforms, test_transforms) = \
        dataflow.data_transforms(FLAGS)
    (train_set, val_set, test_set) = dataflow.dataset(train_transforms,
                                                      val_transforms,
                                                      test_transforms, FLAGS)
    _, calib_loader, _, test_loader = dataflow.data_loader(
        train_set, val_set, test_set, FLAGS)

    if udist.is_master():
        logging.info('Start testing.')
    FLAGS._global_step = 0
    test_meters = mc.get_meters('test')
    validate(0, calib_loader, test_loader, criterion, test_meters,
             model_wrapper, ema, 'test')
    return


# TODO(meijieru): move to common
def validate(epoch, calib_loader, val_loader, criterion, val_meters,
             model_wrapper, ema, phase):
    """Calibrate and validate."""
    assert phase in ['test', 'val']
    model_eval_wrapper = mc.get_ema_model(ema, model_wrapper)

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
                                val_meters,
                                phase=phase)
    return results


def main():
    """Entry."""
    FLAGS.test_only = True
    mc.setup_distributed()
    if udist.is_master():
        FLAGS.log_dir = '{}/{}'.format(FLAGS.log_dir,
                                       time.strftime("%Y%m%d-%H%M%S-eval"))
        setup_logging(FLAGS.log_dir)
        for k, v in _ENV_EXPAND.items():
            logging.info('Env var expand: {} to {}'.format(k, v))
        logging.info(FLAGS)

    set_random_seed(FLAGS.get('random_seed', 0))
    with mc.SummaryWriterManager():
        val()


if __name__ == "__main__":
    main()
