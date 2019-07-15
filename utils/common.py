"""Common utilities."""
import functools
import numbers
import os
import sys
import shutil
import logging
import yaml
import torch
from packaging import version


def get_params_by_name(model, names):
    """Get params/buffers by name."""
    named_parameters = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    named_vars = {**named_parameters, **named_buffers}
    res = [named_vars[name] for name in names]
    return res


def unwrap_state_dict(checkpoint, verbose=True):
    """Remove `module.` prefix for `state_dict`."""
    with_wrapper = False
    new_checkpoint = {}
    for key, val in checkpoint.items():
        if with_wrapper:  # all keys consistent
            assert key.startswith('module.')
        if key.startswith('module.'):
            _, new_key = key.split('.', maxsplit=1)
            if not with_wrapper and verbose:
                logging.info('Unwrap state_dict')
                with_wrapper = True
        else:
            new_key = key
        new_checkpoint[new_key] = val
    return new_checkpoint


def set_random_seed(seed):
    """Set random seed."""
    import random
    import numpy as np

    logging.info('Set seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_exp_dir(path, config_path, scripts_dir=None, blacklist_dirs=[]):
    """Create output directory and backup files for reproducibility.

    Args:
        path: Path to output directory, directory will be setup.
        config_path: Path to config file. Must reside in `{root}/apps`.
        scripts_dir: Path to src scripts. Backup all python src files to
            `path/scripts`.
        blacklist_dirs: A list of paths to ignore when backup.
    """

    def ignore_func(cur_dir, contents, suffixs=None, blacklist_dirs_abs=[]):
        ignore_list = []
        for fpath in contents:
            fpath_abs = os.path.abspath(os.path.join(cur_dir, fpath))
            if not os.path.isdir(fpath_abs):
                _, ext = os.path.splitext(fpath)
                if ext not in suffixs:
                    ignore_list.append(fpath)
            else:
                if fpath_abs in blacklist_dirs_abs:
                    ignore_list.append(fpath)
        return ignore_list

    os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    config_dir = 'apps'
    dst_dir = os.path.join(path, 'scripts')
    scripts_dir = scripts_dir or os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..')
    blacklist_dirs_abs = [
        os.path.abspath(os.path.join(scripts_dir, val))
        for val in (blacklist_dirs + [config_dir])
    ]
    scripts_dir = os.path.abspath(scripts_dir)
    shutil.copytree(scripts_dir,
                    dst_dir,
                    ignore=functools.partial(
                        ignore_func,
                        suffixs=['.py', '.pyx'],
                        blacklist_dirs_abs=blacklist_dirs_abs))

    config_path = os.path.relpath(config_path, scripts_dir)
    if not config_path.startswith(config_dir):
        raise RuntimeError('Config files assume to live in `apps`')
    shutil.copytree(os.path.join(scripts_dir, config_dir),
                    os.path.join(path, config_dir),
                    ignore=functools.partial(ignore_func,
                                             suffixs=['.yml', '.yaml']))


def setup_logging(log_dir=None):
    """Initialize `logging` module."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.close()
    root_logger.handlers.clear()
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format=log_format,
                        datefmt='%m/%d %I:%M:%S %p')
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def save_status(model, model_kwparams, optimizer, ema, epoch, best_val, meters,
                checkpoint_name):
    """Create checkpoint."""
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'ema': ema.state_dict() if ema else None,
            'last_epoch': epoch,
            'best_val': best_val,
            'meters': meters,
        }, '{}.pt'.format(checkpoint_name))
    if model_kwparams is not None:
        with open('{}.yml'.format(checkpoint_name), 'w') as f:
            f.write(str(model_kwparams))


def get_device(x):
    """Find device given tensor or module.

    NOTE: assume all model parameters reside on the same devices.
    """
    if isinstance(x, torch.Tensor):
        return x.device
    elif isinstance(x, torch.nn.Module):
        return next(x.parameters()).device
    else:
        raise RuntimeError('{} do not have `device`'.format(type(x)))


def extract_item(x):
    """Extract value from single value tensor."""
    if isinstance(x, numbers.Number):
        return x
    elif isinstance(x, torch.Tensor):
        return x.item()
    else:
        raise ValueError('Unknown type: {}'.format(type(x)))


def add_prefix(name, prefix=None, split='.'):
    """Add prefix to name if given."""
    if prefix is not None:
        return '{}{}{}'.format(prefix, split, name)
    else:
        return name


def index_tensor_in(tensor, l, raise_error=True):
    """Find the index in list.

    Args:
        tensor: tensor to find.
        l: An iterable object.
        raise_error: If `True`, raise error if tensor is not in the list,
            otherwise return `None`.
    """
    for i, val in enumerate(l):
        if id(tensor) == id(val):
            return i
    if raise_error:
        raise ValueError('Tensor not in list')
    else:
        return None


def check_tensor_in(tensor, iterable):
    """Check whether the tensor is in the object.

    Args:
        tensor: tensor to check.
        l: An instance of `dict` or `list`.
    """
    if isinstance(iterable, dict):
        iterable = iterable.keys()
    elif isinstance(iterable, list):
        pass
    else:
        raise ValueError('Unknown iterable: {}'.format(type(iterable)))
    index = index_tensor_in(tensor, iterable, raise_error=False)
    return index is not None


def get_data_queue_size(data_iter):
    """Get prefetched size."""
    if version.parse(torch.__version__) < version.parse('1.3.0'):
        return data_iter.data_queue.qsize()
    else:
        return data_iter._data_queue.qsize()


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
