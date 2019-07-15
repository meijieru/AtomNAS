"""Modified from https://github.com/JiahuiYu/slimmable_networks/blob/master/utils/distributed.py"""
from collections import OrderedDict

import os
import functools

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel.scatter_gather import scatter_kwargs
from torch._utils import _flatten_dense_tensors
from torch._utils import _unflatten_dense_tensors
from torch._utils import _take_tensors

from torch.distributed import get_rank
from torch.distributed import get_world_size


def _get_env(env_name):
    if env_name not in os.environ:
        raise RuntimeError('${} should be set'.format(env_name))
    return os.environ[env_name]


def init_dist(backend='nccl', **kwargs):
    if dist.is_initialized():
        raise RuntimeError('Should not init distributed twice')
    rank = int(_get_env('RANK'))
    local_rank = int(_get_env('LOCAL_RANK'))
    assert rank % torch.cuda.device_count() == local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, **kwargs)


def assert_initialized():
    if not dist.is_initialized():
        raise RuntimeError('Default process group is not initialized')


def get_local_rank():
    assert_initialized()
    return int(_get_env('LOCAL_RANK'))


def get_local_size():
    assert_initialized()
    return torch.cuda.device_count()


def is_master():
    """check if current process is the master"""
    return get_rank_fallback() == 0


def get_rank_fallback():
    if dist.is_initialized():
        rank = get_rank()
    else:
        rank = 0
    return rank


def get_world_size_fallback():
    if dist.is_initialized():
        world_size = get_world_size()
    else:
        world_size = 1
    return world_size


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_master():
            return func(*args, **kwargs)
        else:
            return None

    return wrapper


def dist_reduce_tensor(tensor, dst=0):
    """Reduce to specific rank"""
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.reduce(tensor, dst=dst)
        if get_rank() == dst:
            tensor.div_(world_size)
    return tensor


def dist_all_reduce_tensor(tensor):
    """Reduce to all ranks"""
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.all_reduce(tensor)
        tensor.div_(world_size)
    return tensor


def _get_coalesced_bucket(tensors, buffer_size_mb=-1):
    if buffer_size_mb > 0:
        buffer_size = buffer_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, buffer_size)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()
    return buckets


def _broadcast_coalesced(tensors, bucket_size_mb=-1):
    buckets = _get_coalesced_bucket(tensors, bucket_size_mb)
    for tensors in buckets:
        flat_tensors = _flatten_dense_tensors(tensors)
        dist.broadcast(flat_tensors, 0)
        for tensor, synced in zip(
                tensors, _unflatten_dense_tensors(flat_tensors, tensors)):
            tensor.copy_(synced)


def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
    buckets = _get_coalesced_bucket(tensors, bucket_size_mb)
    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)


def _allreduce(tensors, coalesce=True, bucket_size_mb=-1):
    world_size = get_world_size()
    if coalesce:
        _allreduce_coalesced(tensors, world_size, bucket_size_mb)
    else:
        handles = []
        for tensor in tensors:
            handle = dist.all_reduce(tensor.div_(world_size), async_op=True)
            handles.append(handle)
        for handle in handles:
            handle.wait()


def allreduce_grads(model, *args, **kwargs):
    grads = [
        param.grad.data
        for param in model.parameters()
        if param.requires_grad and param.grad is not None
    ]
    _allreduce(grads, *args, **kwargs)


def allreduce_bn(model, *args, **kwargs):
    tensors = []
    for name, buffer in model.named_buffers():
        if 'running_var' in name or 'running_mean' in name:
            tensors.append(buffer)
    _allreduce(tensors, *args, **kwargs)


class AllReduceDistributedDataParallel(nn.Module):

    def __init__(self, module, dim=0, broadcast_buffers=True, bucket_cap_mb=25):
        super(AllReduceDistributedDataParallel, self).__init__()
        self.module = module
        self.dim = dim
        self.broadcast_buffers = broadcast_buffers

        self.broadcast_bucket_size_mb = bucket_cap_mb
        self._sync_params()

    def _sync_params(self):
        module_states = list(self.module.state_dict().values())
        if len(module_states) > 0:
            _broadcast_coalesced(module_states, self.broadcast_bucket_size_mb)
        if self.broadcast_buffers:
            buffers = [b.data for b in self.module.buffers()]
            if len(buffers) > 0:
                _broadcast_coalesced(buffers, self.broadcast_bucket_size_mb)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        res = self.module(*inputs[0], **kwargs[0])
        return res
