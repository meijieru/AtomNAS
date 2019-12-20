# AtomNAS: Fine-Grained End-to-End Neural Architecture Search

This is the codebase (including search) for ICLR 2020 paper [AtomNAS: Fine-Grained End-to-End Neural Architecture Search](https://openreview.net/forum?id=BylQSxHFwr).

The network configs for AtomNAS-A/B/C could be checked at `apps/searched/models`, where a list within `inverted_residual_setting` corresponce to `[output_channel, num_repeat, stride, kernel_sizes, hidden_dims, has_first_pointwise]`.


## Setup

### Distributed Training

Set the following ENV variable:
```
$METIS_WORKER_0_HOST: IP address of worker 0
$METIS_WORKER_0_PORT: Port used for initializing distributed environment
$METIS_TASK_INDEX: Index of task
$REMOTE_WORKER_NUM: Number of workers
$REMOTE_WORKER_GPU: Number of GPUs (NOTE: should exactly match local GPU numbers with `CUDA_VISIBLE_DEVICES `)
$REMOTE_OUTPUT: Output directory
```

### Non-Distributed Training (Not Recommend)

Set the following ENV variable:
```
$REMOTE_WORKER_GPU: Number of GPUs (NOTE: should exactly match local GPU numbers with `CUDA_VISIBLE_DEVICES `)
$REMOTE_OUTPUT: Output directory
```


## Reproduce AtomNAS results

For Table 1

- AtomNAS-A: `bash scripts/run.sh apps/slimming/shrink/atomnas_a.yml`
- AtomNAS-B: `bash scripts/run.sh apps/slimming/shrink/atomnas_b.yml`
- AtomNAS-C: `bash scripts/run.sh apps/slimming/shrink/atomnas_c.yml`

If everything is OK, you should get similar results.


## Related Info

1. Requirements
    - python3, pytorch 1.1+, torchvision 0.3+, pyyaml 3.13, lmdb, pyarrow, pillow (pillow-simd recommanded).
    - Prepare ImageNet data following pytorch [example](https://github.com/pytorch/examples/tree/master/imagenet).
    - Optional: Generate lmdb dataset by `utils/lmdb_dataset.py`.

1. Miscellaneous
    - The codebase is a general ImageNet training framework using yaml config with several extension under `apps` dir, based on PyTorch.
        - Support `${ENV}` in yaml config.
        - Support `_include` for hierachy config.
        - Support `_default` key for overloading.
        - Support `xxx.yyy.zzz` for partial overloading.
    - Command: `bash scripts/run.sh {{path_to_yaml_config}}`.


## Acknowledgment

This repo benefit from the following projects
- [slimmable_networks](https://github.com/JiahuiYu/slimmable_networks)
- [apex](https://github.com/NVIDIA/apex)
- [Efficient-PyTorch](https://github.com/Lyken17/Efficient-PyTorch)

Thanks the contributors of these repos!
