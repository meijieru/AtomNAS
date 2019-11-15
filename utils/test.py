"""Util for unit test."""
import numpy as np
import torch


def assertAllClose(a, b, rtol=1e-6, atol=1e-6, msg=None):
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assertAllClose(x, y, rtol=rtol, atol=atol, msg=msg)
    elif isinstance(a, torch.Tensor):
        a = to_numpy(a)
        if isinstance(b, torch.Tensor):
            b = to_numpy(b)
        assertAllClose(a, b, rtol=rtol, atol=atol, msg=msg)
    else:
        msg = msg or ''
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def apply_gradients(gradients, variables):
    assert len(gradients) == len(variables)
    for grad, var in zip(gradients, variables):
        if var.grad is None:
            var.grad = grad
        else:
            var.grad.data.copy_(grad)
