import pytest
import torch
import numpy as np
from scipy.special import logsumexp

from torchmtlr import *


torch.manual_seed(42)


# MTLR module
testdata = [
    (torch.randn((2, 3)), 5), # representative case
    (torch.randn((1, 3)), 5), # one sample
    (torch.randn((2, 1)), 5), # one feature
    (torch.randn((2, 3)), 2), # one time bin + catch-all
]
@pytest.mark.parametrize("x,num_time_bins", testdata)
def test_mtlr_forward(x, num_time_bins):
    """Test `forward` method output shape."""
    mtlr = MTLR(x.size(1), num_time_bins)
    out = mtlr(x)
    expected = (x.size(0), num_time_bins+1)
    assert out.shape == expected


@pytest.mark.parametrize("x,num_time_bins", testdata)
def test_mtlr_grad(x, num_time_bins):
    """Test whether gradients are propagated correctly."""
    mtlr = MTLR(x.size(1), num_time_bins)
    x.requires_grad = True
    loss = mtlr(x).sum()
    loss.backward()
    for p in mtlr.parameters():
        assert p.grad is not None

    assert x.grad is not None


# Likelihood function
testlogits = torch.tensor([[.1, .2, .3], [.4, .5, .6]])
testdata = [
    (
        testlogits,
        torch.tensor([[0, 1, 0], [1, 0, 0]]),
        torch.tensor(2.3039)
    ), # no censoring
    (
        testlogits,
        torch.tensor([[0, 1, 1], [1, 1, 1]]),
        torch.tensor(.3575)
    ), # all censored
    (
        testlogits,
        torch.tensor([[0, 1, 0], [1, 1, 1]]),
        torch.tensor(1.1019)
    ), # censored + uncensored
]
@pytest.mark.parametrize("x,mask,ignore", testdata)
def test_masked_logsumexp(x, mask, ignore):
    expected = logsumexp(x.numpy(), axis=1, b=mask.numpy())
    assert np.allclose(masked_logsumexp(x, mask).numpy(), expected)


@pytest.mark.parametrize("logits,target,expected", testdata)
def test_mtlr_neg_log_likelihood(logits, target, expected):
    model = MTLR(2, logits.size(1)-1)
    nll = mtlr_neg_log_likelihood(logits, target, model, 0, False)
    assert torch.isclose(nll, expected, atol=1e-4)


@pytest.mark.parametrize("logits,target,expected", testdata)
def test_mtlr_neg_log_likelihood_reg(logits, target, expected):
    model = MTLR(2, logits.size(1)-1)
    torch.nn.init.constant_(model.mtlr_weight, 1.)
    nll = mtlr_neg_log_likelihood(logits, target, model, 2., False)
    assert torch.isclose(nll, expected + 2 * (logits.size(1) - 1), atol=1e-4)


# Survival prediction functions
testdata = [
    (torch.ones((1, 4)), torch.tensor([[1., .75, .5, .25]])),
    (torch.tensor([[1., 20., 1., 1.]]), torch.tensor([[1., 1., 0., 0.]])),
    (torch.tensor([[1., 1., 1., 20.]]), torch.ones((1, 4))),
]
@pytest.mark.parametrize("logits,expected", testdata)
def test_mtlr_survival(logits, expected):
    assert torch.allclose(mtlr_survival(logits), expected, atol=1e-6)


@pytest.mark.parametrize("logits,survival", testdata)
def test_mtlr_survival_at_times(logits, survival):
    train_times = np.arange(1, 4, dtype=np.float32)
    pred_times = np.array([.5, 1.5])
    expected = np.array([(survival[:, 0] + survival[:, 1]) / 2,
                         (survival[:, 1] + survival[:, 2]) / 2])
    assert np.allclose(mtlr_survival_at_times(logits, train_times, pred_times), expected)
