import pytest
import torch
import numpy as np

from torchmtlr.utils import encode_survival, make_time_bins


bins = torch.arange(1, 5, dtype=torch.float)
testdata = [
    (3., 1, torch.tensor([0, 0, 0, 1, 0]), bins),
    (2., 0, torch.tensor([0, 0, 1, 1, 1]), bins),
    (0., 1, torch.tensor([1, 0, 0, 0, 0]), bins),
    (6., 1, torch.tensor([0, 0, 0, 0, 1]), bins),
    (2., False, torch.tensor([0, 0, 1, 1, 1]), bins),
    (torch.tensor([3., 2.]), torch.tensor([1, 0]),
     torch.tensor([[0, 0, 0, 1, 0], [0, 0, 1, 1, 1]]), bins),
    (np.array([3., 2.]), np.array([1, 0]),
     torch.tensor([[0, 0, 0, 1, 0], [0, 0, 1, 1, 1]]), bins.numpy())
]
@pytest.mark.parametrize("time,event,expected,bins", testdata)
def test_encode_survival(time, event, expected, bins):
    encoded = encode_survival(time, event, bins)
    assert torch.all(encoded == expected)



