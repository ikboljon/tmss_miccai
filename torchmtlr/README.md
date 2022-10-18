# Multi-task logistic regression for individual survival prediction in PyTorch
Lightweight PyTorch implementation of MTLR for survival prediction.


This package provides an `MTLR` class that can be used just like any other PyTorch module, an implementation of the log likelihood function for training and some handy utility functions. The aims are simplicity and compatibility with the PyTorch ecosystem. 

## Quickstart example
```python
import torch
import torch.nn as nn

from torchmtlr import (MTLR, mtlr_neg_log_likelihood,
                       mtlr_survival, mtlr_survival_at_times)
from torchmtlr.utils import encode_survival, make_time_bins


time = torch.tensor([.5, 1.1, 10.4, 2.3, 3.1])           # time to event for each sample
event = torch.tensor([0, 1, 1, 0, 1], dtype=torch.float) # event indicator for reach sample (0 = censored)
x = torch.randn((5, 10))                                 # features

time_bins = make_time_bins(time, event=event)
target = encode_survival(time, event, time_bins)

model = MTLR(x.shape[1], len(time_bins))

# forward pass
logits = model(x)

# compute minibatch loss
loss = mtlr_neg_log_likelihood(logits, target, model, C1=1., average=True)

with torch.no_grad():
  # predict survival curves at training timepoints
  survival = mtlr_survival(logits)

  # ...or at arbitrary times
  new_times = torch.tensor([1., 1.5, 2.])
  survival = mtlr_survival_at_times(logits, time_bins, new_times)


# use just like any other PyTorch module
model = nn.Sequential(
    nn.Linear(x.shape[1], 64),
    nn.ReLU(inplace=True),
    MTLR(64, len(time_bins))
)
```

See the [notebooks](notebooks) for more usage examples.


## Installation

_(Note: PyPI package coming soon!)_

1. Clone or download the repo.
2. Install the required packages:
```
pip install -r requirements.txt
```
Note: by default, the CPU version of Pytorch is installed. If you want to use a GPU, you need to [install CUDA and Pytorch with GPU support](https://pytorch.org/get-started/locally/).
3. Install `torchmtlr`:
```
pip install -e .
```

## Citation
If you found the package useful for your publication and want to cite it, you can use the following BibTeX entry:

```
@misc{kazmierski2020torchmtlr,
  author = {Kazmierski, Michal},
  title = {torchmtlr: flexible and modular implementation of multi-task logistic regression in PyTorch.},
  year  = {2020},
  url   = {https://github.com/mkazmier/torchmtlr},
}
```


## References
1. C.-N. Yu, R. Greiner, H.-C. Lin, and V. Baracos, ‘Learning patient-specific cancer survival distributions as a sequence of dependent regressors’, in Advances in neural information processing systems 24, pp. 1845–1853.
2. P. Jin, ‘Using Survival Prediction Techniques to Learn Consumer-Specific Reservation Price Distributions’, University of Alberta, Edmonton, AB, 2015.
3. S. Fotso, ‘Deep Neural Networks for Survival Analysis Based on a Multi-Task Framework’, arXiv:1801.05512 [cs, stat], Jan. 2018, Accessed: Feb. 11, 2020. [Online]. Available: http://arxiv.org/abs/1801.05512.
