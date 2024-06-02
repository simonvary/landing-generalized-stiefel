# Landing on the random generalized Stiefel manifold

This repository contains the code for the landing algorithm under the generalized Stiefel manifold constraint without the use of retractions. 

The algorithm is implemented as a PyTorch optimizer; see `solvers/landing_generalized_stiefel/optimizer.py`.

You can find the paper [here](https://arxiv.org/abs/2405.01702).

## Figures

To reproduce the plots using the provided convergence data, you can use the `makefile` in the folder `figures/`.

## Cite

If you use this code please cite:
```
@misc{Vary2024Optimization,
  title = {{Optimization without retraction on the random generalized Stiefel manifold}},
  author = {Vary, Simon and Ablin, Pierre and Gao, Bin and Absil, P.-A.},
  year = {2024},
  month = may,
  eprint = {arXiv:2405.01702}
}
```
