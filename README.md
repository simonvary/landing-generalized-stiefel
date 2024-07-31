# Landing on the random generalized Stiefel manifold

This repository contains the code for the landing algorithm under the generalized Stiefel manifold constraint without the use of retractions. 

The algorithm is implemented as a PyTorch optimizer; see `solvers/landing_generalized_stiefel/optimizer.py`.

You can find the paper [here](https://arxiv.org/abs/2405.01702).

<img alt="Landing diagram" src="https://github.com/simonvary/landing-generalized-stiefel/blob/master/diagram.png?raw=true" width=50% height=50%>

## Figures

To reproduce the plots using the provided convergence data, you can use the `makefile` in the folder `figures/`.

## Cite

If you use this code please cite:
```

@InProceedings{Vary2024Optimization,
  title = 	 {Optimization without Retraction on the Random Generalized Stiefel Manifold},
  author =       {Vary, Simon and Ablin, Pierre and Gao, Bin and Absil, Pierre-Antoine},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {49226--49248},
  year = 	 {2024},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}

```
