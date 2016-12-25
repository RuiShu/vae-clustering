# VAE-Clustering
A collection of experiments that shines light on VAE (containing discrete latent variables) as a clustering algorithm.

We evaluate the unsupervised clustering performance of three closely-related sets of deep generative models:

1. Kingma's [M2 model](https://arxiv.org/abs/1406.5298)
2. A modified-M2 model that implicitly contains a non-degenerate Gaussian mixture latent layer
3. An explicit Gaussian Mixture VAE model

Details about the three models and why to compare them are provided in [this blog post](http://ruishu.io/2016/12/25/gmvae/).

## Results

![](/images/combined.png)

M2 performs poorly as an unsupervised clustering algorithm. We suspect this is attributable to conflicting wishes to use the categorical variable as part of the generative model versus the inference model. By implicitly enforcing the a hidden layer to have a proper Gaussian mixture distribution, the modified-M2 model tips the scale in favor of using the categorical variable as part of the generative model. By using an explicit Gaussian Mixture VAE model, we achieve enable better inference, which leads to higher stability during training and even a stronger incentive to use the categorical variable in the generative model.


## Code set-up

The experiments are implemented using TensorFlow. Since all of the three aforementioned models share very similar formulations, the shared subgraphs are placed in `shared_subgraphs.py`. The `utils.py` file contains some additional functions used during training. The remaining `*.py` files simply implement the three main model classes and other variants that we tried.

We recommend first reading the Jupyter Notebook on [nbviewer](http://nbviewer.jupyter.org/github/RuiShu/vae-clustering/blob/master/experiments.ipynb) in the Chrome browser.

### Dependencies

1. tensorflow
2. tensorbayes
3. numpy
4. scipy
