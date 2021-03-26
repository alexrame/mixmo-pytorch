# MixMo: Mixing Multiple Inputs for Multiple Outputs via Deep Subnetworks

[Alexandre Ramé](https://alexrame.github.io/),  [Rémy Sun](http://perso.eleves.ens-rennes.fr/people/remy.sun/), [Matthieu Cord](http://webia.lip6.fr/~cord/)

```
Work in progress!
```

![](./mixmo_intro.png)

## Citation

We will release soon the PyTorch implementation of our [paper](https://arxiv.org/abs/2103.06132):

```
@article{rame2021ixmo,
    title={MixMo: Mixing Multiple Inputs for Multiple Outputs via Deep Subnetworks},
    author={Alexandre Rame and Remy Sun and Matthieu Cord},
    year={2021},
    journal={arXiv preprint arXiv:2103.06132}
}
```
## Abstract
Recent strategies achieved ensembling “for free” by fitting concurrently diverse subnetworks inside a single base network. The main idea during training is that each subnetwork learns to classify only one of the multiple inputs simultaneously provided. However, the question of how to best mix these multiple inputs has not been studied so far.

In this paper, we introduce MixMo, a new generalized framework for learning multi-input multi-output deep subnetworks. Our key motivation is to replace the suboptimal summing operation hidden in previous approaches by a more appropriate mixing mechanism. For that purpose, we draw inspiration from successful mixed sample data augmentations. We show that binary mixing in features - particularly with rectangular patches from CutMix - enhances results by making subnetworks stronger and more diverse.

We improve state of the art for image classification on CIFAR-100 and Tiny ImageNet datasets. Our easy to implement models notably outperform data augmented deep ensembles, without the inference and memory overheads. As we operate in features and simply better leverage the expressiveness of large networks, we open a new line of research complementary to previous works.


## Acknowledgements
* Our implementation is based on the official repository of [Addressing Failure Prediction by Learning Model Confidence](https://github.com/valeoai/ConfidNet) ([![Apache 2 License](https://img.shields.io/badge/license-Apache%202-yellowgreen.svg)](https://github.com/valeoai/ConfidNet/blob/master/LICENSE)). Thus we thank [Charles Corbière](https://chcorbi.github.io/) for this great work

