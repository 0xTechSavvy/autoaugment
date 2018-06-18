# AutoAugment

My attempt at reproduction of the following paper from Google. I have used Keras and TensorFlow.

* [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501). Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, Quoc V. Le.

There are two components to the code:

1. **Controller:** a recurrent neural network that suggests transformations
2. **Child:** the final neural network trained with the previous suggestion.

Each **child** is trained start-to-finish using the policies produced by the recurrent neural network (controller). The model is then evaluated in the validation set. The tuple (child validation accuracy score, controller softmax probabilities) are then stored in a list.

The **controller** is trained in order to maximize the derivative of its outputs with respect to each weight, $\frac{\partial y}{\partial w}$, times the [0,1] normalized accuracy scores from the previous list. The $y$ outputs are the "controller softmax probabilities" from the previous list.

All this is implemented in the `fit()` function which can be found inside each class.

*Disclaimer:* I am unsure whether the code resembles that of the authors. I have used a lot of information [from this other paper](https://arxiv.org/abs/1707.07012), which is the main citation from AutoAugment.
