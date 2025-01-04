import jax
import jax.random as jr
import equinox as eqx
from typing import Tuple, List, Callable, Union


class FNN(eqx.Module):
    layers: Tuple
    layer_sizes: List
    activation: Union[List[Callable], Callable]

    """
    Fully-connected Neural Network in Equinox
    """

    def __init__(self, layer_sizes, activation, key, dtype):
        self.layer_sizes = layer_sizes
        # activations
        if isinstance(activation, list):
            if not len(layer_sizes) - 1 == len(activation):
                raise ValueError(
                    f"Total number of activation functions does not match number of layers. Got {len(activation)}, expected {len(layer_sizes)-1}"
                )
            self.activation = activation
        else:
            self.activation = activation
        # keys for layers
        keys = jr.split(key, len(layer_sizes))
        # list of layers
        layers = []
        for i in range(1, len(layer_sizes)):
            layers.append(
                eqx.nn.Linear(
                    in_features=layer_sizes[i - 1],
                    out_features=layer_sizes[i],
                    dtype=dtype,
                    key=keys[i - 1],
                )
            )
        self.layers = tuple(layers)

    def __call__(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if isinstance(self.activation, list):
                layer_activation = self.activation[i]
            else:
                layer_activation = self.activation
            x = layer_activation(x)
        x = self.layers[-1](x)
        return x
