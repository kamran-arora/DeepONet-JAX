import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
from nn.fnn import FNN


class DeepONet(eqx.Module):
    branch_net: FNN
    trunk_net: FNN
    bias: jax.Array

    def __init__(self, layer_sizes_branch, layers_sizes_trunk, activation, key, dtype):
        # branch and trunk keys
        branch_key, trunk_key = jr.split(key, 2)
        # activations
        if isinstance(activation, dict):
            branch_activation = activation["branch"]
            trunk_activation = activation["trunk"]
        else:
            branch_activation = trunk_activation = activation
        # create branch and trunk nets
        self.branch_net = FNN(layer_sizes_branch, branch_activation, branch_key, dtype)
        self.trunk_net = FNN(layers_sizes_trunk, trunk_activation, trunk_key, dtype)
        self.bias = jnp.zeros((1,))

    def __call__(self, x_branch, x_trunk):
        """
        x_branch.shape = (s, d_s)
        x_trunk.shape = (q, d_q)
        """
        x_branch = self.branch_net(x_branch)
        x_trunk = self.trunk_net(x_trunk)
        ip = jnp.sum(x_branch * x_trunk, keepdims=True)
        return (ip + self.bias)[0]
