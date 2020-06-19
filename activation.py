import op
from op import OP
import numpy as np
class ReluOp(OP):
    def __call__(self, node_A):
        new_node = OP.__call__(self)
        new_node.parents = [node_A]
        new_node.name = "Relu(%s)" % (node_A.name)
        return new_node

    def compute(self, node,vals):
    
        return np.maximum(vals[0], 0)

    def gradient(self, node, grad):
        return [relu_grad(node.parents[0], grad)]

class ReluGradientOp(OP):
    def __call__(self, node_A, node_B):
        """node_B is output_grad"""
        new_node = OP.__call__(self)
        new_node.parents = [node_A, node_B]
        new_node.name = "ReluGradient(%s)" % (node_A.name)
        return new_node

    def compute(self, node, vals):
        return  np.sign(np.maximum(vals[0], 0)) * vals[1]


    def gradient(self, node, output_grad):
        raise NotImplementedError('Gradient of ReluGradientOp not implemented')

class SigmoidOp(OP):
    def __call__(self, node_A):
        new_node = OP.__call__(self)
        new_node.parents = [node_A]
        new_node.name = 'Sigmoid({0:s})'.format(node_A.name)
        return new_node

    def compute(self, node, vals):
        """
        tanh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))
                = 2*sigmoid(2*x) - 1
        sigmoid(x) = 0.5 + 0.5*tanh(0.5*x)
        """
        t=0.5 + 0.5*np.tanh(0.5*vals[0])
        print("sig",t.shape)
 
        return 0.5 + 0.5*np.tanh(0.5*vals[0])


    def gradient(self, node, grad):
        x = node.parents[0]
        # g = sigmoid(x) * (1 - sigmoid(x))
        # TODO: (upul) obove g failed in unit testing, need to check it.
        g = sigmoid(x) - sigmoid(x) * sigmoid(x)
        return [g * grad]


def softmax_func(x):
    stable_values = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(stable_values) / np.sum(np.exp(stable_values), axis=-1, keepdims=True)

class SoftmaxOp(OP):
    def __call__(self, node_A):
        new_node = OP.__call__(self)
        new_node.parents = [node_A]
        new_node.name = 'SoftmaxOp({0:s})'.format(node_A.name)
        return new_node

    def compute(self, node, vals):
        t= softmax_func(vals[0])
        return softmax_func(vals[0])


    def gradient(self, node, output_grads):
        return [softmax_grad(output_grads)]

class SoftmaxGradientOp(OP):
    def __call__(self, node_A):
        """node_B is output_grad"""
        new_node = OP.__call__(self)
        new_node.parents = [node_A]
        new_node.name = "SoftmaxGradient(%s)" % (node_A.name)
        return new_node

    def compute(self, node, vals):
        s = vals[0].reshape(-1,1)

        t = np.diagflat(s) - np.dot(s, s.T)
        origin_shape=vals[0].shape
        val = vals[0].reshape(1,-1)
        res= np.dot(val,t)
        res=res.reshape(origin_shape)
        print(t)
        return res


    def gradient(self, node, output_grad):
        raise NotImplementedError('Gradient of SoftmaxGradientOp not implemented')

relu = ReluOp()
sigmoid = SigmoidOp()
softmax = SoftmaxOp()
relu_grad = ReluGradientOp()
softmax_grad = SoftmaxGradientOp()