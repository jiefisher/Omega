import numpy as np
import op
import activation
from op import OP

def log_sum_exp(x):
    mx = np.max(x, axis=1, keepdims=True)
    safe = x - mx
    return mx + np.log(np.sum(np.exp(safe), axis=1, keepdims=True))
class CrossEntropyOp(OP):
    def __call__(self, node_A, node_B):
        new_node = OP.__call__(self)
        new_node.parents = [node_A, node_B]
        new_node.name = 'CrossEntropy({0:s}, {1:s})'.format(node_A.name, node_B.name)
        return new_node

    def compute(self, node, vals):
        logits = vals[0]
        actual = vals[1]
        safe_log_softmax = logits - log_sum_exp(logits)
        return np.mean(-1*np.sum(actual * safe_log_softmax, axis=1), keepdims=True)

    def gradient(self, node, grad):
        grad_A = (activation.softmax(node.parents[0]) + -1 * node.parents[1]) * grad
        grad_B = op.zeros_like(node.parents[1])
        return [grad_A, grad_B]


cross_entropy = CrossEntropyOp()