import op
from node import Node
import utils
def gradients(end_node,node_list):

    grad_list={}
    grad_list[end_node]=[op.ones_like(end_node)]
    node_to_grad={}
    topo_list=utils.topo_sort_list([end_node])

    
    for x in reversed(topo_list):
        if True:
            grad = utils.sum_list(grad_list[x])
            node_to_grad[x]=grad
            for i in range(len(x.parents)):
                grads=x.op.gradient(x,grad)
                if x.parents[i] not in grad_list:
                    grad_list[x.parents[i]]=[]
                # print(grads[i].name)
                grad_list[x.parents[i]].append(grads[i])
    for node in node_list:
        print(node.name)
    print("===")
    for node in node_to_grad:
        print(node.name)
    return [node_to_grad[n] for n in node_list]
