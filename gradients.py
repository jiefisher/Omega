import op
from node import Node
import utils
def gradients(end_node,node_list):
    grad_list={}
    grad_list[end_node]=[op.ones_like(end_node)]
    node_to_grad={}
    topo_list=utils.topo_sort_list([end_node])
    print(type(topo_list[0]))
    for topo in topo_list:
        for x in topo:
            
            grad = utils.sum_list(grad_list[x])
            node_to_grad[x]=grad
            for i in range(len(x.parents)):
                print(x.name)
                grads=x.op.gradient(x,grad)
                if x.parents[i] not in grad_list:
                    grad_list[x.parents[i]]=[]
                grad_list[x.parents[i]].append(grads[i])
    return [node_to_grad[n] for n in node_list]
