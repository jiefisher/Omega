import op
from node import Node
import utils
def gradients(end_node,node_list):

    grad_list={}
    grad_list[end_node]=[op.ones_like(end_node)]
    node_to_grad={}
    topo_list=utils.topo_sort_list([end_node])

    
    for x in reversed(topo_list):
        grad = utils.sum_list(grad_list[x])
        node_to_grad[x]=grad
        for i in range(len(x.parents)):
            grads=x.op.gradient(x,grad)
            if x.parents[i] not in grad_list:
                grad_list[x.parents[i]]=[]
            # print(x.name)
            grad_list[x.parents[i]].append(grads[i])
        # print(node_list)
        for nd in node_list:
            
            if type(nd)==list:
                print(nd[0].name)
    # print("===")
    # for node in node_to_grad:
    #     if node!=None:
    #         print(node.name)
    grad_res=[]
    for n in node_list:
        if type(n)!=list:
            grad_res.append(node_to_grad[n])
        else:
            for k in n :
                if k in node_to_grad:
                    grad_res.append(node_to_grad[k])
    # return [node_to_grad[n] for n in node_list ]
    return grad_res
