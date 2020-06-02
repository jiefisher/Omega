def topo_sort(endNode):
    topo_list=[]
    stack=[endNode]
    count={}
    while stack:
        p=stack.pop()
        if p in count:
            count[p]+=1
        else:
            count[p]=1
            stack.extend(p.parents)
    temp=[endNode]
    while temp:
        p=temp.pop()
        topo_list.append(p)
        for parent in p.parents:
            if count[parent]==1:
                temp.append(parent)
            else:
                count[parent]-=1
    return topo_list
def topo_sort_list(node_list):
    topo_order=[]
    for x in node_list:
        y=topo_sort(x)
        topo_order.extend(y)
    return topo_order[::-1]
    # visited = set()
    # topo_order = []
    # for node in node_list:
    #     depth_first_search(node, visited, topo_order)
    # return topo_order

def depth_first_search(node, visited, topo_order):
    """

    :param node:
    :param visited:
    :param topo_order:
    :return:
    """
    if node in visited:
        return
    visited.add(node)
    for n in node.parents:
        depth_first_search(n, visited, topo_order)
    topo_order.append(node)

def sum_list(node_list):
    from operator import add
    from functools import reduce
    return reduce(add,node_list)