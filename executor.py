import utils
from node import *
class Executor:
    def __init__(self,node_list):
        self.node_list=node_list
    def run(self,feed_dict):
        topo_list=utils.topo_sort_list(self.node_list)
        for node in topo_list:
            if True:
                if node.op:
                    input_vals=[]
                    for n in node.parents:
                        if n in feed_dict:
                            input_vals.append(feed_dict[n])
                        else:
                            input_vals.append(n.const)
                    feed_dict[node]=node.op.compute(node,input_vals)
        return [feed_dict[node] for node in self.node_list]
