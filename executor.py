import utils
from node import *
class Executor:
    def __init__(self,node_list):
        self.node_list=node_list
    def run(self,feed_dict):
        topo_list=utils.topo_sort_list(self.node_list)
        for x in topo_list:
            x=reversed(x)
            for node in x:
                if node.op:
                    input_vals=[feed_dict[n] for n in node.parents]
                    feed_dict[node]=node.op.compute(node,input_vals)
        return [feed_dict[node] for node in self.node_list]
