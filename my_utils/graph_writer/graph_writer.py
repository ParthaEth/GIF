import random
import networkx as nx
import matplotlib.pyplot as plt
import wrapt
from pyvis.network import Network


class ModuleSpace:
    def __init__(self, node_tracing_name=None):
        self.module_node_tracing_name = node_tracing_name

    def __enter__(self):
        com_grph.current_color = random.uniform(0, 1)

    def __exit__(self, type, value, traceback):
        com_grph.current_color = 0.5  # default color


class CallWrapper(wrapt.ObjectProxy):
    _name_dict = {}
    _trace_obj_cnt = 0

    def __init__(self, wrapped_obj, node_tracing_name=None):
        super(CallWrapper, self).__init__(wrapped_obj)
        CallWrapper._trace_obj_cnt += 1
        if node_tracing_name is None:
            obj_type_name = self.__wrapped__.__class__.__name__
        else:
            obj_type_name = node_tracing_name

        if obj_type_name in CallWrapper._name_dict:
            obj_count = CallWrapper._name_dict[obj_type_name]
        else:
            obj_count = 0
        CallWrapper._name_dict[obj_type_name] = obj_count + 1
        unique_node_tracing_name = f'{obj_type_name}_{obj_count}'

        self._self_node_tracing_name = unique_node_tracing_name
        com_grph.add_node(None, unique_node_tracing_name, color=com_grph.current_color)

    def __call__(self, *args, **kwargs):
        #print('Tracing')
        output = self.__wrapped__(*args, **kwargs)
        if CallWrapper._trace_obj_cnt > 0 and not com_grph.tracing_info_collected:
            for arg in args:
                self._node_trace(arg)

            for _, arg in kwargs.items():
                self._node_trace(arg)

            try:
                setattr(output, '_self_node_tracing_name', self._self_node_tracing_name)
            except AttributeError:
                pass

        if CallWrapper._trace_obj_cnt > 0:
            CallWrapper._trace_obj_cnt -= 1
        return output

    def _node_trace(self, arg):
        if hasattr(arg, '_self_node_tracing_name'):
            from_node_tracing_name = arg._self_node_tracing_name
            # arg.node_tracing_name = self.node_tracing_name
        else:
            from_node_tracing_name = getattr(arg, 'input_name', f'Input:{com_grph.input_count}')
            com_grph.add_node(None, from_node_tracing_name, color=com_grph.current_color)
            com_grph.input_count += 1

        if self._self_node_tracing_name not in nodes_connected:
            nodes_connected.append(self._self_node_tracing_name)

        com_grph.add_edge(from_node_tracing_name=from_node_tracing_name,
                          to_node_tracing_name=self._self_node_tracing_name)


class NetworkModel():
    def __init__(self):
        self.current_color = 0.5
        self.root_graph = nx.MultiDiGraph()
        self.input_count = 0
        self.tracing_info_collected = False

    def add_node(self, node_obj, node_tracing_name, **kwargs):
        self.root_graph.add_node(node_tracing_name, node_tracing_name=node_tracing_name, **kwargs)
        return node_obj

    def add_edge(self, from_node_tracing_name, to_node_tracing_name):
        self.root_graph.add_edge(from_node_tracing_name, to_node_tracing_name)

    def remove_node(self, node_name):
        self.root_graph.remove_node(node_name)


def terminate_tracing():
    com_grph.tracing_info_collected = True


def draw(net, file_name, canvas_size_100s_px=(16, 38), *args, **kwargs):
    # file_name set to None to leave it in matplot lib draw buffer
    com_grph.tracing_info_collected = False
    # import ipdb; ipdb.set_trace()
    net(*args, **kwargs)
    com_grph.tracing_info_collected = True

    labels = nx.get_node_attributes(com_grph.root_graph, 'node_tracing_name')
    colors = nx.get_node_attributes(com_grph.root_graph, 'color')
    colors = [colors[node] for node in com_grph.root_graph.nodes]
    pos = nx.spring_layout(com_grph.root_graph, scale=1)

    G = Network(f'{canvas_size_100s_px[0]*100}px', f'{canvas_size_100s_px[1]*100}px', directed=True)
    G.from_nx(com_grph.root_graph)
    G.repulsion(node_distance=0,
        central_gravity=0,
        spring_length=0,
        spring_strength=0,
        damping=0
    )
    html_filename = file_name[:-4] + '.html'
    G.write_html(html_filename)
    print(f'Interractive html file written. See {html_filename}')
    
    plt.figure(1, figsize=(canvas_size_100s_px[1], canvas_size_100s_px[0]))
    nx.draw(com_grph.root_graph, pos, with_labels=True, labels=labels, connectionstyle='arc3, rad = 0.1',
            node_color=colors, cmap=plt.cm.summer, vmin=0, vmax=1)

    global nodes_connected
    for node in nodes_connected:
        com_grph.remove_node(node)
    nodes_connected = []

    if file_name is not None:
        plt.savefig(file_name)
        plt.clf()
    print(f'NetworkX plot saved. See {file_name}')


com_grph = NetworkModel()
nodes_connected = []


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            with ModuleSpace() as l1:
                self.fc2 = CallWrapper(torch.nn.Linear(120, 84))

            with ModuleSpace() as l2:
                self.fc3 = CallWrapper(torch.nn.Linear(84, 10))
            # self.relu = CallWrapper(F.relu, node_tracing_name='relu')
            self.add = CallWrapper(torch.add, node_tracing_name='add')

        def forward(self, x):
            x = self.fc2(x)
            x1 = F.relu(x)
            x1.node_tracing_name = self.fc2._self_node_tracing_name
            x = self.add(x, x1)
            x = self.fc3(x)
            return x

        def custom_method(self):
            print('Called custom method successfully.')


    net = Net()
    draw(net, './my_graph.png', (16, 38), torch.zeros((120,)))
    #plt.imsave('./my_graph.png')
    plt.show()
