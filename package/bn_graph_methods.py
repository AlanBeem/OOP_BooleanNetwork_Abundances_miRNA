from package.bn_mir_helper_functions_V1 import get_hamming_distance
from typing import Hashable
import networkx as nx


def state_str(input: list[bool]) -> str:
    return "".join([str(int(s)) for s in input])


def run_in_as_edges(input: list[list[bool]]) -> list[list[str, str]]:
    return [
        [state_str(input[i]), state_str(input[i + 1])] for i in range(len(input) - 1)
    ]


def edge_weights(input: list[list[bool]]) -> list[float]:
    return [
        2 * get_hamming_distance(edge[0], edge[1]) / len(edge[0])
        for edge in run_in_as_edges(input)
    ]


class TrajectoryCycleGraph:  # consider: ...(nx.Graph)
    def __init__(
        self,
    ):
        self.edges = set()
        self.reversed_edges = set()  # related to a topological sort from terminal state
        self.cycle_set = set()
        self.nodes = set()
        # self.cycle_exclude_end = []  # manage this with topological sort method if implemented, remove edges between cycle states from graph, then to a topological sort from each cycle state

    def set_cycle(self, cycle_states_list: list[list[bool]]):
        """works as add cycle: self.cycle_set = union(self.cycle_set, set(strs(cycle states)))"""
        self.cycle_set = self.cycle_set.union(
            set([state_str(s) for s in cycle_states_list])
        )  # {str}
        self.add_all(
            run_in_as_edges(cycle_states_list), edge_weights(cycle_states_list)
        )

    def add_all(
        self, edges: list[list[Hashable, Hashable]], weights: list[float | int] = []
    ):
        """edge is [head,tail]"""
        new_edges = set()
        new_reversed_edges = set()
        if weights == []:  # this won't display in a force-directed layout (I think)
            for edge in edges:
                self.nodes.add(edge[0])
                self.nodes.add(edge[1])
                if not tuple(edge) in self.edges:
                    new_edges.add(tuple(edge))
                if not tuple([edge[1], edge[0]]) in self.reversed_edges:
                    new_reversed_edges.add(tuple([edge[1], edge[0]]))
        else:
            for edge, weight in zip(edges, weights):
                self.nodes.add(edge[0])
                self.nodes.add(edge[1])
                if not tuple([edge[0], edge[1], weight]) in self.edges:
                    new_edges.add(tuple([edge[0], edge[1], weight]))
                if not tuple([edge[1], edge[0]]) in self.reversed_edges:
                    new_reversed_edges.add(tuple([edge[1], edge[0], weight]))
        self.edges = self.edges.union(new_edges)
        self.reversed_edges = self.reversed_edges.union(
            new_reversed_edges
        )  # idea: recursive data strucutre- not code running but kinda
        return self.nodes

    def visualize_nx(
        self,
        reversed: bool = True,
        directed: bool = False,
        without_cycle_state_edges: bool = False,
    ):
        G = nx.DiGraph() if directed else nx.Graph()
        for edge in self.reversed_edges if reversed else self.edges:
            if (
                len(edge) == 3
            ):  # and (without_cycle_state_edges or (not edge[0] in self.cycle_set and not edge[1] in self.cycle_set)):
                G.add_edge(edge[0], edge[1], weight=edge[2])
            else:
                # if without_cycle_state_edges or (not edge[0] in self.cycle_set and not edge[1] in self.cycle_set):
                G.add_edge(edge[0], edge[1])
        # visible_edges = [e for e in G.edges]  # allegedly can use 'transparent' as a color
        # invisible_edges = []
        # for n_1 in self.nodes:
        #     for n_2 in self.nodes:
        #         invisible_edges.append((n_1, n_2))
        #         G.add_edge(n_1, n_2, weight=edge_weights([n_1, n_2])[0], edge_color='white')  # more reading, use 'transparent' or transparent
        nx.draw(G, node_size=10, pos=nx.kamada_kawai_layout(G))
        # print(list(g for g in G.nodes))
        nx.draw(
            G.subgraph(list(g for g in G.nodes if g in self.cycle_set)),
            node_size=15,
            pos=nx.kamada_kawai_layout(G),
            node_color="b",
        )
        # return list(g for g in G.nodes)


# colorsys might be interesting to explore in plotting functions
