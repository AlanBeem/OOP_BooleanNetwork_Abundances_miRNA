import numpy as np

# Plotting functions composed in a Jupyter Notebook then accumulated here
import matplotlib.pyplot as plt
import networkx as nx
from package.abundant_boolean_networks_with_micro_rna import *


# 12 17 2024
def get_truth_table(func) -> str:  # for k=2
    out_string = ["a\tb\toutput\n", "1\t1", "1\t0", "0\t1", "0\t0"]
    for L in range(1, len(out_string)):
        split = out_string[L].split("\t")
        a = bool(int(split[0]))
        b = bool(int(split[1]))
        out_string[L] += f"\t{int(func(a, b))}\n"
    return "\n".join(out_string)


# # 08-21-2024
# def get_colors(total_colors_times_5_over_8: int, shuffled_if_true: bool):
#     got_colors = []
#     shuffled_colors = []
#     for each_color_1 in cm.nipy_spectral(
#         np.linspace(0, 1, int(3 * total_colors_times_5_over_8 / 5), True)
#     ).tolist():
#         got_colors.append(each_color_1)
#     for sc_i in range(4):
#         got_colors.append(cm.gist_stern(np.linspace(0.05, 0.8, 4, True)).tolist()[sc_i])
#     test_int_1 = 0
#     for each_color_2 in cm.gist_earth(
#         np.linspace(0.1, 0.9, total_colors_times_5_over_8, True)
#     ).tolist():
#         got_colors.append(each_color_2)
#     if shuffled_if_true:
#         while len(got_colors) > 0:
#             shuffled_colors.append(
#                 got_colors[SystemRandom().randrange(0, len(got_colors))]
#             )
#             got_colors.remove(shuffled_colors[-1])
#         got_colors = shuffled_colors
#     return got_colors


def get_colors_old(boolean_network: BooleanNetwork, shuffled_if_true: bool):
    got_colors = []
    shuffled_colors = []
    for each_color_1 in cm.nipy_spectral(
        np.linspace(0, 1, 3 * int(len(boolean_network.list_of_all_run_ins) / 5), True)
    ).tolist():
        got_colors.append(each_color_1)
    for sc_i in range(4):
        got_colors.append(cm.gist_stern(np.linspace(0.05, 0.8, 4, True).tolist()[sc_i]))
    test_int_1 = 0
    for each_color_2 in cm.gist_earth(
        np.linspace(0.1, 0.9, int(len(boolean_network.list_of_all_run_ins))), True
    ).tolist():
        got_colors.append(each_color_2)
    if shuffled_if_true:
        while len(got_colors) > 0:
            shuffled_colors.append(
                got_colors[SystemRandom().randrange(0, len(got_colors))]
            )
            got_colors.remove(shuffled_colors[-1])
        got_colors = shuffled_colors
    return got_colors


def select_network(
    num_nodes: int,
    minimum_max_cycle_length: int,
    maximum_max_cycle_length: int,
    minimum_number_of_cycles: int,
    maximum_number_of_cycles: int,
    maximum_number_of_networks: int,
    iterations_limit: int = 400,
):
    adequate_network_bool = False
    a_bn_index_of_maximum_length_cycle = (
        0  # becomes a return value  ### Add long names at the end''
    )
    a_bn_list_1 = []  # becomes a return value
    a_bn_1 = None  # becomes a return value
    while not adequate_network_bool:
        a_bn_list_1.append(
            BooleanNetwork(
                num_nodes, [abn_j for abn_j in range(1, 15)], 2, iterations_limit
            )
        )
        a_bn_list_1[-1].run_ins_from_homogeneous_states()
        for abn_i in range(200):
            a_bn_list_1[-1].add_cycle()
        num_cycles = len(a_bn_list_1[-1].bn_collapsed_cycles)
        if 0 < num_cycles <= maximum_number_of_cycles:
            a_bn_list_1[-1].bn_collapsed_cycles.sort_cycle_records_by_cycle_length()
            a_bn_index_of_maximum_length_cycle = 0
            maximum_cycle_length = len(
                a_bn_list_1[-1].bn_collapsed_cycles.cycle_records[0]
            )
            for abn_q in range(1, len(a_bn_list_1[-1].bn_collapsed_cycles)):
                if (
                    len(a_bn_list_1[-1].bn_collapsed_cycles.cycle_records[abn_q])
                    > maximum_cycle_length
                ):
                    maximum_cycle_length = len(
                        a_bn_list_1[-1].bn_collapsed_cycles.cycle_records[abn_q]
                    )
                    a_bn_index_of_maximum_length_cycle = abn_q
            cycle_number = len(a_bn_list_1[-1].bn_collapsed_cycles)
            if (
                minimum_max_cycle_length
                <= maximum_cycle_length
                < maximum_max_cycle_length
            ) and (cycle_number > minimum_number_of_cycles):
                adequate_network_bool = True
            if len(a_bn_list_1) >= maximum_number_of_networks:
                adequate_network_bool = True
    a_bn_1 = a_bn_list_1[-1]
    return [a_bn_1, a_bn_list_1, a_bn_index_of_maximum_length_cycle]


def get_graph_representation(bn: BooleanNetwork):
    edge_list = []
    # edge_colors = []
    edge_labels = dict()
    for node_i in range(len(bn)):
        # node_i is tail
        if (
            bn.nodes[node_i].function.get_expression_string() != "True"
            and bn.nodes[node_i].function.get_expression_string() != "False"
        ):
            if bn.nodes[node_i].function.get_expression_string().count("a") > 0:
                edge_list.append(tuple([bn.node_inputs_assignments[node_i][0], node_i]))
                edge_labels.update({edge_list[-1]: "a"})
            if bn.nodes[node_i].function.get_expression_string().count("b") > 0:
                edge_list.append(tuple([bn.node_inputs_assignments[node_i][1], node_i]))
                edge_labels.update({edge_list[-1]: "b"})
    # colors
    colors = cm.plasma(np.linspace(0, 1, 1000, True)).tolist()
    for c_i in range(len(colors)):
        colors[c_i][3] = 0.4
    node_labels = {}
    for node_k in range(len(bn)):
        # node_labels[node_k] = str(bn.nodes[node_k].function.get_number())
        node_labels[node_k] = str(
            bn.nodes[node_k].function.get_expression_string()
        )  # str(node_k + 1) + ": " +
    # node_color_indices = [bn.nodes[node_j].expression_index + 1 for node_j in range(len(bn))]
    G = nx.DiGraph(edge_list)
    # list(G.edges())
    # Could take this comparison over all run-in steps, which is still about the network, though less about the cycles
    per_node_bv_avg = [0 for avg_i in range(len(bn))]
    avg_sum = sum(
        [
            len(bn.bn_collapsed_cycles.cycle_records[avg_j])
            for avg_j in range(len(bn.bn_collapsed_cycles))
        ]
    )
    for each_cycle_record in bn.bn_collapsed_cycles.cycle_records:
        for node_L in range(len(bn)):
            for state, next_state in zip(
                each_cycle_record.cycle_states_list[0 : len(each_cycle_record) + 1],
                each_cycle_record.cycle_states_list[1 : len(each_cycle_record) + 2],
            ):
                per_node_bv_avg[node_L] += get_boolean_velocity(
                    state[node_L], next_state[node_L]
                )
            per_node_bv_avg[node_L] /= avg_sum
    # print(per_node_bv_avg)
    node_colors = [colors[int(per_node_bv_avg[nc_i]) * 1000] for nc_i in range(len(bn))]
    fig = plt.figure()
    # nx.draw(G, with_labels=True, labels=node_labels, font_weight='bold', arrows=True, arrowsize=15, node_color=node_colors)  # , edge_color=edge_colors
    nx.draw(
        G,
        with_labels=True,
        labels=node_labels,
        font_weight="bold",
        node_size=300,
        arrows=True,
        arrowsize=15,
        pos=nx.circular_layout(G, 1, None, 2),
        node_color=node_colors,
    )  # , edge_color=edge_colors
    nx.draw_networkx_edge_labels(
        G, edge_labels=edge_labels, pos=nx.circular_layout(G, 1, None, 2)
    )
    # subax2 = plt.subplot(122)
    # nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
    fig.set_size_inches(8, 8)
    return fig


# use graphviz instead


# circa 09-14
def get_list_of_node_sequences(cycle_record: CycleRecord):
    # By using the __len__() method of CycleRecord, this ignores the duplicated first/last entries of cycle states list
    # return [[cycle_record.cycle_states_list[glns_i][glns_j] for glns_i in range(len(cycle_record))] for glns_j in range(len(cycle_record.cycle_states_list[0]))]
    # print(cycle_record) ## it was working, but, autocomplete had suggested a test_sequence_1 as test_cycle_record_1 ... ugh, gotta read those. [[todo implement use of generic would-be-namespace conflict reference names then replace all ? needs to be prefix, still]]
    glns_temp_list = [
        [1 for glns_j in range(len(cycle_record))]
        for glns_k in range(len(cycle_record.cycle_states_list[0]))
    ]
    for glns_i in range(len(cycle_record)):
        for glns_L in range(len(cycle_record.cycle_states_list[0])):
            glns_temp_list[glns_L][glns_i] = cycle_record.cycle_states_list[glns_i][
                glns_L
            ]  # micro design pattern, switching addressing order of 2D ...
    return glns_temp_list


def per_node_sequence_similarity_posterity(
    cycle_record_1: CycleRecord,
    cycle_record_2: CycleRecord,
    boolean_network: BooleanNetwork or None,
):
    if boolean_network is not None:
        pnss_report_string = (
            "Cycle indices:\ncycle_record_1: "
            + str(
                boolean_network.bn_collapsed_cycles.cycle_records.index(cycle_record_1)
            )
            + ", "
            + str(
                boolean_network.bn_collapsed_cycles.cycle_records.index(cycle_record_2)
            )
            + "\n"
        )
    else:
        pnss_report_string = ""
    pnss_report_string += (
        "Cycle lengths:\ncycle_record_1: "
        + str(len(cycle_record_1))
        + ", cycle_record_2: "
        + str(len(cycle_record_2))
        + "\n"
    )
    pnss_similarity_i_j = 0
    for pnss_node_index_1 in range(
        len(cycle_record_1.cycle_states_list[0])
    ):  # For each node
        pnss_report_string += (
            "\nnode index: " + str(pnss_node_index_1) + ", comparing 1 to 2\n"
        )
        pnss_start_position = 0
        pnss_bool_1 = True
        while pnss_bool_1:
            temp_sequence_1 = [
                cycle_record_1.cycle_states_list[
                    (pnss_i + pnss_start_position) % len(cycle_record_1)
                ][pnss_node_index_1]
                for pnss_i in range(len(cycle_record_1))
            ]
            pnss_report_string += (
                "temp_sequence_1: "
                + str(temp_sequence_1)
                + ", cycle_record_1: "
                + str(cycle_record_1.cycle_states_list)
                + "\n"
            )
            temp_sequence_2 = [
                cycle_record_2.cycle_states_list[pnss_j % len(cycle_record_2)][
                    pnss_node_index_1
                ]
                for pnss_j in range(len(cycle_record_1))
            ]
            pnss_report_string += (
                "temp_sequence_2: "
                + str(temp_sequence_2)
                + ", cycle_record_2: "
                + str(cycle_record_2.cycle_states_list)
                + "\n"
            )
            pnss_start_position += 1
            pnss_bool_2 = True
            for pnss_i in range(len(temp_sequence_1)):
                if temp_sequence_1[pnss_i] != temp_sequence_2[pnss_i]:
                    pnss_bool_2 = False
                    break
            if pnss_bool_2:
                pnss_similarity_i_j += 1
                pnss_report_string += "Adding 1 to similarity\n"
                pnss_bool_1 = False
            if pnss_start_position >= len(cycle_record_1):
                pnss_bool_1 = False
    pnss_report_string += "pnss_similarity_i_j: " + str(pnss_similarity_i_j) + "\n"
    pnss_similarity_j_i = 0
    for pnss_node_index in range(
        len(cycle_record_2.cycle_states_list[0])
    ):  # For each node
        pnss_report_string += (
            "\nnode index: " + str(pnss_node_index) + ", comparing 2 to 1\n"
        )
        pnss_start_position = 0
        pnss_bool_1 = True
        while pnss_bool_1:
            temp_sequence_1 = [
                cycle_record_2.cycle_states_list[
                    (pnss_i + pnss_start_position) % (len(cycle_record_2))
                ][pnss_node_index]
                for pnss_i in range(len(cycle_record_2))
            ]
            pnss_report_string += (
                "temp_sequence_1: "
                + str(temp_sequence_1)
                + ", cycle_record_2: "
                + str(cycle_record_2.cycle_states_list)
                + "\n"
            )
            temp_sequence_2 = [
                cycle_record_1.cycle_states_list[pnss_j % (len(cycle_record_1))][
                    pnss_node_index
                ]
                for pnss_j in range(len(cycle_record_2))
            ]  # [This can lead to comparing only 1 of states ... so answer would've been take the minimum of both comparisons
            pnss_report_string += (
                "temp_sequence_2: "
                + str(temp_sequence_2)
                + ", cycle_record_1: "
                + str(cycle_record_1.cycle_states_list)
                + "\n"
            )
            # print("len(cycle_record_1)" + str(len(cycle_record_1)))
            # print("len(cycle_record_2)" + str(len(cycle_record_2)))
            pnss_start_position += 1
            # print()
            pnss_bool_2 = True
            for pnss_i in range(len(temp_sequence_2)):
                if temp_sequence_1[pnss_i] != temp_sequence_2[pnss_i]:
                    pnss_bool_2 = False
                    break
            if pnss_bool_2:
                pnss_report_string += "Adding 1 to similarity\n"
                pnss_similarity_j_i += 1
                pnss_bool_1 = False
            if pnss_start_position >= len(cycle_record_2):
                pnss_bool_1 = False
    pnss_report_string += "pnss_similarity_j_i: " + str(pnss_similarity_j_i) + "\n"
    # third method, find the shorter record first
    pnss_node_index = 0
    pnss_similarity_3 = 0
    if len(cycle_record_1) <= len(cycle_record_2):
        shorter_cycle_states_list = cycle_record_1.cycle_states_list[
            0 : len(cycle_record_1)
        ]
        longer_cycle_states_list = cycle_record_2.cycle_states_list[
            0 : len(cycle_record_2)
        ]
    else:
        shorter_cycle_states_list = cycle_record_2.cycle_states_list[
            0 : len(cycle_record_2)
        ]
        longer_cycle_states_list = cycle_record_1.cycle_states_list[
            0 : len(cycle_record_1)
        ]
    pnss_report_string += (
        "\nreport continued, comparing circular shorter record to longer record"
    )
    for pnss_node_index in range(
        len(cycle_record_2.cycle_states_list[0])
    ):  # For each node
        pnss_report_string += (
            "\nnode index: " + str(pnss_node_index) + ", comparing shorter to longer\n"
        )
        pnss_start_position = 0
        pnss_bool_1 = True
        while pnss_bool_1:
            temp_sequence_1 = [
                shorter_cycle_states_list[
                    (pnss_i + pnss_start_position) % (len(shorter_cycle_states_list))
                ][pnss_node_index]
                for pnss_i in range(len(longer_cycle_states_list))
            ]
            pnss_report_string += (
                "temp_sequence_1: "
                + str(temp_sequence_1)
                + ", shorter_cycle_states_list: "
                + str(shorter_cycle_states_list)
                + "\n"
            )
            temp_sequence_2 = [
                longer_cycle_states_list[pnss_j][pnss_node_index]
                for pnss_j in range(len(longer_cycle_states_list))
            ]
            pnss_report_string += (
                "temp_sequence_2: "
                + str(temp_sequence_2)
                + ", longer_cycle_states_list: "
                + str(longer_cycle_states_list)
                + "\n"
            )
            pnss_start_position += 1
            pnss_bool_2 = True
            for pnss_i in range(len(temp_sequence_2)):
                if temp_sequence_1[pnss_i] != temp_sequence_2[pnss_i]:
                    pnss_bool_2 = False
                    break
            if pnss_bool_2:
                pnss_report_string += "Adding 1 to similarity\n"
                pnss_similarity_3 += 1
                pnss_bool_1 = False
            if pnss_start_position >= len(cycle_record_2):
                pnss_bool_1 = False
    if (
        (pnss_similarity_i_j != pnss_similarity_3)
        or (pnss_similarity_j_i != pnss_similarity_3)
        or (pnss_similarity_i_j != pnss_similarity_j_i)
    ):
        if pnss_similarity_i_j != pnss_similarity_j_i:
            print(
                "i_j != j_i: "
                + str(pnss_similarity_i_j)
                + " != "
                + str(pnss_similarity_j_i)
            )
        if pnss_similarity_i_j != pnss_similarity_3:
            print(
                "i_j != _3: "
                + str(pnss_similarity_i_j)
                + " != "
                + str(pnss_similarity_3)
            )
        if pnss_similarity_j_i != pnss_similarity_3:
            print(
                "j_i != _3: "
                + str(pnss_similarity_j_i)
                + " != "
                + str(pnss_similarity_3)
            )
        print("report:")
        print(pnss_report_string)
    return pnss_similarity_3 / len(cycle_record_1.cycle_states_list[0])


def per_node_sequence_similarity_constant_non_constant(
    cycle_record_1: CycleRecord,
    cycle_record_2: CycleRecord,
    constant_only_if_true: bool,
    non_constant_only_if_true: bool,
):
    pnss_similarity_3 = 0
    if len(cycle_record_1) <= len(cycle_record_2):
        shorter_cycle_states_list = cycle_record_1.cycle_states_list[
            0 : len(cycle_record_1)
        ]
        longer_cycle_states_list = cycle_record_2.cycle_states_list[
            0 : len(cycle_record_2)
        ]
    else:
        shorter_cycle_states_list = cycle_record_2.cycle_states_list[
            0 : len(cycle_record_2)
        ]
        longer_cycle_states_list = cycle_record_1.cycle_states_list[
            0 : len(cycle_record_1)
        ]
    if constant_only_if_true and non_constant_only_if_true:
        print("Conflicting options, returning None")
        print("cycle_record_1" + str(cycle_record_1))
        print("cycle_record_2: " + str(cycle_record_2))
        return None
    else:
        for pnss_node_index in range(
            len(cycle_record_2.cycle_states_list[0])
        ):  # For each node
            pnss_bool_node_constant = True
            for pnss_k in range(len(shorter_cycle_states_list)):
                if (
                    shorter_cycle_states_list[pnss_k][pnss_node_index]
                    != shorter_cycle_states_list[0][pnss_node_index]
                ):
                    pnss_bool_node_constant = False
                    break
            if pnss_bool_node_constant:
                for pnss_L in range(len(longer_cycle_states_list)):
                    if (
                        longer_cycle_states_list[pnss_L][pnss_node_index]
                        != longer_cycle_states_list[0][pnss_node_index]
                    ):
                        pnss_bool_node_constant = False
                        break
            # This statement is not working (what do you know!) had old ..._bool_3 still doesn't work
            if (
                (constant_only_if_true and pnss_bool_node_constant)
                or ((not constant_only_if_true) and non_constant_only_if_true)
            ) and (
                (non_constant_only_if_true and not pnss_bool_node_constant)
                or ((not non_constant_only_if_true) and constant_only_if_true)
            ):
                pnss_start_position = 0
                pnss_bool_1 = True
                while pnss_bool_1:
                    temp_sequence_1 = [
                        shorter_cycle_states_list[
                            (pnss_i + pnss_start_position)
                            % (len(shorter_cycle_states_list))
                        ][pnss_node_index]
                        for pnss_i in range(len(longer_cycle_states_list))
                    ]
                    temp_sequence_2 = [
                        longer_cycle_states_list[pnss_j][pnss_node_index]
                        for pnss_j in range(len(longer_cycle_states_list))
                    ]
                    pnss_start_position += 1
                    pnss_bool_2 = True
                    for pnss_i in range(len(temp_sequence_2)):
                        if temp_sequence_1[pnss_i] != temp_sequence_2[pnss_i]:
                            pnss_bool_2 = False
                            break
                    if pnss_bool_2:
                        pnss_similarity_3 += 1
                        pnss_bool_1 = False
                    if pnss_start_position >= len(cycle_record_2):
                        pnss_bool_1 = False
    return pnss_similarity_3 / len(cycle_record_1.cycle_states_list[0])


def per_node_sequence_similarity(
    cycle_record_1: CycleRecord,
    cycle_record_2: CycleRecord,
    nodes_indices_to_compare: list[int],
):
    """Compares cycle_record_1 to cycle_record_2, for the nodes' indices in nodes_indices_to_compare"""
    pnss_similarity_3 = 0
    if len(nodes_indices_to_compare) == 0:
        pnss_divisor = 1
    else:
        pnss_divisor = len(nodes_indices_to_compare)
    if len(cycle_record_1) <= len(cycle_record_2):
        shorter_cycle_states_list = cycle_record_1.cycle_states_list[
            0 : len(cycle_record_1) + 1
        ]
        longer_cycle_states_list = cycle_record_2.cycle_states_list[
            0 : len(cycle_record_2) + 1
        ]
    else:
        shorter_cycle_states_list = cycle_record_2.cycle_states_list[
            0 : len(cycle_record_2) + 1
        ]
        longer_cycle_states_list = cycle_record_1.cycle_states_list[
            0 : len(cycle_record_1) + 1
        ]
    for pnss_node_index in nodes_indices_to_compare:  # For each node
        pnss_start_position = 0
        pnss_bool_1 = True
        while pnss_bool_1:
            temp_sequence_1 = [
                shorter_cycle_states_list[
                    (pnss_i + pnss_start_position) % (len(shorter_cycle_states_list))
                ][pnss_node_index]
                for pnss_i in range(len(longer_cycle_states_list))
            ]
            temp_sequence_2 = [
                longer_cycle_states_list[pnss_j][pnss_node_index]
                for pnss_j in range(len(longer_cycle_states_list))
            ]
            pnss_start_position += 1
            pnss_bool_2 = True
            for pnss_i in range(len(temp_sequence_2)):
                if temp_sequence_1[pnss_i] != temp_sequence_2[pnss_i]:
                    pnss_bool_2 = False
                    break
            if pnss_bool_2:
                pnss_similarity_3 += 1
                pnss_bool_1 = False
            if pnss_start_position >= len(
                temp_sequence_1
            ):  # 08-21-2024 changed >= to > (increment occurs after last check)  ### OH I see the problem, len(cycle_record_2) not len(longer ? -> shorter
                pnss_bool_1 = False
    return pnss_similarity_3 / pnss_divisor


def align_cycle_record(cycle_record: CycleRecord, origin_sequence: list[bool]):
    """align_cycle_record is a mutator method that alters the state of the CycleRecord parameter"""
    acr_distances = []
    acr_temp_cycle_states_list = []
    for acr_i in range(len(cycle_record)):
        acr_distances.append(
            get_hamming_distance(cycle_record.cycle_states_list[acr_i], origin_sequence)
        )
    acr_index_of_least_distant_state = acr_distances.index(min(acr_distances))
    for acr_j in range(len(cycle_record)):
        acr_temp_cycle_states_list.append(
            cycle_record.cycle_states_list[
                (acr_j + acr_index_of_least_distant_state) % len(cycle_record)
            ]
        )
    acr_temp_cycle_states_list.append(acr_temp_cycle_states_list[0].copy())
    cycle_record.cycle_states_list = copy.deepcopy(acr_temp_cycle_states_list)
    # Post-condition: the cycle_states_list of the CycleRecord parameter starts/ends with the cycle state least distant from origin_sequence


def align_cycle_states_lists_of_cc_to_cons_seq(
    collapsed_cycles: CollapsedCycles,
):  # But also, can this modify the reference? 08-20-2024
    """align_cycle_states_lists_of_cc_to_cons_seq is a mutator method that alters the state of the parameter"""
    acslcc_all_states = []
    for acslcc_i in range(len(collapsed_cycles)):
        for acslcc_j in range(len(collapsed_cycles.cycle_records[acslcc_i])):
            acslcc_all_states.append(
                collapsed_cycles.cycle_records[acslcc_i].cycle_states_list[acslcc_j]
            )
    acslcc_cons_seq = get_consensus_sequence_list(acslcc_all_states, [False, True])
    for acslcc_k in range(len(collapsed_cycles)):
        align_cycle_record(collapsed_cycles.cycle_records[acslcc_k], acslcc_cons_seq)
    # Post-condition: all cycle records of reference CollapsedCycles start / end on the cycle state least distant from the consensus sequence of all cycle states
    # Could weight by cycle states lengths, such that equilibrial states had more ... but how many? multiple lengths of longer cycles, could normalize all to that, but, cycles with more states are different than cycles with less states, still it's just a consensus sequence, so,  ...


def get_least_common_multiple(
    iterable_ints,
):  # TODO rename and refactor, double check LCM algorithm
    glcm_return_int = max(iterable_ints)
    iterable_ints = sorted(iterable_ints, reverse=True)
    for each_int in iterable_ints:
        if glcm_return_int % each_int != 0:
            glcm_return_int *= each_int
    return glcm_return_int


def plot_polar_cycles_cons_hamm(
    boolean_network: BooleanNetwork, plot_one_period_if_true: bool, bv_colors
):
    align_cycle_states_lists_of_cc_to_cons_seq(boolean_network.bn_collapsed_cycles)
    ppc_all_states = []
    for ppc_i in range(len(boolean_network.bn_collapsed_cycles)):
        for ppc_j in range(
            len(boolean_network.bn_collapsed_cycles.cycle_records[ppc_i])
        ):
            ppc_all_states.append(
                boolean_network.bn_collapsed_cycles.cycle_records[
                    ppc_i
                ].cycle_states_list[ppc_j]
            )
    ppc_cons_seq = get_consensus_sequence_list(ppc_all_states, [False, True])
    #
    print("Radius: Hamming distance to consensus of cycle states")
    if not plot_one_period_if_true:
        ppc_cycle_lengths_set = set()
        for each_cycle in boolean_network.bn_collapsed_cycles.cycle_records:
            ppc_cycle_lengths_set.add(len(each_cycle))
        # ppc_least_common_multiple = get_least_common_multiple(ppc_cycle_lengths_set)
        ppc_least_common_multiple = np.lcm.reduce(ppc_cycle_lengths_set)
        print(
            "Theta: 1 step == "
            + (str(2 * math.pi / ppc_least_common_multiple))
            + " radians"
        )
        ppc_theta = [
            ppc_i * 2 * math.pi / ppc_least_common_multiple
            for ppc_i in range(ppc_least_common_multiple)
        ]  # in radians
        ppc_theta.append(0)
        for ppc_k in range(len(boolean_network.bn_collapsed_cycles)):
            ppc_radius = []
            for ppc_j in range(ppc_least_common_multiple + 1):
                ppc_radius.append(
                    get_hamming_distance(
                        boolean_network.bn_collapsed_cycles.cycle_records[ppc_k][
                            ppc_j
                            % (
                                len(
                                    boolean_network.bn_collapsed_cycles.cycle_records[
                                        ppc_k
                                    ]
                                )
                            )
                        ],
                        ppc_cons_seq,
                    )
                )
            plt.polar(ppc_theta, ppc_radius, color=bv_colors[ppc_k], alpha=0.3)
            plt.polar(
                ppc_theta[
                    0 : len(boolean_network.bn_collapsed_cycles.cycle_records[ppc_k])
                    + 1
                ],
                ppc_radius[
                    0 : len(boolean_network.bn_collapsed_cycles.cycle_records[ppc_k])
                    + 1
                ],
                color=bv_colors[ppc_k],
                alpha=0.8,
            )
        plt.show()
    else:
        print("Theta: varied: 1 step == 2 * Pi / (cycle length)")
        for ppc_k in range(len(boolean_network.bn_collapsed_cycles)):
            ppc_theta = [
                ppc_i
                * 2
                * math.pi
                / len(boolean_network.bn_collapsed_cycles.cycle_records[ppc_k])
                for ppc_i in range(
                    len(boolean_network.bn_collapsed_cycles.cycle_records[ppc_k]) + 1
                )
            ]  # in radians
            # ppc_theta.append(0)
            ppc_radius = []
            for ppc_j in range(
                len(boolean_network.bn_collapsed_cycles.cycle_records[ppc_k]) + 1
            ):
                ppc_radius.append(
                    get_hamming_distance(
                        boolean_network.bn_collapsed_cycles.cycle_records[ppc_k][
                            ppc_j
                            % (
                                len(
                                    boolean_network.bn_collapsed_cycles.cycle_records[
                                        ppc_k
                                    ]
                                )
                            )
                        ],
                        ppc_cons_seq,
                    )
                )
            plt.polar(ppc_theta, ppc_radius, color=bv_colors[ppc_k], alpha=0.3)
        plt.show()


def plot_polar_cycles_input_seq(
    boolean_network: BooleanNetwork,
    origin_sequence: list[bool],
    plot_one_period_if_true: bool,
    bv_colors,
):
    align_cycle_states_lists_of_cc_to_cons_seq(boolean_network.bn_collapsed_cycles)
    #
    print("Radius: Hamming distance to: " + str(origin_sequence))
    #
    if not plot_one_period_if_true:
        ppc_cycle_lengths_set = set()
        for each_cycle in boolean_network.bn_collapsed_cycles.cycle_records:
            ppc_cycle_lengths_set.add(len(each_cycle))
        ppc_least_common_multiple = get_least_common_multiple(ppc_cycle_lengths_set)
        print(
            "Theta: 1 step == "
            + (str(2 * math.pi / ppc_least_common_multiple))
            + " radians"
        )
        ppc_theta = [
            ppc_i * 2 * math.pi / ppc_least_common_multiple
            for ppc_i in range(ppc_least_common_multiple)
        ]  # in radians
        ppc_theta.append(0)
        for ppc_k in range(len(boolean_network.bn_collapsed_cycles)):
            ppc_radius = []
            for ppc_j in range(ppc_least_common_multiple + 1):
                ppc_radius.append(
                    get_hamming_distance(
                        boolean_network.bn_collapsed_cycles.cycle_records[ppc_k][
                            ppc_j
                            % (
                                len(
                                    boolean_network.bn_collapsed_cycles.cycle_records[
                                        ppc_k
                                    ]
                                )
                            )
                        ],
                        origin_sequence,
                    )
                )
            plt.polar(ppc_theta, ppc_radius, color=bv_colors[ppc_k], alpha=0.3)
            plt.polar(
                ppc_theta[
                    0 : len(boolean_network.bn_collapsed_cycles.cycle_records[ppc_k])
                    + 1
                ],
                ppc_radius[
                    0 : len(boolean_network.bn_collapsed_cycles.cycle_records[ppc_k])
                    + 1
                ],
                color=bv_colors[ppc_k],
                alpha=0.8,
            )
        plt.show()
    else:
        print("Theta: varied: 1 step == 2 * Pi / (cycle length)")
        for ppc_k in range(len(boolean_network.bn_collapsed_cycles)):
            ppc_theta = [
                ppc_i
                * 2
                * math.pi
                / len(boolean_network.bn_collapsed_cycles.cycle_records[ppc_k])
                for ppc_i in range(
                    len(boolean_network.bn_collapsed_cycles.cycle_records[ppc_k]) + 1
                )
            ]  # in radians
            # ppc_theta.append(0)
            ppc_radius = []
            for ppc_j in range(
                len(boolean_network.bn_collapsed_cycles.cycle_records[ppc_k]) + 1
            ):
                ppc_radius.append(
                    get_hamming_distance(
                        boolean_network.bn_collapsed_cycles.cycle_records[ppc_k][
                            ppc_j
                            % (
                                len(
                                    boolean_network.bn_collapsed_cycles.cycle_records[
                                        ppc_k
                                    ]
                                )
                            )
                        ],
                        origin_sequence,
                    )
                )
            plt.polar(ppc_theta, ppc_radius, color=bv_colors[ppc_k], alpha=0.3)
        plt.show()


def plot_polar_cycles_bv(
    boolean_network: BooleanNetwork, plot_one_period_if_true: bool, bv_colors
):
    align_cycle_states_lists_of_cc_to_cons_seq(boolean_network.bn_collapsed_cycles)
    #
    print("Radius: step to step edit-distance")
    #
    if not plot_one_period_if_true:
        ppc_cycle_lengths_set = set()
        for each_cycle in boolean_network.bn_collapsed_cycles.cycle_records:
            ppc_cycle_lengths_set.add(len(each_cycle))
        ppc_least_common_multiple = np.lcm.reduce(ppc_cycle_lengths_set)
        print(
            "Theta: 1 step == "
            + (str(2 * math.pi / ppc_least_common_multiple))
            + " radians"
        )
        print("least common multiple: " + str(ppc_least_common_multiple))
        ppc_theta = [
            ppc_i * 2 * math.pi / ppc_least_common_multiple
            for ppc_i in range(ppc_least_common_multiple)
        ]  # in radians
        ppc_theta.append(0)
        fig_1, ax_1 = plt.subplots(subplot_kw={"projection": "polar"})
        for ppc_k in range(len(boolean_network.bn_collapsed_cycles)):
            ppc_radius = []
            for ppc_j in range(ppc_least_common_multiple + 1):
                ppc_radius.append(
                    get_hamming_distance(
                        boolean_network.bn_collapsed_cycles.cycle_records[ppc_k][
                            ppc_j
                            % (
                                len(
                                    boolean_network.bn_collapsed_cycles.cycle_records[
                                        ppc_k
                                    ]
                                )
                            )
                        ],
                        boolean_network.bn_collapsed_cycles.cycle_records[ppc_k][
                            (ppc_j - 1)
                            % (
                                len(
                                    boolean_network.bn_collapsed_cycles.cycle_records[
                                        ppc_k
                                    ]
                                )
                            )
                        ],
                    )
                )
            ax_1.plot(ppc_theta, ppc_radius, color=bv_colors[ppc_k], alpha=0.3)
            ax_1.plot(
                ppc_theta[
                    0 : len(boolean_network.bn_collapsed_cycles.cycle_records[ppc_k])
                    + 1
                ],
                ppc_radius[
                    0 : len(boolean_network.bn_collapsed_cycles.cycle_records[ppc_k])
                    + 1
                ],
                color=bv_colors[ppc_k],
                alpha=0.8,
            )
        ax_1.grid(False)
        plt.show()
    else:
        cycle_lengths = [
            24 * 60 / len(boolean_network.bn_collapsed_cycles.cycle_records[cl_i])
            for cl_i in range(len(boolean_network.bn_collapsed_cycles))
        ]
        print(
            "Assuming each cycle is 24 hours, 1 step == "
            + str(cycle_lengths)
            + " minutes"
        )
        fig_1, ax_1 = plt.subplots(subplot_kw={"projection": "polar"})
        for ppc_k in range(len(boolean_network.bn_collapsed_cycles)):
            ppc_theta = [
                ppc_i
                * 2
                * math.pi
                / len(boolean_network.bn_collapsed_cycles.cycle_records[ppc_k])
                for ppc_i in range(
                    len(boolean_network.bn_collapsed_cycles.cycle_records[ppc_k]) + 1
                )
            ]  # in radians
            ppc_radius = []
            for ppc_j in range(
                len(boolean_network.bn_collapsed_cycles.cycle_records[ppc_k]) + 1
            ):
                ppc_radius.append(
                    get_hamming_distance(
                        boolean_network.bn_collapsed_cycles.cycle_records[ppc_k][
                            ppc_j
                            % (
                                len(
                                    boolean_network.bn_collapsed_cycles.cycle_records[
                                        ppc_k
                                    ]
                                )
                            )
                        ],
                        boolean_network.bn_collapsed_cycles.cycle_records[ppc_k][
                            (ppc_j - 1)
                            % (
                                len(
                                    boolean_network.bn_collapsed_cycles.cycle_records[
                                        ppc_k
                                    ]
                                )
                            )
                        ],
                    )
                )
            ax_1.plot(ppc_theta, ppc_radius, color=bv_colors[ppc_k], alpha=0.3)
        ax_1.grid(False)
        ax_1.legend(labels=["" for fl_i in range(len(cycle_lengths))])
        plt.show()


def plot_cycle_similarity_non_constant_nodes(bn: BooleanNetwork):
    print("Comparing non-constant nodes only")
    fig, axs = plt.subplots(1, 2)
    shuffle(bn.bn_collapsed_cycles.cycle_records)
    bn.bn_collapsed_cycles.sort_cycle_records_by_cycle_length()
    cycle_similarity_matrix = [
        [0 for csm_i in range(len(bn.bn_collapsed_cycles))]
        for csm_j in range(len(bn.bn_collapsed_cycles))
    ]
    constant_nodes_per_cycle = [[] for cnpc_i in range(len(bn.bn_collapsed_cycles))]
    non_constant_nodes_per_cycle = [
        [] for cnpc_j in range(len(bn.bn_collapsed_cycles))
    ]  # TODO detailed testing to determine how and when this necessary (namespace) (also, check PEP on reuse of variables guidance) 08-21-2024
    for bn_a in range(len(bn)):  # For each node
        for bn_c in range(
            len(bn.bn_collapsed_cycles)
        ):  # For each cycle append to either constant or non-constant
            csm_bool_2 = True
            for bn_b in range(
                1, len(bn.bn_collapsed_cycles.cycle_records[bn_c])
            ):  # For each state of cycle, except 0th, bn_a: each node
                if (
                    bn.bn_collapsed_cycles.cycle_records[bn_c][bn_b][bn_a]
                    != bn.bn_collapsed_cycles.cycle_records[bn_c][0][bn_a]
                ):
                    csm_bool_2 = False
                    non_constant_nodes_per_cycle[bn_c].append(
                        bn_a
                    )  # Could skip break and do a statement at else in below if; ... imagine getting a big print of a sprint[ ?] or code development process at completion of it, that'd be something to look at
                    break
            if csm_bool_2:
                constant_nodes_per_cycle[bn_c].append(bn_a)
    nodes_to_compare_3 = non_constant_nodes_per_cycle[0]
    for bn_i in range(len(bn.bn_collapsed_cycles)):
        for bn_j in range(len(bn.bn_collapsed_cycles)):
            cycle_similarity_matrix[bn_i][bn_j] = per_node_sequence_similarity(
                bn.bn_collapsed_cycles.cycle_records[bn_i],
                bn.bn_collapsed_cycles.cycle_records[bn_j],
                nodes_to_compare_3,
            )

    test_csm_1 = copy.deepcopy(cycle_similarity_matrix)
    tcsm_max = 0
    tcsm_min = 10
    for t_i in range(len(test_csm_1)):
        for t_j in range(len(test_csm_1)):
            if test_csm_1[t_i][t_j] > tcsm_max:
                tcsm_max = test_csm_1[t_i][t_j]
            if test_csm_1[t_i][t_j] < tcsm_min:
                tcsm_min = test_csm_1[t_i][t_j]
            if t_j < t_i:
                test_csm_1[t_i][t_j] = 0
    axs[0].imshow(test_csm_1, cmap="plasma", vmin=tcsm_min, vmax=tcsm_max)
    axs[0].set_title("Sorted by cycle length")  # Like post-condition
    print(
        "Sorted by cycle length:\n"
        + str(
            [
                len(bn.bn_collapsed_cycles.cycle_records[tcsm_k])
                for tcsm_k in range(len(bn.bn_collapsed_cycles))
            ]
        )
    )
    bn.sort_cycle_records_by_num_observations()
    cycle_similarity_matrix = [
        [0 for csm_i in range(len(bn.bn_collapsed_cycles))]
        for csm_j in range(len(bn.bn_collapsed_cycles))
    ]
    for bn_i in range(len(bn.bn_collapsed_cycles)):
        for bn_j in range(len(bn.bn_collapsed_cycles)):
            cycle_similarity_matrix[bn_i][bn_j] = per_node_sequence_similarity(
                bn.bn_collapsed_cycles.cycle_records[bn_i],
                bn.bn_collapsed_cycles.cycle_records[bn_j],
                nodes_to_compare_3,
            )

    test_csm_2 = copy.deepcopy(cycle_similarity_matrix)
    tcsm_max = 0
    tcsm_min = 10
    for t_i in range(len(test_csm_2)):
        for t_j in range(len(test_csm_2)):
            if test_csm_2[t_i][t_j] > tcsm_max:
                tcsm_max = test_csm_2[t_i][t_j]
            if test_csm_2[t_i][t_j] < tcsm_min:
                tcsm_min = test_csm_2[t_i][t_j]
            if t_j < t_i:
                test_csm_2[t_i][t_j] = 0
    axs[1].imshow(
        test_csm_2, cmap="plasma", vmin=tcsm_min, vmax=tcsm_max
    )  # Sorting by num observations makes a difference, add more sorts-- like, cycle length -- already have that, sick
    axs[1].set_title("Sorted by number of observations")  # Like post-condition
    print(
        "Sorted by number of observations:\n"
        + str(
            [
                bn.bn_collapsed_cycles.cycle_records[tcsm_k].num_observations
                for tcsm_k in range(len(bn.bn_collapsed_cycles))
            ]
        )
    )
    plt.show()
    plt.figure()
    bn.bn_collapsed_cycles.sort_cycle_records_by_hamming_diameter()
    cycle_similarity_matrix = [
        [0 for csm_i in range(len(bn.bn_collapsed_cycles))]
        for csm_j in range(len(bn.bn_collapsed_cycles))
    ]
    for bn_i in range(len(bn.bn_collapsed_cycles)):
        for bn_j in range(len(bn.bn_collapsed_cycles)):
            cycle_similarity_matrix[bn_i][bn_j] = per_node_sequence_similarity(
                bn.bn_collapsed_cycles.cycle_records[bn_i],
                bn.bn_collapsed_cycles.cycle_records[bn_j],
                nodes_to_compare_3,
            )
    test_csm_3 = copy.deepcopy(cycle_similarity_matrix)
    tcsm_max = 0
    tcsm_min = 10
    for t_i in range(len(test_csm_3)):
        for t_j in range(len(test_csm_3)):
            if test_csm_3[t_i][t_j] > tcsm_max:
                tcsm_max = test_csm_3[t_i][t_j]
            if test_csm_3[t_i][t_j] < tcsm_min:
                tcsm_min = test_csm_3[t_i][t_j]
            if t_j < t_i:
                test_csm_3[t_i][t_j] = 0
    print("Sorted by Hamming diameter")
    print(
        "Cycle length:\n"
        + str(
            [
                len(bn.bn_collapsed_cycles.cycle_records[tcsm_k])
                for tcsm_k in range(len(bn.bn_collapsed_cycles))
            ]
        )
    )
    print(
        "Number of observations:\n"
        + str(
            [
                bn.bn_collapsed_cycles.cycle_records[tcsm_k].num_observations
                for tcsm_k in range(len(bn.bn_collapsed_cycles))
            ]
        )
    )
    plt.imshow(test_csm_3, cmap="plasma", vmin=tcsm_min, vmax=tcsm_max)
    plt.show()
    #
    plt.figure()
    bn.bn_collapsed_cycles.sort_by_cycle_similarity(
        bn.bn_collapsed_cycles.get_furthest_reference(
            bn.bn_collapsed_cycles.get_consensus_state()
        )
    )
    cycle_similarity_matrix = [
        [0 for csm_i in range(len(bn.bn_collapsed_cycles))]
        for csm_j in range(len(bn.bn_collapsed_cycles))
    ]
    for bn_i in range(len(bn.bn_collapsed_cycles)):
        for bn_j in range(len(bn.bn_collapsed_cycles)):
            cycle_similarity_matrix[bn_i][bn_j] = per_node_sequence_similarity(
                bn.bn_collapsed_cycles.cycle_records[bn_i],
                bn.bn_collapsed_cycles.cycle_records[bn_j],
                nodes_to_compare_3,
            )
    test_csm_3 = copy.deepcopy(cycle_similarity_matrix)
    tcsm_max = 0
    tcsm_min = 10
    for t_i in range(len(test_csm_3)):
        for t_j in range(len(test_csm_3)):
            if test_csm_3[t_i][t_j] > tcsm_max:
                tcsm_max = test_csm_3[t_i][t_j]
            if test_csm_3[t_i][t_j] < tcsm_min:
                tcsm_min = test_csm_3[t_i][t_j]
            if t_j < t_i:
                test_csm_3[t_i][t_j] = 0
    print("Sorted by cycle similarity")
    print(
        "Cycle length:\n"
        + str(
            [
                len(bn.bn_collapsed_cycles.cycle_records[tcsm_k])
                for tcsm_k in range(len(bn.bn_collapsed_cycles))
            ]
        )
    )
    print(
        "Number of observations:\n"
        + str(
            [
                bn.bn_collapsed_cycles.cycle_records[tcsm_k].num_observations
                for tcsm_k in range(len(bn.bn_collapsed_cycles))
            ]
        )
    )
    plt.imshow(test_csm_3, cmap="plasma", vmin=tcsm_min, vmax=tcsm_max)
    plt.show()


def compare_sequences_of_all_nodes(bn: BooleanNetwork):
    print("Comparing the state-sequences of each node of each cycle")
    fig, axs = plt.subplots(1, 2)
    bn.bn_collapsed_cycles.sort_cycle_records_by_cycle_length()
    cycle_similarity_matrix = [
        [0 for csm_i in range(len(bn.bn_collapsed_cycles))]
        for csm_j in range(len(bn.bn_collapsed_cycles))
    ]
    for bn_i in range(len(bn.bn_collapsed_cycles)):
        for bn_j in range(len(bn.bn_collapsed_cycles)):
            cycle_similarity_matrix[bn_i][bn_j] = per_node_sequence_similarity(
                bn.bn_collapsed_cycles.cycle_records[bn_i],
                bn.bn_collapsed_cycles.cycle_records[bn_j],
                [csm_a for csm_a in range(len(bn))],
            )

    test_csm_1 = copy.deepcopy(cycle_similarity_matrix)
    tcsm_max = 0
    tcsm_min = 10
    for t_i in range(len(test_csm_1)):
        for t_j in range(len(test_csm_1)):
            if test_csm_1[t_i][t_j] > tcsm_max:
                tcsm_max = test_csm_1[t_i][t_j]
            if test_csm_1[t_i][t_j] < tcsm_min:
                tcsm_min = test_csm_1[t_i][t_j]
            if t_j < t_i:
                test_csm_1[t_i][t_j] = 0
    axs[0].imshow(test_csm_1, cmap="plasma", vmin=tcsm_min, vmax=tcsm_max)
    axs[0].set_title("Sorted by cycle length")  # Like post-condition
    print(
        "Sorted by cycle length:\n"
        + str(
            [
                len(bn.bn_collapsed_cycles.cycle_records[tcsm_k])
                for tcsm_k in range(len(bn.bn_collapsed_cycles))
            ]
        )
    )
    bn.sort_cycle_records_by_num_observations()
    cycle_similarity_matrix = [
        [0 for csm_i in range(len(bn.bn_collapsed_cycles))]
        for csm_j in range(len(bn.bn_collapsed_cycles))
    ]
    for bn_i in range(len(bn.bn_collapsed_cycles)):
        for bn_j in range(len(bn.bn_collapsed_cycles)):
            cycle_similarity_matrix[bn_i][bn_j] = per_node_sequence_similarity(
                bn.bn_collapsed_cycles.cycle_records[bn_i],
                bn.bn_collapsed_cycles.cycle_records[bn_j],
                [csm_a for csm_a in range(len(bn))],
            )

    test_csm_2 = copy.deepcopy(cycle_similarity_matrix)
    tcsm_max = 0
    tcsm_min = 10
    for t_i in range(len(test_csm_2)):
        for t_j in range(len(test_csm_2)):
            if test_csm_2[t_i][t_j] > tcsm_max:
                tcsm_max = test_csm_2[t_i][t_j]
            if test_csm_2[t_i][t_j] < tcsm_min:
                tcsm_min = test_csm_2[t_i][t_j]
            if t_j < t_i:
                test_csm_2[t_i][t_j] = 0
    axs[1].imshow(
        test_csm_2, cmap="plasma", vmin=tcsm_min, vmax=tcsm_max
    )  # Sorting by num observations makes a difference, add more sorts-- like, cycle length -- already have that, sick
    axs[1].set_title("By number of observations")  # Like post-condition
    print(
        "Sorted by number of observations:\n"
        + str(
            [
                bn.bn_collapsed_cycles.cycle_records[tcsm_k].num_observations
                for tcsm_k in range(len(bn.bn_collapsed_cycles))
            ]
        )
    )
    plt.show()
    #
    fig_2, axs_2 = plt.subplots(1, 2)
    bn.bn_collapsed_cycles.sort_cycle_records_by_hamming_diameter()
    cycle_similarity_matrix = [
        [0 for csm_i in range(len(bn.bn_collapsed_cycles))]
        for csm_j in range(len(bn.bn_collapsed_cycles))
    ]
    for bn_i in range(len(bn.bn_collapsed_cycles)):
        for bn_j in range(len(bn.bn_collapsed_cycles)):
            cycle_similarity_matrix[bn_i][bn_j] = per_node_sequence_similarity(
                bn.bn_collapsed_cycles.cycle_records[bn_i],
                bn.bn_collapsed_cycles.cycle_records[bn_j],
                [csm_a for csm_a in range(len(bn))],
            )
    test_csm_3 = copy.deepcopy(cycle_similarity_matrix)
    tcsm_max = 0
    tcsm_min = 10
    for t_i in range(len(test_csm_3)):
        for t_j in range(len(test_csm_3)):
            if test_csm_3[t_i][t_j] > tcsm_max:
                tcsm_max = test_csm_3[t_i][t_j]
            if test_csm_3[t_i][t_j] < tcsm_min:
                tcsm_min = test_csm_3[t_i][t_j]
            if t_j < t_i:
                test_csm_3[t_i][t_j] = 0
    axs_2[0].set_title("By maximum intra-cycle Hamm. distance")
    print("\nSorted by maximum intra-cycle Hamming distance")
    print(
        "Cycle length:\n"
        + str(
            [
                len(bn.bn_collapsed_cycles.cycle_records[tcsm_k])
                for tcsm_k in range(len(bn.bn_collapsed_cycles))
            ]
        )
    )
    print(
        "Number of observations:\n"
        + str(
            [
                bn.bn_collapsed_cycles.cycle_records[tcsm_k].num_observations
                for tcsm_k in range(len(bn.bn_collapsed_cycles))
            ]
        )
    )
    axs_2[0].imshow(test_csm_3, cmap="plasma", vmin=tcsm_min, vmax=tcsm_max)
    #
    # 08-23-2024 add + score for similarity of whether nodes are constant (regardless of their value) and maybe for changing or not also, but these additions must be 1 order of magnitude less than for similarity as defined as-is
    # ## Can also define this as each neighbor is that with highest similarity, and arrange them circularly, then present that cut at some arbitrary or non-arbitrary point, maybe at the longest cycle or shortest cycle
    # ### It might be that this brings up an objective function of making abunances similarites greater for two cycle states (where miR is expressed in both, or only in one) and cost is dissimilarity from the same for the goal state defining cycle, this would only be possible in certain cases, this brings up "off-target" effects
    # ### Tangentially: if predictions of knockdown from chimeric reads disagree with measured knockdown (1, is it because of RNA-activation? develop complementary method for RNA-DNA chimeras to check)
    bn.bn_collapsed_cycles.sort_by_cycle_similarity(
        bn.bn_collapsed_cycles.get_furthest_reference(
            bn.bn_collapsed_cycles.get_consensus_state()
        )
    )
    cycle_similarity_matrix = [
        [0 for csm_i in range(len(bn.bn_collapsed_cycles))]
        for csm_j in range(len(bn.bn_collapsed_cycles))
    ]
    for bn_i in range(len(bn.bn_collapsed_cycles)):
        for bn_j in range(len(bn.bn_collapsed_cycles)):
            cycle_similarity_matrix[bn_i][bn_j] = per_node_sequence_similarity(
                bn.bn_collapsed_cycles.cycle_records[bn_i],
                bn.bn_collapsed_cycles.cycle_records[bn_j],
                [csm_a for csm_a in range(len(bn))],
            )
    test_csm_3 = copy.deepcopy(cycle_similarity_matrix)
    tcsm_max = 0
    tcsm_min = 10
    for t_i in range(len(test_csm_3)):
        for t_j in range(len(test_csm_3)):
            if test_csm_3[t_i][t_j] > tcsm_max:
                tcsm_max = test_csm_3[t_i][t_j]
            if test_csm_3[t_i][t_j] < tcsm_min:
                tcsm_min = test_csm_3[t_i][t_j]
            if t_j < t_i:
                test_csm_3[t_i][t_j] = 0
    axs_2[1].set_title("By cycle similarity")
    print("\nSorted by cycle similarity")
    print(
        "Cycle length:\n"
        + str(
            [
                len(bn.bn_collapsed_cycles.cycle_records[tcsm_k])
                for tcsm_k in range(len(bn.bn_collapsed_cycles))
            ]
        )
    )
    print(
        "Number of observations:\n"
        + str(
            [
                bn.bn_collapsed_cycles.cycle_records[tcsm_k].num_observations
                for tcsm_k in range(len(bn.bn_collapsed_cycles))
            ]
        )
    )
    axs_2[1].imshow(test_csm_3, cmap="plasma", vmin=tcsm_min, vmax=tcsm_max)
    plt.show()


# 09-16-2024
def count_mirna_sites_non_1(
    micro_rna_abn: MicroRNAAbundantBooleanNetwork,
    list_of_target_sites: list[ThreePrimeUTRTargetSite],
):
    counted_by_mirna = [0 for cbm_i in range(len(micro_rna_abn.micro_rna_nodes))]
    for each_site in list_of_target_sites:
        counted_by_mirna[
            micro_rna_abn.micro_rna_nodes.index(each_site.micro_rna_node_reference)
        ] += 1
    return sum(
        [
            0 if counted_by_mirna[cbm_j] == 1 else 1
            for cbm_j in range(len(counted_by_mirna))
        ]
    )


def mutate_and_return(
    micro_rna_abn: MicroRNAAbundantBooleanNetwork,
    input_abundant_node: AbundantNode,
    mutation_prob: float,
):
    swap_input_abundant_node = copy.deepcopy(input_abundant_node)
    starting_sequence = swap_input_abundant_node.three_prime_utr
    mutated_sequence = ""
    for each_nt in starting_sequence:
        if SystemRandom().random() < mutation_prob:
            mutated_sequence += get_random_sequence(1, ["A", "U", "G", "C"])
        else:
            mutated_sequence += each_nt
    swap_input_abundant_node.three_prime_utr = mutated_sequence
    swap_input_abundant_node.target_sites = get_three_prime_utr_target_sites(
        micro_rna_abn.micro_rna_nodes, swap_input_abundant_node
    )
    if count_mirna_sites_non_1(
        micro_rna_abn, input_abundant_node.target_sites
    ) > count_mirna_sites_non_1(micro_rna_abn, swap_input_abundant_node.target_sites):
        return input_abundant_node
    else:
        return swap_input_abundant_node


def count_lengths_target_sites(input_target_sites: list[ThreePrimeUTRTargetSite]):
    return sum(
        [
            input_target_sites[clts_i].end_index
            - input_target_sites[clts_i].start_index
            for clts_i in range(len(input_target_sites))
        ]
    )


def mutate_and_return_2(
    micro_rna_abn: MicroRNAAbundantBooleanNetwork,
    input_abundant_node: AbundantNode,
    mutation_prob: float,
):
    # add if over maximum number target sites select for decrease, and always that which increases efficacy of target sites (?)
    swap_input_abundant_node = copy.deepcopy(input_abundant_node)
    starting_sequence = swap_input_abundant_node.three_prime_utr
    mutated_sequence = ""
    for each_nt in starting_sequence:
        if SystemRandom().random() < mutation_prob:
            mutated_sequence += get_random_sequence(1, ["A", "U", "G", "C"])
        else:
            mutated_sequence += each_nt
    swap_input_abundant_node.three_prime_utr = mutated_sequence
    swap_input_abundant_node.target_sites = get_three_prime_utr_target_sites(
        micro_rna_abn.micro_rna_nodes, swap_input_abundant_node
    )
    if (
        count_mirna_sites_non_1(micro_rna_abn, input_abundant_node.target_sites)
        > count_mirna_sites_non_1(micro_rna_abn, swap_input_abundant_node.target_sites)
    ) or count_lengths_target_sites(swap_input_abundant_node.target_sites) / (
        len(swap_input_abundant_node.target_sites) + 1
    ) < count_lengths_target_sites(
        input_abundant_node.target_sites
    ) / (
        len(input_abundant_node.target_sites) + 1
    ):
        return input_abundant_node.three_prime_utr
    else:
        return swap_input_abundant_node.three_prime_utr


def mutate_and_return_3(
    micro_rna_abn: MicroRNAAbundantBooleanNetwork,
    input_abundant_node: AbundantNode,
    mutation_prob: float,
):  # 09-04-2024
    # applies mutagenesis over windows of sequence and selects at each window
    abundant_node_copy = copy.deepcopy(input_abundant_node)
    for nt_i in range(len(input_abundant_node.three_prime_utr) - 8):
        swap_abundant_node = copy.deepcopy(abundant_node_copy)
        starting_sequence = swap_abundant_node.three_prime_utr
        mutated_sequence = ""
        for nt_j in range(8):
            if SystemRandom().random() < mutation_prob:
                mutated_sequence += get_random_sequence(1, ["A", "U", "G", "C"])
            else:
                mutated_sequence += starting_sequence[nt_i : nt_i + nt_j]
        swap_abundant_node.three_prime_utr = (
            starting_sequence[0:nt_i]
            + mutated_sequence
            + starting_sequence[nt_i + len(mutated_sequence) : len(starting_sequence)]
        )
        # this could be simplified to a search per windowed sequence
        swap_abundant_node.target_sites = get_three_prime_utr_target_sites(
            micro_rna_abn.micro_rna_nodes, swap_abundant_node
        )
        if (
            count_mirna_sites_non_1(micro_rna_abn, input_abundant_node.target_sites)
            > count_mirna_sites_non_1(micro_rna_abn, swap_abundant_node.target_sites)
        ) or count_lengths_target_sites(swap_abundant_node.target_sites) / (
            len(swap_abundant_node.target_sites) + 1
        ) < count_lengths_target_sites(
            input_abundant_node.target_sites
        ) / (
            len(input_abundant_node.target_sites) + 1
        ):
            abundant_node_copy = swap_abundant_node
    return abundant_node_copy


def plot_target_sites(
    a_bn_mir: MicroRNAAbundantBooleanNetwork, colors: list, plot_title: str = ""
):
    """assumes that target sites are set up"""
    # a_bn_mir.setup_target_site_lists()
    if plot_title == "":
        plot_count = 0
        for sites in [node.target_sites for node in a_bn_mir.get_plottable_nodes()]:
            for each_target_site in sites:
                print(each_target_site)
                plot_count += 1
                if plot_count > 10:
                    return
                print("\n")
    else:
        brick = [
            188,
            74,
            50,
        ]  # From Google, pallettemaker.com, adjusting the blue down a bit, more like what I was thinking (diagram from earlier in process) (was 60)
        alpha_list = [
            0.01,
            0.01,
            0.01,
            0.01,
            0.04,
            0.4,
            0.6,
            0.8,
            0.85,
            0.85,
            0.85,
            0.85,
            0.85,
            0.85,
            0.85,
            0.85,
            0.85,
            0.85,
            0.85,
            0.85,
            0.85,
            0.85,
            0.85,
            0.85,
        ]  # Later, make alpha values correspond to number of base pairs with gaps in total mir-target  [should something like this be effects vector (starting point for fitting in model) ?]
        scaling_factor = 10
        fig = plt.figure()
        ax = plt.subplot()
        sequence_set = set()
        counter = 0
        target_sites_to_print = []
        index_of_abundant_node_to_print = 0
        list_list_eff_sites = (
            []
        )  # This can accomplish some of ... ensemble calculation, where, if there are a bunch of competing sites the impact of each on targeted abundance is decreased
        for plot_i, each_abundant_node in zip(
            range(a_bn_mir.get_number_of_abundances()), a_bn_mir.get_plottable_nodes()
        ):
            plot_y = 1 + plot_i * 2
            box_x_1 = -10
            # if ((len(each_abundant_node.target_sites) == 4) or (plot_i == a_bn_mir.get_number_of_abundances() - 1)) and target_sites_to_print is None:
            target_sites_to_print.extend(each_abundant_node.target_sites)
            index_of_abundant_node_to_print = plot_i
            box_x_2 = len(each_abundant_node.three_prime_utr) + 10
            box_y_1 = plot_y - 1
            box_y_2 = plot_y + 1
            ax.plot(
                [box_x_1, box_x_1],
                [box_y_1, box_y_2],
                linestyle="dashed",
                color="k",
                alpha=0.5,
            )
            ax.plot(
                [box_x_1, box_x_2],
                [box_y_2, box_y_2],
                linestyle="dashed",
                color="k",
                alpha=0.5,
            )
            ax.plot(
                [box_x_2, box_x_2],
                [box_y_2, box_y_1],
                linestyle="dashed",
                color="k",
                alpha=0.5,
            )
            ax.plot(
                [box_x_1, box_x_2],
                [box_y_1, box_y_1],
                linestyle="dashed",
                color="k",
                alpha=0.5,
            )
            ax.plot(
                [0, len(each_abundant_node.three_prime_utr)],
                [plot_y, plot_y],
                color="k",
                alpha=0.4,
            )
            ax.scatter(2 * box_x_1, plot_y, color=colors[plot_i], marker=".")
            for plot_j in range(len(each_abundant_node.target_sites)):
                sequence_set.add(
                    each_abundant_node.target_sites[
                        plot_j
                    ].micro_rna_node_reference.sequence
                )
                counter += 1
                x_1 = each_abundant_node.target_sites[plot_j].start_index
                x_2 = each_abundant_node.target_sites[plot_j].end_index
                avg_x = (x_1 + x_2) / 2
                site_size = (x_2 - x_1) * scaling_factor
                tick_color = colors[
                    a_bn_mir.micro_rna_nodes.index(
                        each_abundant_node.target_sites[plot_j].micro_rna_node_reference
                    )
                ]
                plt.scatter(
                    avg_x,
                    plot_y,
                    alpha=alpha_list[x_2 - x_1],
                    marker="|",
                    s=site_size**1.2,
                    color=tick_color,
                )
        plt.title(plot_title)
        fig.set_size_inches(6, 9)
        plt.yticks([])
        plt.xlabel("nt")
        plt.show()  # Maybe use CycleRecord to track number of observations of each sequence
        #
        print("Target sites of boxed 3'-UTR sequence\n")
        for each_target_site in target_sites_to_print:
            print(each_target_site)
            print()


def figure_1_1(network: BooleanNetwork, index: int = 0):
    network.bn_collapsed_cycles.sort_cycle_records_by_cycle_length()
    fig_1_current_states_list = network.bn_collapsed_cycles.cycle_records[
        index
    ].cycle_states_list
    test_bn_string_list = [
        str(f'{"node " + str(tbn_k + 1) + ": ":>11s}')
        for tbn_k in range(len(fig_1_current_states_list[0]))
    ]  # one for each node
    for (
        each_list_1
    ) in fig_1_current_states_list:  # each list is of length number of nodes
        for tbn_l in range(len(fig_1_current_states_list[0])):  # for each node
            # ## From Python documentation: f'result: {value:{width}.{precision}}' ##
            test_bn_string_list[tbn_l] += str(f"{str(int(each_list_1[tbn_l])):^5s}")
            if each_list_1 is fig_1_current_states_list[-2]:
                test_bn_string_list[tbn_l] += "|"
            else:
                test_bn_string_list[tbn_l] += " "
    print(
        f"{get_ordinal_string(network.bn_collapsed_cycles.get_index(each_list_1), True)} cycle\n"
    )
    print("\n".join(test_bn_string_list))


def figure_1_2(boolean_network: BooleanNetwork, cycle_record_index: int):
    # bv_1 = get_boolean_velocity_list(boolean_network.bn_collapsed_cycles.cycle_records[cycle_record_index])
    print(
        "1: node state changed, 0: node state did not change:\n"
    )  # could use == != or is, not
    print(
        get_boolean_velocity_list(
            boolean_network.bn_collapsed_cycles.cycle_records[cycle_record_index]
        )[0]
    )
    # print("Figure 1.2: Text output of node state changes over the same cycle as in Figure 1.1:\n")
    # print("1: node state changed, 0: node state did not change:\n")  # could use == != or is, not
    # bv_string_list = [str(f'{"node " + str(tbn_k + 1):>11s}: ') for tbn_k in range(len(bv_1[0]))]
    # bv_total_string = "\n" + str(f'{"total: ":>11s}: ')
    # for each_list_1 in bv_1:
    #     for tbn_l in range(len(bv_1[0])):
    #         bv_string_list[tbn_l] += str(f'{str(int(each_list_1[tbn_l])):^11s}')
    #     bv_total_string += str(f'{str(sum(each_list_1)):^11s}')
    # bv_string_list = [bv_string_list[add_mark_i] + "|" for add_mark_i in range(len(bv_string_list))]
    # print("\n".join(bv_string_list))
    # print(bv_total_string)
    # #
    # constant_node_index_list = []
    # for bv_L in range(len(bv_1[0])):
    #     bv_bool_1 = True
    #     for bv_m in range(len(bv_1)):
    #         if bv_1[bv_m][bv_L] != bv_1[0][bv_L]:
    #             bv_bool_1 = False
    #             break
    #     if bv_bool_1:
    #         constant_node_index_list.append(bv_L)
    # print("\nNodes with constant state in this cycle: " + str([constant_node_index_list[bv_n] + 1 for bv_n in range(
    #     len(constant_node_index_list))]))  # Ordinal reference to nodes ...
    # print("\nNumber of inputs per constant node: " + str(
    #     [boolean_network.nodes[constant_node_index_list[bv_o]].function.get_k() for bv_o in
    #      range(len(constant_node_index_list))]))
    # print("\nInputs per constant node: " + str(
    #     [boolean_network.node_inputs_assignments[constant_node_index_list[bv_o]] for bv_o in
    #      range(len(constant_node_index_list))]))


def figure_2_1(boolean_network: BooleanNetwork):
    bv_colors = get_colors(len(boolean_network.list_of_all_run_ins), True)
    print(
        "Run-in steps vs. number of node state changes over all run-ins, highlighting run-ins from homogeneous initial conditions"
    )
    print(
        "After run-in's from random states initial conditions, and one run-in from each homogeneous initial condition ("
        + str(len(boolean_network.list_of_all_run_ins))
        + " total run-ins),"
    )  # Or, plot first 202
    print("Number of observed cycles: " + str(len(boolean_network.bn_collapsed_cycles)))
    print("Average cycle length: " + str(boolean_network.get_avg_cycle_length()))
    print()  # Could also look at the average node state over some set of synchronized run-ins, and this could be meaningful for a particular network, and, various motifs
    print(
        "The two run-ins from homogeneous conditions terminate in cycles "
        + str(
            boolean_network.bn_collapsed_cycles.get_index(
                boolean_network.list_of_all_run_ins[0][-1]
            )
        )
        + " (from all False), and "
        + str(
            boolean_network.bn_collapsed_cycles.get_index(
                boolean_network.list_of_all_run_ins[1][-1]
            )
        )
        + " (from all True), and are depicted in black (dot-dashed: from all False, dashed: from all True)."
    )  # as a scatter plot, not sure how informative (maybe if colored by terminal cycle...)
    print(
        "\n(The number of nodes changed per step is equal to the Hamming distance from the preceding step.)\n"
    )
    bv_run_in_list_1 = [
        get_boolean_velocity_list(boolean_network.list_of_all_run_ins[bv_k])
        for bv_k in range(len(boolean_network.list_of_all_run_ins))
    ]
    bv_summed_run_in_list_1 = []
    hist_y = []
    #
    fig_1, axs_1 = plt.subplots(
        2, 2, sharex="col", sharey="row", width_ratios=[4, 1], height_ratios=[4, 1]
    )  # fig_2, axs_2 = plt.subplots(num_row, num_col)
    fig_1.set_size_inches(8, 8)
    color_total = 0
    color_sum = [0, 0, 0]
    for bv_j in range(2, len(bv_run_in_list_1)):
        bv_summed_run_in_list_1.append(
            [
                sum(bv_run_in_list_1[bv_j][bv_k])
                for bv_k in range(len(bv_run_in_list_1[bv_j]))
            ]
        )
        axs_1[0, 0].plot(
            [bv_L for bv_L in range(len(bv_summed_run_in_list_1[-1]))],
            bv_summed_run_in_list_1[-1],
            alpha=0.15,
            color=bv_colors[bv_j],
            linewidth=1,
            linestyle=(0, (3, 3)),
        )
        axs_1[0, 0].scatter(
            [bv_L for bv_L in range(len(bv_summed_run_in_list_1[-1]))],
            bv_summed_run_in_list_1[-1],
            alpha=0.2,
            color=bv_colors[bv_j],
            marker=".",
            s=2,
        )
        color_sum[0] += bv_colors[bv_j][0]
        color_sum[1] += bv_colors[bv_j][1]
        color_sum[2] += bv_colors[bv_j][2]
        color_total += 1
        axs_1[0, 0].scatter(
            len(bv_summed_run_in_list_1[-1]) - 1,
            bv_summed_run_in_list_1[-1][-1],
            marker="|",
            alpha=0.25,
            color=bv_colors[bv_j],
            s=80,
        )
    for hy_i in range(len(bv_summed_run_in_list_1)):
        for hy_j in range(len(bv_summed_run_in_list_1[hy_i])):
            hist_y.append(bv_summed_run_in_list_1[hy_i][hy_j])
    # bv_summed_run_in_list.append()  # keep first two run ins in the total data
    axs_1[0, 0].plot(
        [bv_n for bv_n in range(len(bv_summed_run_in_list_1[0]))],
        bv_summed_run_in_list_1[0],
        color="w",
        alpha=0.25,
        linestyle="-",
        marker=".",
        linewidth=3,
    )
    axs_1[0, 0].plot(
        [bv_o for bv_o in range(len(bv_summed_run_in_list_1[1]))],
        bv_summed_run_in_list_1[1],
        color="w",
        alpha=0.25,
        linestyle="-",
        marker=".",
        linewidth=3,
    )
    axs_1[0, 0].plot(
        [bv_n for bv_n in range(len(bv_summed_run_in_list_1[0]))],
        bv_summed_run_in_list_1[0],
        color="k",
        alpha=0.85,
        linestyle="-.",
        marker=".",
    )
    axs_1[0, 0].plot(
        [bv_o for bv_o in range(len(bv_summed_run_in_list_1[1]))],
        bv_summed_run_in_list_1[1],
        color="k",
        alpha=0.85,
        linestyle="--",
        marker=".",
    )
    axs_1[0, 0].scatter(
        len(bv_summed_run_in_list_1[0]) - 1,
        bv_summed_run_in_list_1[0][-1],
        marker="$"
        + str(
            boolean_network.bn_collapsed_cycles.get_index(
                boolean_network.list_of_all_run_ins[0][-1]
            )
        )
        + "$",
        color="k",
        s=200,
    )
    axs_1[0, 0].scatter(
        len(bv_summed_run_in_list_1[1]) - 1,
        bv_summed_run_in_list_1[1][-1],
        marker="$"
        + str(
            boolean_network.bn_collapsed_cycles.get_index(
                boolean_network.list_of_all_run_ins[1][-1]
            )
        )
        + "$",
        color="k",
        s=200,
    )  # "|"
    axs_1[0, 0].set_xlabel("steps")
    axs_1[0, 0].set_ylabel("step to step edit distance")
    axs_1[0, 0].set_ylim(0, 22)
    axs_1[0, 0].set_yticks([bv_y for bv_y in range(0, 22, 2)])
    axs_1[0, 0].set_title("Node state changes per step for all run-ins", loc="left")
    #
    bv_lengths = [
        len(bv_summed_run_in_list_1[bv_i])
        for bv_i in range(len(bv_summed_run_in_list_1))
    ]
    bv_steps_summed = [0 for bv_j in range(max(bv_lengths))]
    bv_obs_per_step = [[] for bv_w in range(max(bv_lengths))]
    bv_sample_n_per_step = bv_steps_summed.copy()
    for bv_m in range(len(bv_summed_run_in_list_1)):
        for bv_p in range(len(bv_summed_run_in_list_1[bv_m])):
            bv_steps_summed[bv_p] += bv_summed_run_in_list_1[bv_m][bv_p]
            bv_obs_per_step[bv_p].append(bv_summed_run_in_list_1[bv_m][bv_p])
            bv_sample_n_per_step[bv_p] += 1
    bv_avg = [
        bv_steps_summed[bv_q] / bv_sample_n_per_step[bv_q]
        for bv_q in range(len(bv_steps_summed))
    ]
    axs_1[1, 0].plot(bv_avg, color="k")
    for bv_error_x in range(len(bv_avg)):
        axs_1[1, 0].errorbar(
            bv_error_x,
            bv_avg[bv_error_x],
            np.std(bv_obs_per_step[bv_error_x]),
            color=[0.6, 0.6, 0.6],
        )
    axs_1[1, 0].set_ylabel("step to step edit distance")
    axs_1[1, 0].set_xlabel("steps")
    axs_1[1, 0].set_title("Average per step, aligned from first step, with st. dev.")
    axs_1[0, 1].set_xlabel("obs.")
    axs_1[0, 1].set_ylabel("per step node changes")
    axs_1[0, 1].hist(
        hist_y,
        orientation="horizontal",
        rwidth=0.5,
        color=[
            color_sum[0] / color_total,
            color_sum[2] / color_total,
            color_sum[2] / color_total,
        ],
        bins=22,
    )
    plt.show()


def figure_2_2(
    boolean_network: BooleanNetwork,
    scale_to_least_common_multiple_if_true: bool,
    colors: list,
):
    print(
        "Figure 2.2: Cycle steps vs. number of node state changes over all cycles, cycle lengths are marked by "
        + "vertical tick marks"
    )
    bv_cycles_list = [
        get_boolean_velocity_list(
            boolean_network.bn_collapsed_cycles.cycle_records[bvc_i].cycle_states_list
        )
        for bvc_i in range(len(boolean_network.bn_collapsed_cycles))
    ]
    bv_summed_run_in_list_2 = []
    if not scale_to_least_common_multiple_if_true:
        num_steps = boolean_network.longest_cycle_length() + 1
    else:
        num_steps = (
            get_least_common_multiple(
                [
                    len(boolean_network.bn_collapsed_cycles.cycle_records[f22_i])
                    for f22_i in range(len(boolean_network.bn_collapsed_cycles))
                ]
            )
            + 1
        )
    for bv_j in range(len(boolean_network.bn_collapsed_cycles)):
        bv_summed_run_in_list_2 = [
            sum(bv_cycles_list[bv_j][bv_k % len(bv_cycles_list[bv_j])])
            for bv_k in range(num_steps)
        ]
        plt.plot(
            [bv_L for bv_L in range(num_steps)],
            bv_summed_run_in_list_2,
            alpha=0.5,
            linestyle="--",
            color=colors[
                int(len(colors) * bv_j / len(boolean_network.bn_collapsed_cycles))
            ],
            linewidth=1,
            marker="|",
            markevery=len(boolean_network.bn_collapsed_cycles.cycle_records[bv_j]),
            label=str(bv_j),
        )
    plt.xlabel("steps")
    plt.ylabel("step to step edit distance")
    plt.ylim(-1, 22)
    plt.yticks([bv_y for bv_y in range(0, 22, 2)])
    plt.title("Figure 2:", loc="left")
    if len(boolean_network.bn_collapsed_cycles) < 30:
        plt.figlegend()
    plt.show()  # TODO Denote maximum cycle lengths in legend (calculate maximum (maybe add method to BN), then for each
    #                   legend entry, if its length is == maximum length, say so)
    # Equilibrial states need to display properly, a tick mark each step -> are, but at 0, alter plot limits ### Figure
    # legend have cycle lengths also, for all cycles


def get_lambda_consensus_list(sequence_1: list, sequence_2: list):
    return [
        lambda p: (p == sequence_1[glcl_i]) or (p == sequence_2[glcl_i])
        for glcl_i in range(len(sequence_1))
    ]


def get_lambda_distance(sequence: list, lambda_list: list):
    return 1  # TODO Finish this


# TODO For two sequences, lambda expressions of both sequence elements for each element, and a (p == seq_element 1) or
#  (p == True seq_element 2)


class HammingSetup:
    def __init__(self, boolean_network: BooleanNetwork):
        # Setup for Hamming distances, currently: arbitrary, TODO determine these principal components (?) by some measurement of the resulting projection  ### TODO consider "one-way" or masked sequences Hamming distances (one way would be, if the ref sequence has a 1, but query sequence has a 0, 0, but if ref sequence has a 0 and query sequence has a 1, 1 ???)  ### TODO sort random sequences by Hamming distance from 0th sequence?
        #
        boolean_network_cycle_states = []
        for each_cycle_record in boolean_network.bn_collapsed_cycles.cycle_records:
            for hamm_i in range(len(each_cycle_record)):
                boolean_network_cycle_states.append(
                    each_cycle_record.cycle_states_list[hamm_i]
                )
        boolean_network_consensus_cycle_state = get_consensus_sequence_list(
            boolean_network_cycle_states, [False, True]
        )
        boolean_network_hamm_run_ins = []
        for hamm_j in range(2, len(boolean_network.list_of_all_run_ins)):
            boolean_network_hamm_run_ins.append(
                [
                    get_hamming_distance(
                        boolean_network_consensus_cycle_state,
                        boolean_network.list_of_all_run_ins[hamm_j][hamm_k],
                    )
                    for hamm_k in range(
                        len(boolean_network.list_of_all_run_ins[hamm_j])
                    )
                ]
            )
        #
        boolean_network_1_states = []
        maximum_number_observations = max(
            [
                boolean_network.bn_collapsed_cycles.cycle_records[
                    net_i
                ].num_observations
                for net_i in range(len(boolean_network.bn_collapsed_cycles))
            ]
        )
        selected_cycle_1 = boolean_network.bn_collapsed_cycles.get_reference(
            boolean_network.list_of_all_run_ins[0][-1]
        )
        for num_i in range(1, len(boolean_network.bn_collapsed_cycles.cycle_records)):
            if (
                boolean_network.bn_collapsed_cycles.cycle_records[
                    num_i
                ].num_observations
                > maximum_number_observations
            ):
                maximum_number_observations = (
                    boolean_network.bn_collapsed_cycles.cycle_records[
                        num_i
                    ].num_observations
                )
                selected_cycle_1 = boolean_network.bn_collapsed_cycles.cycle_records[
                    num_i
                ]
        for hamm_i in range(len(selected_cycle_1)):
            boolean_network_1_states.append(selected_cycle_1.cycle_states_list[hamm_i])
        boolean_network_consensus_selected_cycle_state = get_consensus_sequence_list(
            boolean_network_1_states, [False, True]
        )
        #
        boolean_network_hamm_run_ins = []
        for hamm_j in range(2, len(boolean_network.list_of_all_run_ins)):
            boolean_network_hamm_run_ins.append(
                [
                    get_hamming_distance(
                        boolean_network_consensus_selected_cycle_state,
                        boolean_network.list_of_all_run_ins[hamm_j][hamm_k],
                    )
                    for hamm_k in range(
                        len(boolean_network.list_of_all_run_ins[hamm_j])
                    )
                ]
            )
        num_random_sequences = 15
        self.first_sequences = [boolean_network_consensus_selected_cycle_state] + [
            get_random_list(len(boolean_network), [False, True])
            for fig_143_i in range(num_random_sequences)
        ]
        # second_sequences = [list(reversed(get_boolean_complementary_sequence_list(boolean_network_consensus_cycle_state)))] + [get_random_list(len(boolean_network), [False, True]) for fig_143_j in range(num_random_sequences)]
        # reverse complement of consensus was effectively a defined random sequence [but complement might make sense]
        # ... check [[09-13-2024 relative Hamming distance]]
        self.second_sequences = [
            [False for fig_143_i in range(len(boolean_network))]
        ] + [
            get_random_list(len(boolean_network), [False, True])
            for fig_143_j in range(num_random_sequences)
        ]
        self.linear_function = [1 for fig_143_k in range(len(self.first_sequences))]
        for fig_143_L in range(len(self.linear_function)):
            for fig_143_m in range(fig_143_L):
                self.linear_function[fig_143_L] = self.linear_function[fig_143_L] / (
                    0.5 * math.e
                )
            if fig_143_L > 0:
                self.linear_function[fig_143_L] *= 0.5
            if fig_143_L > 4:
                self.linear_function[fig_143_L] *= 0.5
            self.linear_function[fig_143_L] = max(0.01, self.linear_function[fig_143_L])

    def get_x_y(self, state: list[bool]) -> tuple[float, float]:
        x = sum(
            [
                self.linear_function[fig_143_n]
                * get_hamming_distance(self.first_sequences[fig_143_n], state)
                for fig_143_n in range(len(self.linear_function))
            ]
        )
        y = sum(
            [
                self.linear_function[fig_143_o]
                * get_hamming_distance(self.second_sequences[fig_143_o], state)
                for fig_143_o in range(len(self.linear_function))
            ]
        )
        return (x, y)


def figure_2_3(
    boolean_network: BooleanNetwork, hamming_setup: HammingSetup, bv_colors: list
):
    set_of_state_strings = set()
    selected_run_in_states = []
    print("Figure 3.1: 2D plot of observed cycles, and homogeneous conditions")
    print()
    print(
        "Equilibrial states are represented as a +, and non-equilibrial states are represented by a points connected "
        + "by a dotted line"
    )  # (though no intermediate states are calculated between points, we could imagine that these " +
    # "node state changes occur in a random or characteristic order for each step, or each particular state change," +
    # " however, we could also apply this to equilibrial states, and this may bring up some of the philosophical " +
    # "problems one could have with synchronous updating of such a network, though TBD for any scheme of " +
    # "asynchronous updating and continuous causal relationships you could define binary states and ?). Grey " +
    # "circles represent homogeneous initial conditions in this Hamming space.")  # this is actually not true, these
    # behave like a net with a random node in terms of finite state machine behavior (they're not finite state machines)
    # if it were that update order was fixed or deterministic, they'd be finite state machines, and some relation
    # to a BN would exist
    print(
        "If lines appear solid, this is because a dotted line going to and from a point overlap in such a way as to"
        + " appear solid."
    )
    print(
        "\nAxis 1 (x): a linear combination of the Hamming distances to a first set of sequences \n(the consensus of"
        + " the most-observed cycle, and 15 random sequences).\nAxis 2 (y): a linear combination of the Hamming "
        + "distance to a second set of sequences \n(all False, and 15 random sequences).\n"
    )
    plt.figure()
    for each_run_in_state in selected_run_in_states:
        each_x = sum(
            [
                hamming_setup.linear_function[fig_143_n]
                * get_hamming_distance(
                    hamming_setup.first_sequences[fig_143_n], each_run_in_state
                )
                for fig_143_n in range(len(hamming_setup.linear_function))
            ]
        )
        each_y = sum(
            [
                hamming_setup.linear_function[fig_143_o]
                * get_hamming_distance(
                    hamming_setup.second_sequences[fig_143_o], each_run_in_state
                )
                for fig_143_o in range(len(hamming_setup.linear_function))
            ]
        )
        plt.scatter(each_x, each_y, color=[0.9, 0.9, 0.9], alpha=1)
    max_hamm_x = 0
    min_hamm_x = 1000
    max_hamm_y = 0
    min_hamm_y = 1000
    all_panels_x = []
    all_panels_y = []
    #
    for each_cycle_record_1 in boolean_network.bn_collapsed_cycles.cycle_records:
        each_x = []
        each_y = []
        all_panels_x.append([])
        all_panels_y.append([])
        for each_cycle_state in each_cycle_record_1.cycle_states_list:
            each_x.append(
                sum(
                    [
                        hamming_setup.linear_function[fig_143_n]
                        * get_hamming_distance(
                            hamming_setup.first_sequences[fig_143_n], each_cycle_state
                        )
                        for fig_143_n in range(len(hamming_setup.linear_function))
                    ]
                )
            )
            each_y.append(
                sum(
                    [
                        hamming_setup.linear_function[fig_143_o]
                        * get_hamming_distance(
                            hamming_setup.second_sequences[fig_143_o], each_cycle_state
                        )
                        for fig_143_o in range(len(hamming_setup.linear_function))
                    ]
                )
            )
            all_panels_x[-1].append(each_x[-1])
            all_panels_y[-1].append(each_y[-1])
        if len(each_x) == 2:
            plt.scatter(
                each_x,
                each_y,
                color=bv_colors[
                    int(
                        len(bv_colors)
                        * boolean_network.bn_collapsed_cycles.cycle_records.index(
                            each_cycle_record_1
                        )
                        / len(boolean_network.bn_collapsed_cycles)
                    )
                ],
                alpha=0.9,
                label=str(
                    boolean_network.bn_collapsed_cycles.cycle_records.index(
                        each_cycle_record_1
                    )
                ),
                s=25,
                marker="P",
            )
        else:
            plt.plot(
                each_x,
                each_y,
                color=bv_colors[
                    int(
                        len(bv_colors)
                        * boolean_network.bn_collapsed_cycles.cycle_records.index(
                            each_cycle_record_1
                        )
                        / len(boolean_network.bn_collapsed_cycles)
                    )
                ],
                alpha=0.9,
                label=str(
                    boolean_network.bn_collapsed_cycles.cycle_records.index(
                        each_cycle_record_1
                    )
                ),
                linestyle="--",
                marker=".",
                linewidth=0.75,
            )
        if max(each_x) > max_hamm_x:
            max_hamm_x = max(each_x)
        if min(each_x) < min_hamm_x:
            min_hamm_x = min(each_x)
        if max(each_y) > max_hamm_y:
            max_hamm_y = max(each_y)
        if min(each_y) < min_hamm_y:
            min_hamm_y = min(each_y)
    max_hamm_y *= 1.05
    min_hamm_y /= 1.05
    max_hamm_x *= 1.05
    min_hamm_x /= 1.05
    # plt.xlim(min_hamm_x, max_hamm_x)
    # plt.ylim(min_hamm_y, max_hamm_y)
    plt.xlabel(
        "Hamming distance to consensus state of most-observed cycle and random states"
    )  # TODO shouldn't it be the least observed cycle?
    plt.ylabel("Hamming distance to all-False state and random states")
    if len(boolean_network.bn_collapsed_cycles) < 25:
        plt.figlegend(title="Cycle indices", loc="center right")
    else:
        print("Over 25 cycle were detected")
    # axs_1[1].set_xlabel("linear function index")
    # axs_1[1].set_ylabel("scalar")
    # axs_1[1].set_xticks([fig_143_n for fig_143_n in range(len(hamming_setup.linear_function))])
    # axs_1[1].scatter([lf_i for lf_i in range(len(hamming_setup.linear_function))], hamming_setup.linear_function, marker='+', color='k')
    # axs_1[1].legend(labels=["Hamming distance weights"])
    print(
        "Weights of Hamming distances: "
        + str(
            [
                round(hamming_setup.linear_function[lf_i], 4)
                for lf_i in range(len(hamming_setup.linear_function))
            ]
        )
        + "\n"
    )
    plt.show()
    num_col = 4
    num_row = int(len(boolean_network.bn_collapsed_cycles) / num_col)
    if len(boolean_network.bn_collapsed_cycles) / 4 - num_row > 0.0001:
        num_row += 1
    fig_2, axs_2 = plt.subplots(num_row, num_col)
    fig_2.set_size_inches(12, 3 * num_row)
    fig_2.suptitle("Figure 3.2: Each cycle of Figure 3.1, same axes as Figure 3.1")
    fig_2.supxlabel("Hamming distance to consensus cycle state and random states")
    fig_2.supylabel("Hamming distance to all-False state and random states")
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs_2.flat:
        ax.label_outer()
    each_cycle_index = 0
    for fig_panels_i in range(num_row):
        for fig_panels_j in range(num_col):
            if num_col * fig_panels_i + fig_panels_j < len(
                boolean_network.bn_collapsed_cycles
            ):
                axs_2[fig_panels_i, fig_panels_j].set_title(
                    f"Cycle: {num_col * fig_panels_i + fig_panels_j} "
                    + str(
                        boolean_network.bn_collapsed_cycles.cycle_records[
                            num_col * fig_panels_i + fig_panels_j
                        ].num_observations
                    )
                )
                axs_2[fig_panels_i, fig_panels_j].set_xlim(min_hamm_x, max_hamm_x)
                axs_2[fig_panels_i, fig_panels_j].set_ylim(min_hamm_y, max_hamm_y)
                if len(all_panels_x[num_col * fig_panels_i + fig_panels_j]) < 3:
                    axs_2[fig_panels_i, fig_panels_j].scatter(
                        all_panels_x[num_col * fig_panels_i + fig_panels_j],
                        all_panels_y[num_col * fig_panels_i + fig_panels_j],
                        color=bv_colors[
                            int(
                                len(bv_colors)
                                * each_cycle_index
                                / len(boolean_network.bn_collapsed_cycles)
                            )
                        ],
                        marker="P",
                    )
                    each_cycle_index += 1
                else:
                    axs_2[fig_panels_i, fig_panels_j].plot(
                        all_panels_x[num_col * fig_panels_i + fig_panels_j],
                        all_panels_y[num_col * fig_panels_i + fig_panels_j],
                        color=bv_colors[
                            int(
                                len(bv_colors)
                                * each_cycle_index
                                / len(boolean_network.bn_collapsed_cycles)
                            )
                        ],
                        linestyle="--",
                        marker=".",
                        linewidth=0.75,
                    )
                    each_cycle_index += 1
    plt.show()
    # plt.title("TEST" + "".join([" " for fig_space_i in range(50)]))


def figure_3_1(
    boolean_network: BooleanNetwork,
):  # ca 01 26 2025- better solution: use sympy and matrix output
    # Runtime: ~30 seconds
    boolean_network.compute_unit_perturbations_matrix()
    print(
        "Figure 2.3: Transitions resulting from unit perturbations of all observed cycle states of "
        + str(boolean_network)
    )
    print(
        "Number of cycle perturbations records: "
        + str(len(boolean_network.cycles_unit_perturbations_records))
        + " == "
        + str(
            sum(
                [
                    len(boolean_network.bn_collapsed_cycles.cycle_records[up_n])
                    for up_n in range(len(boolean_network.bn_collapsed_cycles))
                ]
            )
        )
        + " * "
        + str(len(boolean_network))
        + "(= sum(length of all cycles) * number of nodes)\n"
    )
    print("The row represents the starting cycle index\n")
    print("The column represents the ending cycle index\n")
    unit_perturbations_strings_list_1 = [
        ""
        for up_i in range(
            len(boolean_network.cycles_unit_perturbations_transition_matrix)
        )
    ]
    up_index_labels = [
        str(up_k)
        for up_k in range(
            len(boolean_network.cycles_unit_perturbations_transition_matrix)
        )
    ]
    up_index_labels[-1] = "None"
    for up_j, each_row in zip(
        range(len(boolean_network.cycles_unit_perturbations_transition_matrix)),
        boolean_network.cycles_unit_perturbations_transition_matrix,
    ):
        unit_perturbations_strings_list_1[up_j] += str(
            f"{up_index_labels[up_j]:^6s}  |  "
        ) + "".join(
            [str(f"{str(each_row[up_k]):^7s}") for up_k in range(len(each_row))]
        )
    print(
        str(f'{"":^6s}  |  ')
        + "".join(
            [
                str(f"{str(up_index_labels[up_m]):^7s}")
                for up_m in range(len(up_index_labels))
            ]
        )
    )
    print("".join(["-" for up_L in range(len(unit_perturbations_strings_list_1[0]))]))
    for each_string_1 in unit_perturbations_strings_list_1:
        print(each_string_1)
    print("\n")
    print(
        "Figure 2.4: Transitions resulting from unit perturbations to states with a None cycle index "
        + str(boolean_network)
    )
    print(
        "Number of perturbations records: "
        + str(
            len(boolean_network.cycles_unit_perturbations_records)
            + len(boolean_network.t_trajectories_unit_perturbations_records)
            + len(boolean_network.u_trajectories_unit_perturbations_records)
        )
        + " == (number of states in run-ins without a cycle + number of states with an index of None in run-ins with a cycle) * number of nodes\n"
    )
    print("The row represents the starting cycle index\n")
    print("The column represents the ending cycle index\n")
    unit_perturbations_strings_list_2 = [
        ""
        for up_i in range(
            len(boolean_network.total_unit_perturbations_transition_matrix)
        )
    ]
    up_index_labels = [
        str(up_k)
        for up_k in range(
            len(boolean_network.total_unit_perturbations_transition_matrix)
        )
    ]
    up_index_labels[-1] = "None"
    # 08-17-2024 Adding only None state terminated transitions, totals - cycles
    terminated_transitions_matrix = (
        boolean_network.total_unit_perturbations_transition_matrix.copy()
    )
    for f31_i in range(len(terminated_transitions_matrix)):
        for f31_j in range(len(terminated_transitions_matrix[0])):
            terminated_transitions_matrix[f31_i][
                f31_j
            ] -= boolean_network.cycles_unit_perturbations_transition_matrix[f31_i][
                f31_j
            ]
    for up_j, each_row in zip(
        range(len(terminated_transitions_matrix)), terminated_transitions_matrix
    ):
        unit_perturbations_strings_list_2[up_j] += str(
            f"{up_index_labels[up_j]:^6s}  |  "
        ) + "".join(
            [str(f"{str(each_row[up_k]):^7s}") for up_k in range(len(each_row))]
        )
    print(
        str(f'{"":^6s}  |  ')
        + "".join(
            [
                str(f"{str(up_index_labels[up_m]):^7s}")
                for up_m in range(len(up_index_labels))
            ]
        )
    )
    print("".join(["-" for up_L in range(len(unit_perturbations_strings_list_2[0]))]))
    for each_string in unit_perturbations_strings_list_2:
        print(each_string)


def figure_3_2(boolean_network: BooleanNetwork):
    for more_cycles in range(1000):
        boolean_network.add_cycle()
    boolean_network.compute_unit_perturbations_matrix()
    print(
        "Figure 3.2: Transitions resulting from unit perturbations of all observed cycle states of "
        + str(boolean_network)
    )
    print(
        "Number of cycle perturbations records: "
        + str(len(boolean_network.cycles_unit_perturbations_records))
        + " == "
        + str(
            sum(
                [
                    len(boolean_network.bn_collapsed_cycles.cycle_records[up_n])
                    for up_n in range(len(boolean_network.bn_collapsed_cycles))
                ]
            )
        )
        + " * "
        + str(len(boolean_network))
        + "(= sum(length of all cycles) * number of nodes)\n"
    )
    print("The row represents the starting cycle index\n")
    print("The column represents the ending cycle index\n")
    unit_perturbations_strings_list_1 = [
        ""
        for up_i in range(
            len(boolean_network.cycles_unit_perturbations_transition_matrix)
        )
    ]
    up_index_labels = [
        str(up_k)
        for up_k in range(
            len(boolean_network.cycles_unit_perturbations_transition_matrix)
        )
    ]
    up_index_labels[-1] = "None"
    for up_j, each_row in zip(
        range(len(boolean_network.cycles_unit_perturbations_transition_matrix)),
        boolean_network.cycles_unit_perturbations_transition_matrix,
    ):
        unit_perturbations_strings_list_1[up_j] += str(
            f"{up_index_labels[up_j]:^6s}  |  "
        ) + "".join(
            [str(f"{str(each_row[up_k]):^7s}") for up_k in range(len(each_row))]
        )
    print(
        str(f'{"":^6s}  |  ')
        + "".join(
            [
                str(f"{str(up_index_labels[up_m]):^7s}")
                for up_m in range(len(up_index_labels))
            ]
        )
    )
    print("".join(["-" for up_L in range(len(unit_perturbations_strings_list_1[0]))]))
    for each_string_1 in unit_perturbations_strings_list_1:
        print(each_string_1)
    print("\n")
    print(
        "Figure 2.4: Transitions resulting from unit perturbations of all observed states of "
        + str(boolean_network)
    )
    # for cycles
    print(
        "Number of perturbations records: "
        + str(
            len(boolean_network.cycles_unit_perturbations_records)
            + len(boolean_network.t_trajectories_unit_perturbations_records)
            + len(boolean_network.u_trajectories_unit_perturbations_records)
        )
        + " == (number of states in run-ins without a cycle + number of states with an index of None in run-ins with a "
        + "cycle + number of cycle states) * number of nodes\n"
    )
    print("The row represents the starting cycle index\n")
    print("The column represents the ending cycle index\n")
    unit_perturbations_strings_list_2 = [
        ""
        for up_i in range(
            len(boolean_network.total_unit_perturbations_transition_matrix)
        )
    ]
    up_index_labels = [
        str(up_k)
        for up_k in range(
            len(boolean_network.total_unit_perturbations_transition_matrix)
        )
    ]
    up_index_labels[-1] = "None"
    for up_j, each_row in zip(
        range(len(boolean_network.total_unit_perturbations_transition_matrix)),
        boolean_network.total_unit_perturbations_transition_matrix,
    ):
        unit_perturbations_strings_list_2[up_j] += str(
            f"{up_index_labels[up_j]:^6s}  |  "
        ) + "".join(
            [str(f"{str(each_row[up_k]):^7s}") for up_k in range(len(each_row))]
        )
    print(
        str(f'{"":^6s}  |  ')
        + "".join(
            [
                str(f"{str(up_index_labels[up_m]):^7s}")
                for up_m in range(len(up_index_labels))
            ]
        )
    )
    print("".join(["-" for up_L in range(len(unit_perturbations_strings_list_2[0]))]))
    for each_string in unit_perturbations_strings_list_2:
        print(each_string)


def log_histogram_given_axes(hist_x: list[int | float], axs):
    # bins should be something divided by dpi, ... do objects that interact and have plot-chains, or something
    for lhga_i, each_x in zip(range(len(hist_x)), hist_x):
        if (
            each_x != 0
        ):  # Or up to twice the minimum ? which would be 0 for a lot of them ...
            hist_x[lhga_i] = math.log2(each_x)
    axs.hist(hist_x, color="k", bins=200)
    axs.set_yscale("log")
    axs.set_xlabel("log2(magnitudes of un-normalized values)")  # Intended use ...
    axs.set_ylabel("obs.")  # TODO log2 re perceptual brightness


def figure_3_3(boolean_network: BooleanNetwork):
    heat_map_matrix_1 = copy.deepcopy(
        boolean_network.cycles_unit_perturbations_transition_matrix
    )
    row_sums_list = [
        sum(heat_map_matrix_1[rsl_i]) for rsl_i in range(len(heat_map_matrix_1))
    ]
    hist_1_x = []
    for rsl_L in range(len(row_sums_list)):
        if row_sums_list[rsl_L] == 0:
            row_sums_list[rsl_L] = (
                1  # Avoids division by zero error and won't affect output (0/1 = 0)
            )
    for rsl_j in range(len(heat_map_matrix_1)):
        for rsl_k in range(len(heat_map_matrix_1[0])):  # though it is square
            hist_1_x.append(heat_map_matrix_1[rsl_j][rsl_k])
            heat_map_matrix_1[rsl_j][rsl_k] = (
                heat_map_matrix_1[rsl_j][rsl_k] / row_sums_list[rsl_j]
            )
    # fig_c, ax_c = plt.subplots(2, 1, height_ratios=[4, 1])
    fig_c, ax_c = plt.subplots(2, 1, height_ratios=[8, 1])
    ax_c[0].imshow(heat_map_matrix_1)  # Should also be normalized to cycle length ?
    log_histogram_given_axes(hist_1_x, ax_c[1])
    # print("Heatmap 1")
    plt.show()
    #
    # If we compare a number of these quantitatively, we see that networks of a given size with relatively more cycles
    # tend to have ... TODO
    # If they were completely densely packed, then every unit perturbation would be a transition
    # [08-17-2024: example of that densely packed cell state space: all nodes have state t+1 = state at t, and only*
    # themselves as input, * can probably be relaxed, every condition would be an equilibrial state]
    # or normalize to num starting states * num ending staets
    # Could also normalize to inverse of number of 0 length transitions (perturbed state is in end cycle)
    # print("Figure 3.3.2: Heatmap of transition matrix for terminated run-in None-index reference states")
    # heat_map_matrix_2 = copy.deepcopy(boolean_network.cycles_unit_perturbations_transition_matrix)
    # hist_2_x = []
    # for f332_i in range(len(heat_map_matrix_2)):
    #     for f332_j in range(len(heat_map_matrix_2)):
    #         heat_map_matrix_2[f332_i][f332_j] = boolean_network.total_unit_perturbations_transition_matrix[f332_i][f332_j] - boolean_network.cycles_unit_perturbations_transition_matrix[f332_i][f332_j]
    # row_sums_list = [sum(heat_map_matrix_2[rsl_i]) for rsl_i in range(len(heat_map_matrix_2))]
    # for rsl_L in range(len(row_sums_list)):
    #     if row_sums_list[rsl_L] == 0:
    #         row_sums_list[rsl_L] = 1  # Avoids division by zero error and won't affect output (0/1 = 0)
    # for rsl_j in range(len(heat_map_matrix_2)):
    #     for rsl_k in range(len(heat_map_matrix_2[0])):  # though it is square
    #         hist_2_x.append(heat_map_matrix_2[rsl_j][rsl_k])
    #         heat_map_matrix_2[rsl_j][rsl_k] = heat_map_matrix_2[rsl_j][rsl_k] / row_sums_list[rsl_j]
    # fig_t, ax_t = plt.subplots(2, 1, height_ratios=[4, 1])
    # ax_t[0].imshow(heat_map_matrix_2)  # Should also be normalized to cycle length ?
    # log_histogram_given_axes(hist_2_x, ax_t[1])
    # print("Heatmap 2")
    # plt.show()
    # #
    # print("Figure 3.3.3: Heatmap of terminated trajectories transition matrix entries normalized to num starting cycle "
    #       "states * num ending cycle states")  # TODO Rethink normalization, here, could be only ending cycle [starting]
    # heat_map_matrix_3 = copy.deepcopy(boolean_network.cycles_unit_perturbations_transition_matrix)
    # hist_3_x = []
    # for f332_i in range(len(heat_map_matrix_3)):
    #     for f332_j in range(len(heat_map_matrix_3)):
    #         heat_map_matrix_3[f332_i][f332_j] = boolean_network.total_unit_perturbations_transition_matrix[f332_i][f332_j] - boolean_network.cycles_unit_perturbations_transition_matrix[f332_i][f332_j]
    # for rsl_j in range(len(heat_map_matrix_2)):
    #     for rsl_k in range(len(heat_map_matrix_3[0])):  # though it is square
    #         if rsl_k != len(heat_map_matrix_3[0]) - 1 and rsl_j != len(heat_map_matrix_3) - 1:
    #             heat_map_matrix_3[rsl_j][rsl_k] = heat_map_matrix_2[rsl_j][rsl_k] / (
    #                               len(boolean_network.bn_collapsed_cycles.cycle_records[rsl_j]) * len(
    #                               boolean_network.bn_collapsed_cycles.cycle_records[
    #                               rsl_k]))  # len(boolean_network.bn_collapsed_cycles.cycle_records)
    #         else:
    #             heat_map_matrix_2[rsl_j][rsl_k] = 0  # DON'T 08-18 This doesn't make sense, don't leave out row, fix /0
    #             #   Makes sense because of normalization-- no numbers for None index
    #     row_sums_list = [sum(heat_map_matrix_2[rsl_i]) for rsl_i in range(len(heat_map_matrix_2))]
    #     for rsl_L in range(len(row_sums_list)):
    #         if row_sums_list[rsl_L] == 0:
    #             row_sums_list[rsl_L] = 1  # Avoids division by zero error and won't affect output, because of how used
    #     for rsl_m in range(len(heat_map_matrix_2[0])):
    #         hist_3_x.append(heat_map_matrix_3[rsl_j][rsl_m])
    #         heat_map_matrix_3[rsl_j][rsl_m] = heat_map_matrix_3[rsl_j][rsl_m] / row_sums_list[rsl_m]
    # fig = plt.figure()
    # fig.set_size_inches(10, 10)
    # fig, ax = plt.subplots(2, 1, height_ratios=[4, 1])
    # ax[0].imshow(heat_map_matrix_3)  # Should also be normalized to cycle length ?
    # log_histogram_given_axes(hist_3_x, ax[1])
    # print("Heatmap 3")
    # plt.show()


def figure_3_4(
    boolean_network: BooleanNetwork,
):  # TODO Add a network graph representation here, highlight high P nodes
    print("Figure 3.4.1: ")
    # P for each perturbation (show plus and minus, for each node (row), two columns) you could get this measure
    # for genes, and it'd mostly come down to how you define State (define it at the perturbed state, and in run-in
    # therefrom (finite state machine, for effects of perturbation)
    # make columns
    # "Let State := cycle at cycle detection\n" # For cell stuff, this is one measure, but also, R^n of abundances
    # and distance
    # Add columns, divided by number of possible perturbation times
    perturbation_columns = [[0, 0] for f34 in range(len(boolean_network))]
    possible_perturbations_columns = [
        [0, 0] for f34_1 in range(len(boolean_network))
    ]  # can be turned off or on
    normalized_p_columns = [
        [0, 0] for f34_2 in range(len(boolean_network))
    ]  # perturbation_columns.copy()
    for each_perturbation_record in boolean_network.cycles_unit_perturbations_records:
        for each_state in boolean_network.bn_collapsed_cycles.cycle_records[
            each_perturbation_record.start_index
        ]:
            # if its on then it can be turned off, if its off (False) then it can be turned on
            # int(True) -> 1, ppc[node][0] represents state is True, and can be turned off
            possible_perturbations_columns[
                each_perturbation_record.perturbed_node_index
            ][1 - int(each_state[each_perturbation_record.perturbed_node_index])] += 1
            # testing: each row (per node) should sum to the same number, equal to total cycle states
        if each_perturbation_record.start_index != each_perturbation_record.end_index:
            if each_perturbation_record.reference_state[
                each_perturbation_record.perturbed_node_index
            ]:
                perturbation_columns[each_perturbation_record.perturbed_node_index][
                    0
                ] += 1
            else:
                perturbation_columns[each_perturbation_record.perturbed_node_index][
                    1
                ] += 1
    for p_i in range(len(perturbation_columns)):  # addressed rows by columns
        for p_j in range(2):
            # if possible_perturbations_columns[p_i][p_j] == 0:
            #     possible_perturbations_columns[p_i][p_j] = 1
            #     print("should be 0: " + str(perturbation_columns[p_i][p_j]))
            normalized_p_columns[p_i][p_j] = perturbation_columns[p_i][p_j] / (
                1
                if perturbation_columns[p_i][p_j] == 0
                else possible_perturbations_columns[p_i][p_j]
            )
    strings_p_output_1 = [
        "Let State := cycle at cycle detection\n",
        "P := probability that a net changes State\n",
        "Total State changes and P for unit perturbations to",
        str(boolean_network),
        "\n",
        "        " + str(f'{"turns off":^12s}') + "|" + str(f'{"turns on":^12s}'),
    ]
    strings_p_output_2 = []
    strings_p_dash_line = "".join(["-" for f34_k in range(len(strings_p_output_1[-1]))])
    strings_p_output_1[-1] += "  P".rjust(20).ljust(40)
    strings_p_dash_line += "".join([" " for f34_L in range(15)]) + "".join(
        ["-" for f_34_L in range(35)]
    )
    for f34_j in range(len(perturbation_columns)):
        strings_p_output_2.append(
            "node "
            + str(f"{str(f34_j + 1):>3s}")
            + ": "
            + str(f"{perturbation_columns[f34_j][0]:^12d}")
            + "|"
            + str(f"{perturbation_columns[f34_j][1]:^12d}")
            + (
                str(
                    f'{str(perturbation_columns[f34_j][0]) + "/" + 
                                   str(possible_perturbations_columns[f34_j][0]):^20s}'
                )
                + "|"
                + str(
                    f'{str(perturbation_columns[f34_j][1]) + "/" +
                                   str(possible_perturbations_columns[f34_j][1]):^20s}'
                )
            ).rjust(50)
        )
    # from https://stackoverflow.com/a/5676676/25597680
    strings_p_output_1[-1] = strings_p_output_1[-1].rjust(len(strings_p_output_2[-1]))
    for each_line in ["\n".join(strings_p_output_1)] + strings_p_output_2:
        print("\n".join([each_line, strings_p_dash_line]))
    # TODO For perturbation with the most total P (-, +), heatmap over all cycles as in 3_3
    print("Figure 3.4.2: Heatmap of fractions at right in 3.4.1")
    plt.figure()
    plt.imshow(normalized_p_columns)
    plt.show()


# Revised 11/18/2024  # problem was likely related to not calling .notify() in sorting methods of cycle manager
def get_transition_indices(boolean_network: BooleanNetwork, number_of_indices: int):
    transitions_indices = []
    temp_index = 0
    while len(transitions_indices) < number_of_indices:
        transitions_bool = True
        while transitions_bool:
            # temp_index = SystemRandom().randrange(0, len(boolean_network.cycles_unit_perturbations_records))
            temp_index += 1
            temp_record = boolean_network.cycles_unit_perturbations_records[temp_index]
            if (
                temp_record.start_index != temp_record.end_index
                and len(transitions_indices) > 0
            ):
                transitions_bool_2 = True
                for tr_i in range(len(transitions_indices)):
                    if (
                        temp_record.start_index
                        == boolean_network.cycles_unit_perturbations_records[
                            transitions_indices[tr_i]
                        ].start_index
                    ):
                        transitions_bool_2 = False
                if transitions_bool_2:
                    transitions_indices.append(temp_index)
                    transitions_bool = False
            elif temp_record.start_index != temp_record.end_index:
                transitions_indices.append(temp_index)
                transitions_bool = False
    return transitions_indices


def plot_a_transition(
    boolean_network: BooleanNetwork,
    hamming_setup: HammingSetup,
    transition_index: int,
    text_plot_if_true: bool,
    bv_colors: list,
):
    # Originally referred to Figure 1 data, TBD TODO Objects for figures and object for whole thing
    end_of_transition_x = None
    end_of_transition_y = None
    perturbation_record = boolean_network.cycles_unit_perturbations_records[
        transition_index
    ]
    if (
        perturbation_record.end_index is None
    ):  # 08-17-2024 end_index of None (I think) came up, could be start_index
        # if it's start_index there's errors elsewhere, ... , TODO unittest
        text_plot_if_true = False
        print("Perturbation record end_index is None, plotting run_in as ending cycle:")
        list_of_perturbation_states = [
            perturbation_record.run_in,
            boolean_network.bn_collapsed_cycles.cycle_records[
                perturbation_record.start_index
            ].cycle_states_list,
            perturbation_record.run_in,
        ]
    else:
        list_of_perturbation_states = [
            perturbation_record.run_in,
            boolean_network.bn_collapsed_cycles.cycle_records[
                perturbation_record.start_index
            ].cycle_states_list,
            boolean_network.bn_collapsed_cycles.cycle_records[
                perturbation_record.end_index
            ].cycle_states_list,
        ]
    # Text output
    if text_plot_if_true:
        fig_2_states_list = perturbation_record.run_in  # Redundant
        test_bn_string_list = [
            str(f'{"node " + str(tbn_k + 1):>11s}: ')
            + str(f"{str(perturbation_record.reference_state[tbn_k]):^11s}-")
            for tbn_k in range(len(fig_2_states_list[0]))
        ]  # one for each node
        for each_list_1 in fig_2_states_list:
            for tbn_l in range(len(boolean_network)):
                test_bn_string_list[tbn_l] += str(f"{str(each_list_1[tbn_l]):^11s}")
                # if each_list_1 is fig_1_current_states_list[-2]:
                #     test_bn_string_list[tbn_l] += "|"
                # else:
                #     test_bn_string_list[tbn_l] += " "
        print(
            "\ncycle index: "
            + "".join(
                [str(f"{str(perturbation_record.start_index):^11s}")]
                + [
                    str(
                        f"{str(boolean_network.bn_collapsed_cycles.get_index(each_state)):^12s}"
                    )
                    for each_state in perturbation_record.run_in
                ]
            )
            + "\n"
        )
        print("\n".join(test_bn_string_list))
    # print(list_of_perturbation_states)  # was for debugging- now prints bytearrays TODO check on whether this makes string comparisons more whack
    # Hamming space output
    for each_states_list in list_of_perturbation_states:
        if list_of_perturbation_states.index(each_states_list) == 0:
            transitions_color = "k"
            transitions_line_style = "dashed"
            transitions_line_width = 1
            transitions_label = "transition"
            transitions_alpha = 0.75
        elif list_of_perturbation_states.index(each_states_list) == 1:
            transitions_color = bv_colors[
                int(
                    len(bv_colors)
                    * perturbation_record.start_index
                    / len(boolean_network.bn_collapsed_cycles)
                )
            ]
            transitions_line_style = "solid"
            transitions_line_width = 2
            transitions_label = "starting cycle"
            transitions_alpha = 1
        else:
            transitions_color = bv_colors[
                int(
                    len(bv_colors)
                    * perturbation_record.end_index
                    / len(boolean_network.bn_collapsed_cycles)
                )
            ]
            transitions_line_style = "solid"
            transitions_line_width = 2
            transitions_label = "ending cycle"
            transitions_alpha = 1
        each_x = []
        each_y = []
        for each_cycle_state in each_states_list:
            each_x.append(
                sum(
                    [
                        hamming_setup.linear_function[fig_143_n]
                        * get_hamming_distance(
                            hamming_setup.first_sequences[fig_143_n], each_cycle_state
                        )
                        for fig_143_n in range(len(hamming_setup.linear_function))
                    ]
                )
            )
            each_y.append(
                sum(
                    [
                        hamming_setup.linear_function[fig_143_o]
                        * get_hamming_distance(
                            hamming_setup.second_sequences[fig_143_o], each_cycle_state
                        )
                        for fig_143_o in range(len(hamming_setup.linear_function))
                    ]
                )
            )
            if list_of_perturbation_states.index(each_states_list) == 0:
                if each_cycle_state is each_states_list[-1]:
                    end_of_transition_x = each_x[-1]
                    end_of_transition_y = each_y[-1]
        if (
            len(each_x) == 2
            and list_of_perturbation_states.index(each_states_list) != 0
        ):
            plt.scatter(
                each_x,
                each_y,
                color=transitions_color,
                linestyle=transitions_line_style,
                marker="P",
                linewidth=transitions_line_width,
                alpha=transitions_alpha,
                label=transitions_label,
                s=100,
            )
        else:
            plt.plot(
                each_x,
                each_y,
                color=transitions_color,
                linestyle=transitions_line_style,
                linewidth=transitions_line_width,
                alpha=transitions_alpha,
                label=transitions_label,
            )
    transition_ref_x = sum(
        [
            hamming_setup.linear_function[fig_143_n]
            * get_hamming_distance(
                hamming_setup.first_sequences[fig_143_n],
                perturbation_record.reference_state,
            )
            for fig_143_n in range(len(hamming_setup.linear_function))
        ]
    )
    transition_ref_y = sum(
        [
            hamming_setup.linear_function[fig_143_o]
            * get_hamming_distance(
                hamming_setup.second_sequences[fig_143_o],
                perturbation_record.reference_state,
            )
            for fig_143_o in range(len(hamming_setup.linear_function))
        ]
    )
    transition_perturbed_x = sum(
        [
            hamming_setup.linear_function[fig_143_n]
            * get_hamming_distance(
                hamming_setup.first_sequences[fig_143_n],
                list_of_perturbation_states[0][0],
            )
            for fig_143_n in range(len(hamming_setup.linear_function))
        ]
    )
    transition_perturbed_y = sum(
        [
            hamming_setup.linear_function[fig_143_o]
            * get_hamming_distance(
                hamming_setup.second_sequences[fig_143_o],
                list_of_perturbation_states[0][0],
            )
            for fig_143_o in range(len(hamming_setup.linear_function))
        ]
    )
    plt.scatter(
        transition_ref_x,
        transition_ref_y,
        marker=".",
        color=bv_colors[
            int(
                len(bv_colors)
                * perturbation_record.start_index
                / len(boolean_network.bn_collapsed_cycles)
            )
        ],
        label="reference state",
        s=100,
        alpha=0.6,
    )
    plt.scatter(
        transition_perturbed_x,
        transition_perturbed_y,
        marker=".",
        color="k",
        alpha=0.75,
        label="perturbed state",
        s=80,
    )
    plt.plot(
        [transition_ref_x, transition_perturbed_x],
        [transition_ref_y, transition_perturbed_y],
        linestyle="dotted",
        linewidth=1,
        color="k",
    )
    plt.scatter(
        end_of_transition_x,
        end_of_transition_y,
        marker="x",
        s=40,
        alpha=0.75,
        color="k",
        label="end of transition",
    )
    plt.figlegend()
    plt.show()


# DONE Figure out error relating to reference state not in starting cycle states


def plot_transition_s(
    boolean_network: BooleanNetwork,
    hamming_setup: HammingSetup,
    transition_index: int,
    text_plot_if_true: bool,
    bv_colors: list,
):
    transitions_bool = True
    transitions_int = 0
    this_transition = boolean_network.cycles_unit_perturbations_records[
        transition_index
    ]
    transitions_indices = [transition_index]
    for tr_L, each_perturbation_record in zip(
        range(len(boolean_network.cycles_unit_perturbations_records)),
        boolean_network.cycles_unit_perturbations_records,
    ):
        if each_perturbation_record.start_index == this_transition.start_index:
            if each_perturbation_record.end_index == this_transition.end_index:
                transitions_indices.append(tr_L)
    # print("Number of transitions: " + str(len(transitions_indices)))
    for each_transition_index in transitions_indices:
        transition_index = each_transition_index
        end_of_transition_x = None
        end_of_transition_y = None
        perturbation_record = boolean_network.cycles_unit_perturbations_records[
            transition_index
        ]
        list_of_perturbation_states = [
            perturbation_record.run_in,
            boolean_network.bn_collapsed_cycles.cycle_records[
                perturbation_record.start_index
            ].cycle_states_list,
            boolean_network.bn_collapsed_cycles.cycle_records[
                perturbation_record.end_index
            ].cycle_states_list,
        ]
        # Text output
        if text_plot_if_true:
            fig_2_states_list = boolean_network.cycles_unit_perturbations_records[
                transition_index
            ].run_in  # Redundant
            test_bn_string_list = [
                str(f'{"node " + str(tbn_k + 1):>11s}: ')
                + str(
                    f"{str(boolean_network.cycles_unit_perturbations_records[transitions_indices[0]].reference_state[tbn_k]):^11s}-"
                )
                for tbn_k in range(len(fig_2_states_list[0]))
            ]  # one for each node
            for tbn_h in range(len(fig_2_states_list[0])):
                if (
                    boolean_network.cycles_unit_perturbations_records[
                        transition_index
                    ].perturbed_node_index
                    == tbn_h
                ):
                    # test_bn_string_list[tbn_h] += str(f'{str(fig_2_states_list[0][tbn_h]) + "*":^11s}')
                    test_bn_string_list[tbn_h] += str(
                        f'{"(" + str(fig_2_states_list[0][tbn_h]) + ")":^11s}'
                    )
                else:
                    test_bn_string_list[tbn_h] += str(
                        f"{str(fig_2_states_list[0][tbn_h]):^11s}"
                    )
            for each_list_1 in fig_2_states_list[1 : len(fig_2_states_list)]:
                for tbn_l in range(len(fig_2_states_list[0])):
                    test_bn_string_list[tbn_l] += str(f"{str(each_list_1[tbn_l]):^11s}")
            print(
                "\ncycle index: "
                + "".join(
                    [
                        str(
                            f"{str(boolean_network.cycles_unit_perturbations_records[transition_index].start_index):^12s}"
                        )
                    ]
                    + [
                        str(
                            f"{str(boolean_network.bn_collapsed_cycles.get_index(each_state)):^12s}"
                        )
                        for each_state in boolean_network.cycles_unit_perturbations_records[
                            transition_index
                        ].run_in
                    ]
                )
                + "\n"
            )
            print("\n".join(test_bn_string_list))
        # Hamming space output
        for each_states_list in list_of_perturbation_states:
            if list_of_perturbation_states.index(each_states_list) == 0:
                transitions_color = "k"
                transitions_line_style = "dashed"
                transitions_line_width = 1
                transitions_label = "transition"
                transitions_alpha = 0.75
            elif list_of_perturbation_states.index(each_states_list) == 1:
                transitions_color = bv_colors[
                    int(
                        len(bv_colors)
                        * perturbation_record.start_index
                        / len(boolean_network.bn_collapsed_cycles)
                    )
                ]
                transitions_line_style = "solid"
                transitions_line_width = 2
                transitions_label = "starting cycle"
                transitions_alpha = 1
            else:
                transitions_color = bv_colors[
                    int(
                        len(bv_colors)
                        * perturbation_record.end_index
                        / len(boolean_network.bn_collapsed_cycles)
                    )
                ]
                transitions_line_style = "solid"
                transitions_line_width = 2
                transitions_label = "ending cycle"
                transitions_alpha = 1
            each_x = []
            each_y = []
            for each_cycle_state in each_states_list:
                each_x.append(
                    sum(
                        [
                            hamming_setup.linear_function[fig_143_n]
                            * get_hamming_distance(
                                hamming_setup.first_sequences[fig_143_n],
                                each_cycle_state,
                            )
                            for fig_143_n in range(len(hamming_setup.linear_function))
                        ]
                    )
                )
                each_y.append(
                    sum(
                        [
                            hamming_setup.linear_function[fig_143_o]
                            * get_hamming_distance(
                                hamming_setup.second_sequences[fig_143_o],
                                each_cycle_state,
                            )
                            for fig_143_o in range(len(hamming_setup.linear_function))
                        ]
                    )
                )
                if list_of_perturbation_states.index(each_states_list) == 0:
                    if each_cycle_state is each_states_list[-1]:
                        end_of_transition_x = each_x[-1]
                        end_of_transition_y = each_y[-1]
            if (
                transitions_bool and transitions_int <= 2
            ) or list_of_perturbation_states.index(each_states_list) == 0:
                transitions_int += 1
                if (
                    len(each_x) == 2
                    and list_of_perturbation_states.index(each_states_list) != 0
                ):
                    plt.scatter(
                        each_x,
                        each_y,
                        color=transitions_color,
                        linestyle=transitions_line_style,
                        linewidth=transitions_line_width,
                        alpha=transitions_alpha,
                        label=transitions_label,
                        s=100,
                        marker="P",
                    )
                else:
                    plt.plot(
                        each_x,
                        each_y,
                        color=transitions_color,
                        linestyle=transitions_line_style,
                        linewidth=transitions_line_width,
                        alpha=transitions_alpha,
                        label=transitions_label,
                    )
        transition_ref_x = sum(
            [
                hamming_setup.linear_function[fig_143_n]
                * get_hamming_distance(
                    hamming_setup.first_sequences[fig_143_n],
                    perturbation_record.reference_state,
                )
                for fig_143_n in range(len(hamming_setup.linear_function))
            ]
        )
        transition_ref_y = sum(
            [
                hamming_setup.linear_function[fig_143_o]
                * get_hamming_distance(
                    hamming_setup.second_sequences[fig_143_o],
                    perturbation_record.reference_state,
                )
                for fig_143_o in range(len(hamming_setup.linear_function))
            ]
        )
        transition_perturbed_x = sum(
            [
                hamming_setup.linear_function[fig_143_n]
                * get_hamming_distance(
                    hamming_setup.first_sequences[fig_143_n],
                    list_of_perturbation_states[0][0],
                )
                for fig_143_n in range(len(hamming_setup.linear_function))
            ]
        )
        transition_perturbed_y = sum(
            [
                hamming_setup.linear_function[fig_143_o]
                * get_hamming_distance(
                    hamming_setup.second_sequences[fig_143_o],
                    list_of_perturbation_states[0][0],
                )
                for fig_143_o in range(len(hamming_setup.linear_function))
            ]
        )
        plt.scatter(
            transition_ref_x,
            transition_ref_y,
            marker=".",
            color=bv_colors[
                int(
                    len(bv_colors)
                    * perturbation_record.start_index
                    / len(boolean_network.bn_collapsed_cycles)
                )
            ],
            label="reference state",
            s=100,
            alpha=0.6,
        )
        plt.scatter(
            transition_perturbed_x,
            transition_perturbed_y,
            marker=".",
            color="k",
            alpha=0.75,
            label="perturbed state",
            s=80,
        )
        plt.plot(
            [transition_ref_x, transition_perturbed_x],
            [transition_ref_y, transition_perturbed_y],
            linestyle="dotted",
            linewidth=1,
            color="k",
        )
        plt.scatter(
            end_of_transition_x,
            end_of_transition_y,
            marker="x",
            s=40,
            alpha=0.75,
            color="k",
            label="end of transition",
        )
    # from https://stackoverflow.com/a/13589144/25597680
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    # plt.figlegend()
    plt.show()


def figure_3_5(
    boolean_network: BooleanNetwork, hamming_setup: HammingSetup, bv_colors: list
):
    """requires that parameter BooleanNetwork has computed unit perturbations matrix, or has a list of cycle
    UnitPerturbationRecords"""
    # TODO add total for which start index != end index [ extent to which matrix is not diagonal ? ]
    some_indices = get_transition_indices(boolean_network, 10)
    print(
        "Figure 2.1: A transition from calculation of unit perturbations, as text output, and in the same Hamming space as above:"
    )
    plot_a_transition(boolean_network, hamming_setup, some_indices[0], True, bv_colors)
    print(
        "Figure 2.2: 9 transitions from calculation of unit perturbations in the same Hamming space as above:"
    )
    for tr_j in range(1, len(some_indices)):
        print(
            "1 of "
            + str(
                boolean_network.cycles_unit_perturbations_transition_matrix[
                    boolean_network.cycles_unit_perturbations_records[
                        some_indices[tr_j]
                    ].start_index
                ][
                    boolean_network.cycles_unit_perturbations_records[
                        some_indices[tr_j]
                    ].end_index
                ]
            )
            + " transitions from cycle "
            + str(
                boolean_network.cycles_unit_perturbations_records[
                    some_indices[tr_j]
                ].start_index
            )
            + " to cycle "
            + str(
                # boolean_network.cycles_unit_perturbations_records[some_indices[tr_j]].end_index))
                boolean_network.bn_collapsed_cycles.get_index(
                    boolean_network.cycles_unit_perturbations_records[
                        some_indices[tr_j]
                    ].run_in[-1]
                )
            )
        )
        plot_a_transition(
            boolean_network, hamming_setup, some_indices[tr_j], False, bv_colors
        )
    # For all transitions between two cycles
    print(
        "Figure 2.3: All transitions between two cycles, in the same Hamming space as set in Hamming setup"
    )
    transitions_index_1 = get_transition_indices(boolean_network, 1)[0]
    transitions_index_2 = 0
    transitions_bool_3 = True
    while transitions_bool_3:
        for tr_m in range(
            len(boolean_network.cycles_unit_perturbations_records)
        ):  # Should really go through the matrix, check for both i,j and j,i entries non-zero but, this works, pretty slowly...
            if (
                boolean_network.cycles_unit_perturbations_records[tr_m].start_index
                == boolean_network.cycles_unit_perturbations_records[
                    transitions_index_1
                ].end_index
            ):
                if (
                    boolean_network.cycles_unit_perturbations_records[tr_m].end_index
                    == boolean_network.cycles_unit_perturbations_records[
                        transitions_index_1
                    ].start_index
                ):
                    transitions_index_2 = tr_m
        if transitions_index_2 == 0:
            transitions_index_1 = get_transition_indices(boolean_network, 1)[0]
        else:
            transitions_bool_3 = False
    # 11/18/2024: bug fix: reference states not on cycle states  # 12/1/2024 wondering whether this is due to reindexing cycles?
    # yep, that was it- didn't have .notify() in some of sort functions
    print("from cycle A to cycle B")
    plot_transition_s(
        boolean_network, hamming_setup, transitions_index_1, False, bv_colors
    )
    print("from cycle B to cycle A")
    plot_transition_s(
        boolean_network, hamming_setup, transitions_index_2, False, bv_colors
    )


# separately, a text plot method TODO simple text plot method (decomposition of earlier / above code)
def plot_current_states_list(
    boolean_network: BooleanNetwork,
    hamming_setup: HammingSetup,
    bv_colors: list,
    each_x_alpha: list[float],
):
    """This is specifically for Figure: Noise"""
    # TODO Take average of entries of colors for each state_list, use as line color
    #   or plot a black line with colored points  [animation might be cool]
    # Hamming space output
    pcsl_color = "k"  # [0.5, 0.5, 0.5]
    pcsl_line_style = "solid"
    pcsl_line_width = 0.5
    pcsl_label = "IC: " + "".join([str(int(boolean_network.current_states_list[0][0]))])
    pcsl_alpha = 1
    each_x = []
    each_y = []
    for pcsl_state in boolean_network.current_states_list:
        each_x.append(
            sum(
                [
                    hamming_setup.linear_function[fig_143_n]
                    * get_hamming_distance(
                        hamming_setup.first_sequences[fig_143_n], pcsl_state
                    )
                    for fig_143_n in range(len(hamming_setup.linear_function))
                ]
            )
        )
        each_y.append(
            sum(
                [
                    hamming_setup.linear_function[fig_143_o]
                    * get_hamming_distance(
                        hamming_setup.second_sequences[fig_143_o], pcsl_state
                    )
                    for fig_143_o in range(len(hamming_setup.linear_function))
                ]
            )
        )
    while len(each_x_alpha) < len(each_x):
        each_x_alpha = [each_x_alpha[0]] + each_x_alpha
    while len(each_x_alpha) > len(each_x):
        each_x_alpha.remove(each_x_alpha[0])
    # Maybe e^2, e^1 + e^2, e^1
    pcsl_a = (
        len(boolean_network.current_states_list)
        * each_x_alpha[0]
        / (each_x_alpha[0] + each_x_alpha[int(len(each_x_alpha) / 2)] + each_x_alpha[0])
    )
    pcsl_b = (
        len(boolean_network.current_states_list)
        * each_x_alpha[int(len(each_x_alpha) / 2)]
        / (each_x_alpha[0] + each_x_alpha[int(len(each_x_alpha) / 2)] + each_x_alpha[0])
    )
    # pcsl_c = len(boolean_network.current_states_list) * each_x_alpha[0] / (each_x_alpha[0] + each_x_alpha[int(len(each_x_alpha)/2)] + each_x_alpha[0])
    plt.scatter(each_x[0], each_y[0], marker="+", s=10, color="k")
    for pcsl_k in range(2, 27, 2):
        plt.plot(
            each_x[0:pcsl_k],
            each_y[0:pcsl_k],
            color="k",
            linestyle=(1, (2, 2)),
            alpha=0.02,
        )
    plt.plot(
        each_x[0 : int(pcsl_a)],
        each_y[0 : int(pcsl_a)],
        color=pcsl_color,
        linestyle=pcsl_line_style,
        linewidth=pcsl_line_width + 0.02,
        alpha=0.001,
        label=pcsl_label,
    )
    plt.plot(
        each_x[int(pcsl_a) : int(pcsl_b)],
        each_y[int(pcsl_a) : int(pcsl_b)],
        color=pcsl_color,
        linestyle=pcsl_line_style,
        linewidth=pcsl_line_width + 0.03,
        alpha=0.01,
        label=pcsl_label,
    )
    plt.plot(
        each_x[int(pcsl_b) : len(each_x)],
        each_y[int(pcsl_b) : len(each_x)],
        color=pcsl_color,
        linestyle=pcsl_line_style,
        linewidth=pcsl_line_width + 0.05,
        alpha=0.1,
        label=pcsl_label,
    )
    plt.plot(
        each_x[
            int(len(each_x) - boolean_network.get_avg_cycle_length() * 3) : len(each_x)
        ],
        each_y[
            int(len(each_x) - boolean_network.get_avg_cycle_length() * 3) : len(each_x)
        ],
        alpha=0.5,
        color="k",
    )
    plt.plot(
        each_x[
            int(len(each_x) - boolean_network.get_avg_cycle_length() * 2) : len(each_x)
        ],
        each_y[
            int(len(each_x) - boolean_network.get_avg_cycle_length() * 2) : len(each_x)
        ],
        alpha=0.3,
        color="k",
    )
    plt.plot(
        each_x[int(len(each_x) - boolean_network.get_avg_cycle_length()) : len(each_x)],
        each_y[int(len(each_x) - boolean_network.get_avg_cycle_length()) : len(each_x)],
        alpha=0.1,
        color="k",
    )
    for f35_i in range(len(each_x)):
        pcsl_index = boolean_network.bn_collapsed_cycles.get_index(
            boolean_network.current_states_list[f35_i]
        )
        if pcsl_index is not None:
            plt.scatter(
                each_x[f35_i],
                each_y[f35_i],
                alpha=each_x_alpha[f35_i],
                color=bv_colors[
                    int(
                        len(bv_colors)
                        * pcsl_index
                        / len(boolean_network.bn_collapsed_cycles)
                    )
                ],
                s=1,
            )
        else:
            plt.scatter(each_x[f35_i], each_y[f35_i], alpha=0.2, color="k", s=1)
            # Strange bug: (each_x_alpha[f35_i])/2 computing as list / int ... 08-18-2024, replaced with 0.2
    # plt.figlegend(set(plt.get_figlabels()))
    pcsl_temp_most_recent_state_ref = boolean_network.bn_collapsed_cycles.get_reference(
        boolean_network.current_states_list[-1]
    )
    if pcsl_temp_most_recent_state_ref is not None:
        plt.scatter(
            each_x[-1],
            each_y[-1],
            color=bv_colors[
                int(
                    len(bv_colors)
                    * boolean_network.bn_collapsed_cycles.get_index(
                        boolean_network.current_states_list[-1]
                    )
                    / len(boolean_network.bn_collapsed_cycles)
                )
            ],
            marker=".",
            alpha=0.7,
            s=30,
        )
    else:
        plt.scatter(each_x[-1], each_y[-1], marker="s", color="k", alpha=0.7, s=30)
    plt.show()


def figure_noise(
    perturb_prob: float,
    steps: int,
    boolean_network: BooleanNetwork,
    initial_conditions: list[bool],
    hamming_setup: HammingSetup,
    colors: list,
):
    boolean_network.unrecorded_states_with_noise(
        steps, initial_conditions, perturb_prob
    )
    #
    # a + b + c = len(current states list)
    pcsl_a = math.exp(1)
    pcsl_b = math.exp(2)
    pcsl_c = math.exp(1)
    each_x_alpha = (
        [
            pcsl_a / (pcsl_a + pcsl_b + pcsl_c)
            for xa_i in range(
                int(
                    len(boolean_network.current_states_list)
                    * pcsl_a
                    / (pcsl_a + pcsl_b + pcsl_c)
                )
            )
        ]
        + [
            pcsl_b / (pcsl_a + pcsl_b + pcsl_c)
            for xa_j in range(
                int(
                    len(boolean_network.current_states_list)
                    * pcsl_b
                    / (pcsl_a + pcsl_b + pcsl_c)
                )
            )
        ]
        + [
            pcsl_c / (pcsl_a + pcsl_b + pcsl_c)
            for xa_k in range(
                int(
                    len(boolean_network.current_states_list)
                    * pcsl_b
                    / (pcsl_a + pcsl_b + pcsl_c)
                )
            )
        ]
        + [
            [
                (pcsl_a + pcsl_b) / (pcsl_a + pcsl_b + pcsl_c)
                for xa_L in range(
                    int(
                        len(boolean_network.current_states_list)
                        * pcsl_b
                        / (pcsl_a + pcsl_b + pcsl_c)
                    )
                )
            ]
        ]
    )
    print(
        str(steps)
        + " steps with "
        + str(perturb_prob)
        + " probability of unit perturbation"
    )
    print(
        "Initial condtiion: "
        + "".join(
            str(int(initial_conditions[fn_i]))
            for fn_i in range(len(initial_conditions))
        )
    )
    plot_current_states_list(boolean_network, hamming_setup, colors, each_x_alpha)
    #
    boolean_network.unrecorded_states_with_bv_noise(
        steps, initial_conditions, perturb_prob
    )
    print(
        str(steps)
        + " steps with "
        + str(perturb_prob)
        + " * per node Boolean velocity probability of unit perturbation"
    )
    print(
        "Initial condtiion: "
        + "".join(
            str(int(initial_conditions[fn_i]))
            for fn_i in range(len(initial_conditions))
        )
    )
    plot_current_states_list(boolean_network, hamming_setup, colors, each_x_alpha)


def figure_abundances(boolean_network: BooleanNetwork, colors: list):
    """This figure assumes all cycles are 24 hours... and determines a common multiple of steps such that all cycles fit,
    TODO implement that: make random_setup_2 that takes B.N. as param and calculates as below
    """
    # TODO MAKE FADING PLOTTING FROM INITIAL CONDITION, DASHED
    cycle_lengths_set = set()
    common_multiple = 1
    for each_cycle in boolean_network.bn_collapsed_cycles.cycle_records:
        cycle_lengths_set.add(len(each_cycle))
    for each_length in cycle_lengths_set:
        if common_multiple % each_length != 0:
            common_multiple *= each_length
    print("common_multiple: " + str(common_multiple))
    print("common_multiple / each cycle length:")
    for each_cycle_2 in boolean_network.bn_collapsed_cycles.cycle_records:
        print(
            str(common_multiple)
            + "/"
            + str(len(each_cycle_2))
            + " == "
            + str(common_multiple / len(each_cycle_2))
        )
    num_col = 3
    num_row = int(len(boolean_network.bn_collapsed_cycles) / num_col)
    if len(boolean_network.bn_collapsed_cycles) / num_col - num_row > 0.0001:
        num_row += 1
    fig_a, axs_a = plt.subplots(num_row, num_col)
    fig_a.set_size_inches(12, 3 * num_row)
    fig_a.suptitle("Figure X Abundances: Each cycle of preceding Figure")
    fig_a.supxlabel("steps")
    fig_a.supylabel("abundance")
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs_a.flat:
        ax.label_outer()
    each_cycle_index = 0
    all_panels_x = []
    all_panels_y = []
    for fig_panels_i in range(num_row):
        all_panels_x.append([])
        all_panels_y.append([])
        for fig_panels_j in range(num_col):
            if num_col * fig_panels_i + fig_panels_j < len(
                boolean_network.bn_collapsed_cycles
            ):
                axs_a[fig_panels_i, fig_panels_j].set_title(
                    "Cycle length: "
                    + str(
                        len(
                            boolean_network.bn_collapsed_cycles.cycle_records[
                                num_col * fig_panels_i + fig_panels_j
                            ]
                        )
                    )
                    + " steps"
                )
                axs_a[fig_panels_i, fig_panels_j].plot(
                    all_panels_x[num_col * fig_panels_i + fig_panels_j],
                    all_panels_y[num_col * fig_panels_i + fig_panels_j],
                    color=colors[
                        int(
                            len(colors)
                            * each_cycle_index
                            / len(boolean_network.bn_collapsed_cycles)
                        )
                    ],
                    linestyle="--",
                    marker=".",
                    linewidth=0.75,
                )
                each_cycle_index += 1
    plt.show()


def appendix_figs(boolean_network: BooleanNetwork):
    # Make this one loaded data, and increase replicates number to a large number and run overnight
    # add file write and use .txt file or .csv -- Can figure that out tomorrow ### And to add to that, implement something to plot only non-redundant points, and, ... really should just quantify this numerically
    # Expected runtime: 5.2223 hours, actual runtime:
    bn_list_1 = [[], []]
    bn_list_n = [10, 20, 40, 80, 120, 200]  # TODO Every 10, or even 5?
    # base_replicates = 800  # replicates = base_replicates / N
    replicates = 10
    # Figure 2 ### Runtime ~2 minutes ### Runtime ~3m8s
    all_functions_hist_x = []
    all_functions_xy = [[] for abn_w in range(len(bn_list_n))]
    all_functions_avg_k = [[] for abn_u in range(len(bn_list_n))]
    excl_taut_contr_hist_x = []
    excl_taut_contr_xy = [[] for abn_v in range(len(bn_list_n))]
    excl_taut_contr_avg_k = [[] for abn_t in range(len(bn_list_n))]
    for each_n in bn_list_n:
        for abn_o in range(replicates):  # int(base_replicates/each_n):
            a_bns_all_functions_excl_taut_contr = [
                BooleanNetwork(each_n, [abn_k for abn_k in range(16)], 2, 200),
                BooleanNetwork(each_n, [abn_L for abn_L in range(1, 15)], 2, 200),
            ]
            bn_list_1[0].append(a_bns_all_functions_excl_taut_contr[0])
            bn_list_1[1].append(a_bns_all_functions_excl_taut_contr[1])
            for each_bn in a_bns_all_functions_excl_taut_contr:
                for abn_m in range(200):
                    each_bn.add_cycle()
                if len(each_bn.bn_collapsed_cycles) == 0:
                    for abn_p in range(200):
                        each_bn.add_cycle()
            if len(a_bns_all_functions_excl_taut_contr[0].bn_collapsed_cycles) > 0:
                all_functions_xy[bn_list_n.index(each_n)].append(
                    len(a_bns_all_functions_excl_taut_contr[0].bn_collapsed_cycles)
                )
                all_functions_avg_k[bn_list_n.index(each_n)].append(
                    a_bns_all_functions_excl_taut_contr[0].get_avg_k()
                )
                for each_all_cycle in a_bns_all_functions_excl_taut_contr[
                    0
                ].bn_collapsed_cycles.cycle_records:
                    for abn_x_all in range(each_all_cycle.num_observations):
                        all_functions_hist_x.append(len(each_all_cycle))
            if len(a_bns_all_functions_excl_taut_contr[1].bn_collapsed_cycles) > 0:
                excl_taut_contr_xy[bn_list_n.index(each_n)].append(
                    len(a_bns_all_functions_excl_taut_contr[1].bn_collapsed_cycles)
                )
                excl_taut_contr_avg_k[bn_list_n.index(each_n)].append(
                    a_bns_all_functions_excl_taut_contr[1].get_avg_k
                )
                for each_exc_cycle in a_bns_all_functions_excl_taut_contr[
                    1
                ].bn_collapsed_cycles.cycle_records:
                    for abn_x_all in range(each_exc_cycle.num_observations):
                        excl_taut_contr_hist_x.append(len(each_exc_cycle))
    #
    plt.figure()
    print(
        "Cycle length vs. Number of observations, for "
        + str(replicates)
        + " replicates of networks with N="
        + str(bn_list_n)
    )
    print("Skew:")
    print(
        "All functions: "
        + str(skew_sample(all_functions_hist_x))
        + ",  Excluding tautology and contradiction: "
        + str(skew_sample(excl_taut_contr_hist_x))
    )
    print(
        "Usually, the skew for All functions is more positive than the skew for functions excluding tautology and contradiction"
    )
    plt.xlabel("Cycle length")
    plt.ylabel("Number of observations")
    plt.hist(
        all_functions_hist_x,
        bins=max(all_functions_hist_x),
        alpha=0.5,
        color="b",
        label="All Boolean functions of up to 2 inputs",
    )
    plt.hist(
        excl_taut_contr_hist_x,
        bins=max(excl_taut_contr_hist_x),
        alpha=0.5,
        color="r",
        label="Boolean functions of up to 2 inputs excluding tautology and contradiction",
    )
    plt.figlegend()
    plt.show()
    print(
        "N vs. number of observed cycles per network, for "
        + str(replicates)
        + " replicates of networks with N="
        + str(bn_list_n)
        + ", and selected network (N="
        + str(len(boolean_network))
        + ")"
    )
    print(
        "Each observation is plotted as a circle, and the selected network is marked with a black +. The average for each N is plotted as a horizontal tick mark."
    )
    plt.figure()
    plt.xlabel("N")
    plt.xticks([bn_list_n[abn_v] for abn_v in range(len(bn_list_n))])
    plt.ylabel("Number of observed cycles")
    for abn1_j in range(len(bn_list_n)):
        plt.scatter(
            [bn_list_n[abn1_j] for abn1_k in range(len(all_functions_xy[abn1_j]))],
            all_functions_xy[abn1_j],
            color="b",
            alpha=0.5,
        )
        plt.scatter(
            [bn_list_n[abn1_j] for abn1_L in range(len(excl_taut_contr_xy[abn1_j]))],
            excl_taut_contr_xy[abn1_j],
            color="r",
            alpha=0.5,
        )
        add_to_sum = 0
        add_to_divisor = 0
        if bn_list_n[abn1_j] == len(boolean_network):
            add_to_sum = len(boolean_network.bn_collapsed_cycles)
            add_to_divisor = 1
        plt.scatter(
            bn_list_n[abn1_j],
            sum(all_functions_xy[abn1_j]) / len(all_functions_xy[abn1_j]),
            marker="_",
            s=150,
            color="b",
        )
        plt.scatter(
            bn_list_n[abn1_j],
            (sum(excl_taut_contr_xy[abn1_j]) + add_to_sum)
            / (len(excl_taut_contr_xy[abn1_j]) + add_to_divisor),
            marker="_",
            s=150,
            color="r",
        )
    plt.scatter(
        len(boolean_network),
        len(boolean_network.bn_collapsed_cycles),
        color="r",
        alpha=0.5,
    )
    plt.scatter(
        len(boolean_network),
        len(boolean_network.bn_collapsed_cycles),
        marker="+",
        color="k",
    )
    plt.figlegend(
        labels=[
            "All Boolean functions of up to 2 inputs",
            "Boolean functions of up to 2 inputs excluding tautology and contradiction",
            "Average",
            "Average",
            "Selected network",
        ]
    )
    # for abn1_j in range(len(bn_list_n)):
    #     plt.violinplot(all_functions_xy[abn1_j], [abn1_j])
    #     plt.violinplot(excl_taut_contr_xy[abn1_j], [abn1_j])  # TODO Figure these out, these would be a good way to display this data  ### Yea, do figure this out, it's a ton of points to plot as it is
    plt.show()
    # Scattergram of log(run-in duration (w/o cycle states) + 1) vs. log(terminal cycle length + 1)  ### Query t_records[].run_in of TrajectoriesManager, randomly sampling terminated run-ins ~ 5%, to limit computation # TODO Move to appendix section, use loaded data
    s1_x = []
    s1_y = []
    for each_network_s_1 in bn_list_1[0]:
        for each_t_record_1 in each_network_s_1.bn_trajectories.t_records:
            if SystemRandom().random() > 0.95:
                s1_temp_terminal_cycle_length = len(
                    each_network_s_1.bn_collapsed_cycles.get_reference(
                        each_t_record_1.run_in[-1]
                    )
                )
                s1_temp_cycle_indices = [
                    each_network_s_1.bn_collapsed_cycles.get_index(
                        each_t_record_1.run_in[s1_i]
                    )
                    for s1_i in range(len(each_t_record_1.run_in))
                ]
                s1_temp_run_in_duration = s1_temp_cycle_indices.count(None)
                s1_x.append(math.log(s1_temp_run_in_duration + 1))
                s1_y.append(math.log(s1_temp_terminal_cycle_length + 1))
    s2_x = []
    s2_y = []
    for each_network_s_2 in bn_list_1[1]:
        for each_t_record_2 in each_network_s_2.bn_trajectories.t_records:
            if SystemRandom().random() > 0.95:
                s2_temp_terminal_cycle_length = len(
                    each_network_s_2.bn_collapsed_cycles.get_reference(
                        each_t_record_2.run_in[-1]
                    )
                )
                s2_temp_cycle_indices = [
                    each_network_s_2.bn_collapsed_cycles.get_index(
                        each_t_record_2.run_in[s2_i]
                    )
                    for s2_i in range(len(each_t_record_2.run_in))
                ]
                s2_temp_run_in_duration = s2_temp_cycle_indices.count(
                    None
                )  # TODO CAN THESE ALL BE EACH_NETWORK AND IT STILL WORKS?
                s2_x.append(math.log(s2_temp_run_in_duration + 1))
                s2_y.append(math.log(s2_temp_terminal_cycle_length + 1))
    s3_x = []
    s3_y = []
    for each_t_record_3 in boolean_network.bn_trajectories.t_records:
        s3_temp_terminal_cycle_length = len(
            boolean_network.bn_collapsed_cycles.get_reference(
                each_t_record_3.run_in[-1]
            )
        )  # len(CycleRecord)
        s3_temp_cycle_indices = [
            boolean_network.bn_collapsed_cycles.get_index(
                (each_t_record_3.run_in[s3_i])
                for s3_i in range(len(each_t_record_3.run_in))
            )
        ]
        s3_temp_run_in_duration = s3_temp_cycle_indices.count(None)
        s3_x.append(math.log(s3_temp_run_in_duration + 1))
        s3_y.append(math.log(s3_temp_terminal_cycle_length + 1))
    plt.scatter(
        s3_x,
        s3_y,
        color="y",
        marker="x",
        alpha=0.25,
        label="Selected N=20, K=" + str(boolean_network.get_avg_k()),
    )
    plt.scatter(
        s2_x,
        s2_y,
        color="r",
        marker="+",
        label="Excluding tautology and contradiction (sample, ~5%)",
    )
    plt.scatter(
        s1_x,
        s1_y,
        color="k",
        marker=".",
        s=10,
        label="All Boolean functions (sample, ~5%)",
    )
    plt.legend(loc="upper left", draggable=True)
    plt.xlabel("log(run-in duration to cycle start + 1)")
    plt.ylabel("log(terminal cycle length + 1)")
    plt.title(
        "Boolean networks: N="
        + str(bn_list_n)
        + ", K<2\nAfter run-in's from 202 initial conditions per net"
    )
    print(
        "Number of cycles in behavior of selected net: "
        + str(len(boolean_network.bn_collapsed_cycles))
    )
    plt.show()
    print(
        "Pseudorandom numbers generated using SystemRandom() - Version: "
        + str(SystemRandom.VERSION)
        + ", Python 3.12"
    )


# # microRNA plotting stuff
# def plot_target_sites(a_bn_mir: MicroRNAAbundantBooleanNetwork, colors):
#     if not a_bn_mir.target_sites_set_up:
#         a_bn_mir.setup_target_site_lists()
#     scaling_factor = 8
#     fig = plt.figure()
#     ax = plt.subplot()
#     for plot_i, each_abundant_node in zip(range(a_bn_mir.get_number_of_abundances()),
#                                           a_bn_mir.get_all_abundant_nodes()):
#         x_set = set()
#         plot_y = a_bn_mir.get_number_of_abundances() * 2 + 1 - plot_i * 2
#         ax.plot([0, len(each_abundant_node.three_prime_utr)], [plot_y, plot_y], color='k', alpha=0.4)
#         ax.scatter(2 * (-10), plot_y, color=colors[plot_i], marker='.')
#         # plt.scatter([plot_k for plot_k in range(len(each_abundant_node.three_prime_utr))], [plot_y for plot_L in range(len(each_abundant_node.three_prime_utr))], s=1, color='k', alpha=0.1)  # maybe make it look more like nucleotides ... actually, ### Even this one is quite slow
#         # for plot_m in range(len(each_abundant_node.three_prime_utr)):
#         #     ax.scatter(plot_m, plot_y, marker="$" + each_abundant_node.three_prime_utr[plot_m] + "$", s=4, color='k', alpha=0.8)  # maybe make it look more like nucleotides ... actually, DONE, should be little letters  ### SEEMS TO BE QUITE SLOW
#         counter = 0
#         for plot_j in range(len(each_abundant_node.target_sites)):
#             counter += 1
#             # print("...")
#             x_1 = each_abundant_node.target_sites[plot_j].start_index
#             x_2 = each_abundant_node.target_sites[plot_j].end_index
#             avg_x = (x_1 + x_2) / 2
#             x_set.add(avg_x)
#             site_size = ((x_2 - x_1) ** 2) * scaling_factor
#             tick_color = colors[
#                 a_bn_mir.micro_rna_nodes.index(each_abundant_node.target_sites[plot_j].micro_rna_node_reference)]
#             plt.scatter(avg_x, plot_y, color=tick_color, alpha=0.8, marker='|', s=site_size)
#             # TODO a fillbetween showing size of each site
#         # x_plot = [each_x for each_x in x_set]
#         # plt.scatter(x_plot, [plot_y for each_x in x_plot], color='r', alpha=0.8, marker='.')  # much faster
#         # print("Counter: " + str(counter))
#     fig.set_size_inches(6, 9)
#     plt.show()
