# helper functions for compositions of boolean_networks.py and micro_rna.py  (Use more descriptive names!) TODO Names
import copy
import math
from random import shuffle
import re
from random import SystemRandom
import matplotlib.pyplot as plt
import numpy as np
import numpy.random

# from Bio import SeqIO
from matplotlib import cm

# from boolean_networks import CycleRecord, CollapsedCycles

nt = ["A", "U", "G", "C"]


def skew_population(observations: list[int]):
    """skewness takes a 1D list of numerical observations, such as cycle length, and returns sample skewness.
    Where skewness would encounter division by zero, skewness returns None."""
    if (
        (len(observations) <= 1)
        or (observations is None)
        or (min(observations) == max(observations))
    ):
        return None
    else:
        sample_n = len(observations)
        x_bar = sum(observations) / sample_n
        m3 = 0
        m2 = 0
        for each_observation in observations:
            m3 += (each_observation - x_bar) ** 3
            m2 += (each_observation - x_bar) ** 2
        # for s_m in [m2, m3]:
        #     s_m = s_m / sample_n  ### 08-02-2024
        m2 = m2 / sample_n
        m3 = m3 / sample_n
        return m3 / (m2 ** (3 / 2))


def skew_sample(observations: list[int]):
    """skewness takes a 1D list of numerical observations, such as cycle length, and returns sample skewness.
    Where skewness would encounter division by zero, skewness returns None."""
    # sample formula from https://www.macroption.com/skewness-formula/
    if (
        (len(observations) <= 1)
        or (observations is None)
        or (min(observations) == max(observations))
    ):
        return None
    else:
        sample_n = len(observations)
        x_bar = sum(observations) / sample_n
        m3 = 0
        m2 = 0
        for each_observation in observations:
            m3 += (each_observation - x_bar) ** 3
            m2 += (each_observation - x_bar) ** 2
        return (sample_n * math.pow(sample_n - 1, 1 / 2) / (sample_n - 2)) * (
            m3 / (m2 ** (3 / 2))
        )


def get_random_sequence(length: int | list, seq_elements: list[str]):
    grs_output_sequence = ""  # Could just append this to seq_elements and iterate through 1 less of len(seq_elements)
    if isinstance(length, list):
        length = len(length)
    for grs_i in range(length):
        grs_output_sequence += seq_elements[
            SystemRandom().randrange(0, len(seq_elements), 1)
        ]
    return grs_output_sequence


# return "".join([seq_elements... ])   ### with below get_random_list(), some available Python syntax


def get_random_list(length: int, seq_elements: list):
    return [
        seq_elements[SystemRandom().randrange(0, len(seq_elements), 1)]
        for grl_i in range(length)
    ]


def get_consensus_sequence_list(sequences: list[list], seq_elements: list):
    # TODO mixed bases
    #   see nucleotide_complement
    # another idea for this is to make a structure of [[0 for length seq_elements] for length sequences], populate the
    # values over the sequences, then ... report total?
    # cons_seq_list = None
    # if len(sequences) > 0:
    #     cons_seq_list = []
    #     element_counts = []
    #     for cons_seq_list_i in range(len(sequences[0])):
    #         element_counts.append([0 for element in seq_elements])
    #     for cons_seq_list_k in range(len(sequences)):
    #         for cons_seq_list_i in range(len(sequences[0])):
    #             for cons_seq_list_l in range(len(seq_elements)):
    #                 if sequences[cons_seq_list_k][cons_seq_list_i] == seq_elements[cons_seq_list_l]:
    #                     element_counts[cons_seq_list_i][cons_seq_list_l] += 1
    #     for cons_seq_list_q in range(len(element_counts)):
    #         max_count = 0
    #         index_max_count = -1
    #         for cons_seq_list_r in range(len(seq_elements)):
    #             if element_counts[cons_seq_list_q][cons_seq_list_r] > max_count:
    #                 max_count = element_counts[cons_seq_list_q][cons_seq_list_r]
    #                 index_max_count = cons_seq_list_r
    #         cons_seq_list.append(seq_elements[index_max_count])
    # return cons_seq_list
    return get_weighted_consensus_sequence_list(
        sequences, seq_elements, [1 for gcsl_i in range(len(sequences))]
    )


def get_weighted_consensus_sequence_list(
    sequences: list[list], seq_elements: list, weights_list: list[float]
):
    w_cons_seq_list = None
    if len(sequences) > 0:
        w_cons_seq_list = []
        element_counts = []
        for w_cons_seq_list_i in range(len(sequences[0])):
            element_counts.append([0 for element in seq_elements])
        for w_cons_seq_list_k in range(len(sequences)):
            for w_cons_seq_list_i in range(len(sequences[0])):
                for w_cons_seq_list_l in range(len(seq_elements)):
                    if (
                        sequences[w_cons_seq_list_k][w_cons_seq_list_i]
                        == seq_elements[w_cons_seq_list_l]
                    ):
                        element_counts[w_cons_seq_list_i][
                            w_cons_seq_list_l
                        ] += weights_list[w_cons_seq_list_k]
        for w_cons_seq_list_q in range(len(element_counts)):
            max_count = 0
            index_max_count = -1
            for w_cons_seq_list_r in range(len(seq_elements)):
                if element_counts[w_cons_seq_list_q][w_cons_seq_list_r] > max_count:
                    max_count = element_counts[w_cons_seq_list_q][w_cons_seq_list_r]
                    index_max_count = w_cons_seq_list_r
            w_cons_seq_list.append(seq_elements[index_max_count])
    return w_cons_seq_list


def get_consensus_sequence(
    sequences: list[str],
):  # 09-06-2024
    sequence_lists = []
    for each_seq in sequences:
        sequence_lists.append([])
        for each_nt in each_seq:
            sequence_lists[-1].append(each_nt)
    out_sequence = get_consensus_sequence_list(sequence_lists, nt)
    return "".join(out_sequence)


def get_boolean_complementary_sequence_list(input_boolean_sequence: list[bool]):
    return [
        not input_boolean_sequence[gbcs_i]
        for gbcs_i in range(len(input_boolean_sequence))
    ]


def get_hamming_distance(sequence_a, sequence_b):
    """get_hamming_distance accepts any objects addressable by index, such as str or list, and counts the number of
    elements that are not the same and returns that value (int), TODO edit cost function ?
    """
    if len(sequence_a) != len(sequence_b):
        print(
            f"sequences are of unequal lengths:\nsequence_a: {sequence_a}\nsequence_b: {sequence_b}"
        )
        return None
    else:
        out_hamming_distance = 0
        for hamming_i in range(len(sequence_a)):
            if sequence_a[hamming_i] != sequence_b[hamming_i]:
                out_hamming_distance += 1
        return out_hamming_distance


def get_all_list_bool(length: int, sequence_elements: list[bool] = [True, False]):
    """For a net of 3 Boolean functions:
    There are 2^3 possible states of [A,B,C]: [0,0,0] [0,0,1] [0,1,0] [0,1,1] [1,0,0] [1,0,1] [1,1,0] [1,1,1].
    get_all_list_bool(length, sequence_elements) returns a list of all possible list(bool) of given length.
    """
    galb_return_sequence_list = [[]]
    for galb_i in range(length):
        galb_temp_sequence_list = []
        for each_list in galb_return_sequence_list:
            for each_element in sequence_elements:
                galb_temp_sequence_list.append(each_list + [each_element])
        galb_return_sequence_list = galb_temp_sequence_list.copy()
    return galb_return_sequence_list


def list_bool_to_integer(
    list_of_bool: list[bool],
):  # 08-06-2024, related to CCTree (not currently using)
    # (0||1) * (2^index)  ### TODO alternatively named, below: bool_state_to_int
    return sum(
        [
            int(each_input_bool) * (2**lbti_index)
            for lbti_index, each_input_bool in zip(
                range(len(list_of_bool)), reversed(list_of_bool)
            )
        ]
    )


# Functions from Copy_08_06_2024_CausalNetworkAndMiRNA_Models_Current.py   w/ modification                             #
def get_all_sequences(sequence_length, sequence_elements):
    out_sequences = [""]
    while len(out_sequences[0]) < sequence_length:
        intermediate_sequences = []
        for sequence in out_sequences:
            for sequence_element in sequence_elements:
                intermediate_sequences.append(sequence + str(sequence_element))
        out_sequences = intermediate_sequences.copy()
    return out_sequences.copy()


def get_hamming_diameter(list_of_sequences):
    ghd_maximum = 0
    for ghd_i in range(len(list_of_sequences)):
        for ghd_j in range(len(list_of_sequences)):
            if ghd_i != ghd_j:
                ghd_temp_value = get_hamming_distance(
                    list_of_sequences[ghd_i], list_of_sequences[ghd_j]
                )
                if ghd_temp_value > ghd_maximum:
                    ghd_maximum = ghd_temp_value
    return ghd_maximum


def get_average_hamming_distances(list_of_sequences):
    gahd_total_distance = 0
    gahd_temp_distance = 0
    gahd_total_comparisons = 0
    for gahd_i in range(len(list_of_sequences)):
        gahd_temp_distance = 0
        for gahd_j in range(len(list_of_sequences)):
            if gahd_i != gahd_j:  # But why not compare it to itself?
                gahd_temp_distance += get_hamming_distance(
                    list_of_sequences[gahd_i], list_of_sequences[gahd_j]
                )
        gahd_total_distance += gahd_temp_distance / (len(list_of_sequences) - 1)
    return gahd_total_distance / (len(list_of_sequences) - 1)


def get_ordinal_string(ordinal_input_integer, append_suffix_if_true):
    if ordinal_input_integer is None:
        gos_output_string = "None"
        suffix = ""
    else:
        gos_output_string = str(int(ordinal_input_integer))
        if 20 > int(ordinal_input_integer) > 10:
            suffix = "th"
        else:
            if int(ordinal_input_integer) % 10 == 1:
                suffix = "st"
            elif int(ordinal_input_integer) % 10 == 2:
                suffix = "nd"
            elif int(ordinal_input_integer) % 10 == 3:
                suffix = "rd"
            else:
                suffix = "th"
    return str(gos_output_string + suffix)


def nucleotide_complement(an_nt, complement_dna_if_true=False):
    if an_nt.islower():
        an_nt = an_nt.upper()  # 08-27-2024
    if complement_dna_if_true:
        pairs = [
            ["A", "T"],
            ["C", "G"],
            ["N", "N"],
            ["-", "-"],
        ]  # [["A"], ["T"], ["C"], ["G"]],
    else:
        pairs = [["A", "U"], ["C", "G"], ["N", "N"], ["-", "-"]]
    if not complement_dna_if_true and an_nt == "T":
        an_nt = "U"
    elif complement_dna_if_true and an_nt == "U":
        an_nt = "T"
    for i in range(len(pairs)):
        for j in range(len(pairs[i])):
            if an_nt == pairs[i][j]:
                return pairs[i][(j + 1) % len(pairs[i])]
    return SystemRandom().choice(
        nt
    )  # ???? but really this won't execute unless this is unmatched
    # return an_nt


# sequences = ["AAA", "AUA", "UUA", "GGC", "CCC", "N", "NNCCAAGGUU--", "WAAAUU---NNNA{}[]"]
# for each_str in sequences:
#     print("input: " + each_str)
#     print("".join([nucleotide_complement(each_str[es_i], False) for es_i in range(len(each_str))]))


def get_rna_reverse_complement(input_nt_sequence: str):
    return "".join(
        [
            nucleotide_complement(input_nt_sequence[grc_i], False)
            for grc_i in range(len(input_nt_sequence) - 1, -1, -1)
        ]
    )


def get_rna_complement(input_nt_sequence: str):  # 08-27-2024
    input_nt_sequence = input_nt_sequence.strip()
    out_sequence_nt = ""
    for each_nt in input_nt_sequence:
        out_sequence_nt += nucleotide_complement(each_nt, False)
    return out_sequence_nt


# Pasted from micro_rna_v2_MICRO_RNA_COMMENTS_FINAL_SAVE__08_10_2024.py
class RegSeqGenerator:
    def __init__(self, master_regulatory_sequence_length, regulatory_letters_list):
        self.master_reg_seq = ""
        self.sequence_elements = regulatory_letters_list.copy()
        self.regulatory_sequences = []
        for rsg_i in range(master_regulatory_sequence_length):
            self.master_reg_seq += str(
                self.sequence_elements[
                    SystemRandom().randrange(0, len(self.sequence_elements), 1)
                ]
            )
        self.walk_index = 0

    def get_random_reg_seq(self, reg_seq_length: int):
        self.regulatory_sequences.append(
            "".join(
                [
                    self.sequence_elements[
                        SystemRandom().randrange(0, len(self.sequence_elements))
                    ]
                    for rsg_i in range(reg_seq_length)
                ]
            )
        )
        return self.regulatory_sequences[-1]

    def get_subseq_of_master_reg_seq(
        self, reg_seq_length: int, index_of_starting_nucleotide: int
    ):
        """this treats the master regulatory sequence indices as circular and returns a subsequence|length ==
        reg_seq_length"""
        self.regulatory_sequences.append(
            "".join(
                [
                    self.master_reg_seq[rsg_j % len(self.master_reg_seq)]
                    for rsg_j in range(
                        index_of_starting_nucleotide,
                        index_of_starting_nucleotide + reg_seq_length + 1,
                        1,
                    )
                ]
            )
        )
        return self.regulatory_sequences[-1]

    def get_walked_random_reg_seq(
        self,
        reg_seq_length: int,
        random_walk_start: float,
        random_walk_end: float,
        sticky_index: float,
    ):
        """this treats the master regulatory sequence indices as circular and returns a sequence composed of random
        sequence elements and nucleotides from random walks of self.master_reg_seq: str
        """
        gwrrs_walking = False
        gwrrs_walk_index = 0
        gwrrs_sequence = ""
        while len(gwrrs_sequence) < reg_seq_length:
            if not gwrrs_walking:
                if SystemRandom().random() < random_walk_start:
                    gwrrs_walking = True
                    if SystemRandom().random() < sticky_index:
                        gwrrs_walk_index = self.walk_index
                    else:
                        gwrrs_walk_index = SystemRandom().randrange(
                            len(self.master_reg_seq)
                        )
            else:
                if SystemRandom().random() < random_walk_end:
                    gwrrs_walking = False
                    self.walk_index = gwrrs_walk_index
            if gwrrs_walking:
                gwrrs_sequence += self.master_reg_seq[
                    gwrrs_walk_index % len(self.master_reg_seq)
                ]
                gwrrs_walk_index += 1
            else:
                gwrrs_sequence += self.sequence_elements[
                    SystemRandom().randrange(0, len(self.sequence_elements))
                ]
        self.regulatory_sequences.append(gwrrs_sequence)
        return self.regulatory_sequences[-1]


def get_color(plotting_index, helper_colors):
    if helper_colors is not None:
        return helper_colors[plotting_index % len(helper_colors)]
    else:
        return [0, 0, 0, 1]


# Move this to plotting helper class - -
def plot_abundances(
    abundances: list[list[float]],
    plot_title: str,
    colors_abundances: list,
    plot_from_zero_if_true: bool,
    vertical_markers: list[float],
    starting_color,
    ending_color,
    fig_size_in_inches: list[float | int] | None,
    xticks: list[int] | None,
):
    pa_plotting_index = 0
    fig = plt.figure()
    plt.title(plot_title)
    if xticks is None:
        plt.xticks([pa_ticks * 5 for pa_ticks in range(int(len(abundances) / 5))])
    else:
        plt.xticks(xticks)
    plt.xlabel("steps")
    plt.ylabel("abundance")
    for pa_r in range(len(abundances[0])):
        if plot_from_zero_if_true:
            plot_row = [0]
        else:
            plot_row = []
        for pa_q in range(len(abundances)):
            plot_row.append(abundances[pa_q][pa_r])
        # plt.plot(plot_row, color=get_color(pa_plotting_index, colors_abundances))
        plt.plot(plot_row, color=get_color(pa_plotting_index, colors_abundances))
        pa_plotting_index += 1
    if len(vertical_markers) > 0:  # TODO keyword argument?
        # fill_y = plt.ylim()[1]
        # fill_x1_1 = 0
        # fill_x2_1 = vertical_markers[0]
        # fill_x1_2 = vertical_markers[1]
        # fill_x2_2 = plt.xlim()[1]
        for each_marker in vertical_markers:
            plt.vlines(
                vertical_markers,
                ymin=plt.ylim()[0],
                ymax=plt.ylim()[1],
                linestyle="dashed",
                color="k",
                alpha=0.75,
            )
        # plt.fill_betweenx(fill_y, [fill_x1_1], [fill_x2_1], alpha=0.1, color=starting_color)
        # plt.fill_betweenx(fill_y, [fill_x1_2], [fill_x2_2], alpha=0.1, color=ending_color)
    if fig_size_in_inches is not None:
        fig.set_size_inches(fig_size_in_inches[0], fig_size_in_inches[1])

    plt.show()


def get_boolean_velocity(input_t, input_t_plus_one):
    if input_t != input_t_plus_one:
        return 1
    else:
        return 0


def get_boolean_velocity_list(
    input_states_list: list[list[bool]],
):  # -> list[list[int]]
    return [
        [
            get_boolean_velocity(
                input_states_list[gbvl_j][gbvl_i], input_states_list[gbvl_j + 1][gbvl_i]
            )
            for gbvl_i in range(len(input_states_list[0]))
        ]
        for gbvl_j in range(len(input_states_list) - 1)
    ]


def get_boolean_velocity_list_per_node(input_states_list: list[list[bool]]):
    return sum(
        [
            sum(
                [
                    get_boolean_velocity(
                        input_states_list[gbvl_j][gbvl_i],
                        input_states_list[gbvl_j + 1][gbvl_i],
                    )
                    for gbvl_i in range(len(input_states_list[0]))
                ]
            )
            / len(input_states_list[0])
            for gbvl_j in range(len(input_states_list) - 1)
        ]
    ) / (len(input_states_list) - 1)


def get_avg_boolean_velocity(list_of_states_lists: list[list[list[bool]]]):
    return sum(
        [
            get_boolean_velocity_list_per_node(list_of_states_lists[gabv_i])
            for gabv_i in range(len(list_of_states_lists))
        ]
    ) / len(
        list_of_states_lists
    )  # For a net's cycles, for example
    # 09-19-2024 this is intended for comparison of networks to each other


def get_least_common_multiple(iterable_ints):
    glcm_return_int = max(iterable_ints)
    iterable_ints = sorted(iterable_ints, reverse=True)
    for each_int in iterable_ints:
        if glcm_return_int % each_int != 0:
            glcm_return_int *= each_int
    return glcm_return_int


# 08-22-2024
def bool_state_to_int(state: list[bool]):
    return sum([int(state[bsi_i]) * (2**bsi_i) for bsi_i in range(len(state))])


def per_node_sequence_similarity(
    cycle_states_list_1, cycle_states_list_2, nodes_indices_to_compare: list[int] | None
):
    """Compares cycle_states_list_1 to cycle_states_list_2, for the nodes' indices in nodes_indices_to_compare, or for
    all nodes if nodes_indices_to_compare is None"""
    pnss_similarity_3 = 0
    if nodes_indices_to_compare is None:
        nodes_indices_to_compare = [
            nic_i for nic_i in range(len(cycle_states_list_1[0]))
        ]
    if len(nodes_indices_to_compare) == 0:
        pnss_divisor = 1
    else:
        pnss_divisor = len(nodes_indices_to_compare)
    if len(cycle_states_list_1) <= len(cycle_states_list_2):
        shorter_cycle_states_list = cycle_states_list_1
        longer_cycle_states_list = cycle_states_list_2
    else:
        shorter_cycle_states_list = cycle_states_list_2
        longer_cycle_states_list = cycle_states_list_1
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
            if pnss_start_position >= len(temp_sequence_1):
                pnss_bool_1 = False
    return pnss_similarity_3 / pnss_divisor


def get_2d_dimensions_swapped(matrix_2d: list[list]):
    return [
        [matrix_2d[g2dds_i][g2dds_j] for g2dds_i in range(len(matrix_2d))]
        for g2dds_j in range(len(matrix_2d[0]))
    ]


def get_auc_list(abundance_states_list: list[list[float]]):
    auc_from_trapezoid_rule = [0 for auc_i in range(len(abundance_states_list[0]))]
    rows_are_each_abundance = get_2d_dimensions_swapped(abundance_states_list)
    for er_i in range(len(rows_are_each_abundance)):
        for ev_i in range(len(rows_are_each_abundance[0]) - 1):
            auc_from_trapezoid_rule[er_i] += (
                rows_are_each_abundance[er_i][ev_i]
                + rows_are_each_abundance[er_i][ev_i + 1]
            ) / 2
    return auc_from_trapezoid_rule
