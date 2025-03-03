from package.boolean_networks import BooleanNetwork


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
    """returns the consensus sequence of sequences, each sequence with a weight,
    this assumes that all input sequences are of the same length as the first
    listed sequence, and requires that all sequences be at least the length of
    the first listed sequence."""
    w_cons_seq_list = None
    if len(sequences) > 0:
        w_cons_seq_list = []
        element_counts = [[0 for _ in seq_elements] for __ in sequences[0]]
        for i in range(len(sequences)):  # which sequence
            for j in range(len(sequences[0])):
                for k in range(len(seq_elements)):
                    if sequences[i][j] == seq_elements[k]:
                        element_counts[j][
                            k  # alternatively, use .index(element), but still O(n), n::=|sequence elements|
                        ] += weights_list[
                            i
                        ]  # which weight
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


def get_hamming_distance(seq_a, seq_b) -> int:
    """get_hamming_distance accepts any objects addressable by index, such as str or list, and counts the number of
    elements that are not the same and returns that value (int).\n\nWill crash for len(seq_a) != len(seq_b).
    """
    return sum([int(seq_a[i] != seq_b[i]) for i in range(len(seq_a))])


class HammingSetup:  # TODO make an interactive linear combination of sequences
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
