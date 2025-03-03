from package.boolean_networks import BooleanNetwork
from package.helper import HammingSetup, get_hamming_distance
import matplotlib.pyplot as plt


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


def plot_transition_s(
    boolean_network: BooleanNetwork,
    hamming_setup: HammingSetup,
    bv_colors: list,
    transition_index: int | None = None,
    text_plot_if_true: bool = False,
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
