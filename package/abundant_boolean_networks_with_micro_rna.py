import matplotlib.pyplot as plt
from package.boolean_networks import *
from package.abundant_boolean_networks import *
from package.micro_rna_2 import *


def plot_abundances_and_micro_rna(
    abundances: list[list[float]],
    micro_rna_abundances: list[list[list[float]]],
    plot_title: str,
    abundances_alpha: float,
    micro_rna_alpha: float,
    colors,
):
    if len(abundances) != len(micro_rna_abundances):
        print("Lengths of input lists do not match")
        return
    pa_plotting_index = 0
    plt.figure()
    plt.title(plot_title)
    plt.xticks(
        [
            pa_ticks * (len(abundances) / 10)
            for pa_ticks in range(int(len(abundances) / 10))
        ]
    )
    plt.xlabel("steps")
    plt.ylabel("abundance")
    plot_pri_xy = None
    plot_pre_xy = None
    plot_mirmir_xy = None
    plot_ago_loaded_xy = None
    plotted_maximum = 0
    for pa_r in range(len(abundances[0])):
        plot_row = []  # [0]
        for pa_q in range(len(abundances)):
            plot_row.append(abundances[pa_q][pa_r])
            if plot_row[-1] > plotted_maximum:
                plotted_maximum = plot_row[-1]
        plt.plot(plot_row, color=colors[pa_plotting_index], alpha=abundances_alpha)
        pa_plotting_index += 1
    for pa_t in range(len(micro_rna_abundances[0])):
        plot_pri = [0]  # [0]
        plot_pre = [0]  # [0]
        plot_mirmir = [0]  # [0]
        plot_ago_loaded = [0]  # [0]
        for pa_s in range(len(micro_rna_abundances)):
            plot_pri.append(micro_rna_abundances[pa_s][pa_t][0])
            plot_pre.append(micro_rna_abundances[pa_s][pa_t][1])
            plot_mirmir.append(micro_rna_abundances[pa_s][pa_t][2])
            plot_ago_loaded.append(micro_rna_abundances[pa_s][pa_t][3])
            if (
                max(plot_pri[-1], plot_pre[-1], plot_mirmir[-1], plot_ago_loaded[-1])
                > plotted_maximum
            ):
                plotted_maximum = max(
                    plot_pri[-1], plot_pre[-1], plot_mirmir[-1], plot_ago_loaded[-1]
                )
        plot_pri_xy = (0.618 * len(abundances), plot_pri[int(0.618 * len(abundances))])
        plot_pre_xy = (0.618 * len(abundances), plot_pre[int(0.618 * len(abundances))])
        plot_mirmir_xy = (
            0.618 * len(abundances),
            plot_mirmir[int(0.618 * len(abundances))],
        )
        plot_ago_loaded_xy = (
            0.618 * len(abundances),
            plot_ago_loaded[int(0.618 * len(abundances))],
        )
        plt.plot(
            plot_pri,
            linestyle="dashed",
            label="inline",
            alpha=micro_rna_alpha,
            color="k",
        )
        plt.plot(plot_pre, linestyle="dashed", alpha=micro_rna_alpha, color="k")
        plt.plot(plot_mirmir, linestyle="dashed", alpha=micro_rna_alpha, color="k")
        plt.plot(plot_ago_loaded, linestyle="dashed", alpha=micro_rna_alpha, color="k")
        pa_plotting_index += 1
    plt.annotate(
        "Primary micro-RNA", xy=plot_pri_xy, xytext=(plot_pri_xy[0], plot_pri_xy[1] + 5)
    )
    plt.annotate(
        "Precursor micro-RNA",
        xy=plot_pre_xy,
        xytext=(plot_pre_xy[0], plot_pre_xy[1] + 5),
    )
    plt.annotate(
        "miRNA:miRNA*",
        xy=plot_mirmir_xy,
        xytext=(plot_mirmir_xy[0], plot_mirmir_xy[1] + 5),
    )
    plt.annotate(
        "Ago2-loaded miRNA",
        xy=plot_ago_loaded_xy,
        xytext=(plot_ago_loaded_xy[0], plot_ago_loaded_xy[1] + 5),
    )
    # plt.xlim(plotted_maximum + 0.1*plotted_maximum)
    plt.show()
    # TODO make this (and other...) plotting method(s) composition that calls a get axes method, and plots that


# TODO Add endogenous microRNA 09-07 maybe snubbed for now, they're all endogenous, user needs to add as applicable and
#  keep track of them
# TODO add transfection (Requires abundances curves, can do default models from eg Qiagen, other reagents providers)
# A GUI flowchart experiment would be so cool-- project idea TODO
class MicroRNAAbundantBooleanNetwork(AbundantBooleanNetwork):
    """function indices include: Contradiction: 0, Tautology: 15, Random: 16, Alternating: 17, Sequence: 18.
    \nFor 16, 17, 18, additional setup is required (see boolean_networks.py)"""

    def __init__(
        self,
        num_nodes: int,
        boolean_function_indices: list[int],
        maximum_number_of_inputs: int,
        iterations_limit: int,
        setup_add_cycles: int,
    ):
        super().__init__(
            num_nodes,
            boolean_function_indices,
            maximum_number_of_inputs,
            iterations_limit,
            setup_add_cycles,
        )
        self.current_micro_rna_abundances_list = None
        self.additional_abundant_nodes = (
            []
        )  # these are all assigned True or some step calculated function
        self.micro_rna_nodes = []
        # TODO later overload methods to the point where all eg abundances data is provided, and all that's done is the
        #  calculations of rates
        self.endogenous_micro_rna_nodes = []
        self.target_sites_set_up = False
        # for target site evaluation using all-sequences comparisons
        self.all_seq_lists = []
        for all_seq_i in range(1, 10):
            self.all_seq_lists.append(
                get_all_sequences(all_seq_i, ["A", "U", "G", "C"])
            )

    # these methods, while nice in terms of having stuff do while getting the hang of a new (again) language, are really
    # not needed, and in this case, they still must all be addressed again to modify their sequence fields
    # [[09-07 at least a good marker for intent]]
    def add_additional_nodes(self, num_additional_nodes: int):
        self.additional_abundant_nodes.append(
            each_node
            for each_node in [AbundantNode() for pabn_i in range(num_additional_nodes)]
        )
        self.target_sites_set_up = False

    def add_abundant_nodes(self, number_of_abundant_nodes: int):
        super().add_abundant_nodes(number_of_abundant_nodes)
        # [SystemRandom().randrange(0, len(self)) for aan_i in range(number_of_abundant_nodes)])
        self.target_sites_set_up = False

    def add_nested_nodes(
        self,
        number_of_nested_nodes: int,
        nodes_per_node: int,
        random_if_true: bool,
        num_to_average_per_update: int,
    ):
        super().add_nested_nodes(
            number_of_nested_nodes,
            nodes_per_node,
            random_if_true,
            num_to_average_per_update,
        )
        self.target_sites_set_up = False

    def add_graded_nodes(
        self, list_of_inputs_assignments: list[list[int]]
    ):  # these 08-26-2024
        super().add_graded_nodes(list_of_inputs_assignments)
        self.target_sites_set_up = False

    def add_micro_rna_nodes(self, micro_rna_nodes: list[MicroRNANode] | int):
        """providing a list of microRNA nodes erases previous miRNA nodes"""
        if isinstance(micro_rna_nodes, int):
            for _ in range(micro_rna_nodes):
                self.micro_rna_nodes.append(node := MicroRNANode())
                node.random_setup(self.longest_cycle_length())
                node.sequence = SystemRandom().choices(nt, k=21)
            self.target_sites_set_up = False
        else:
            self.micro_rna_nodes = []  # 08-27-2024  # remove 2025
            for amrn_i in range(len(micro_rna_nodes)):
                self.micro_rna_nodes.append(micro_rna_nodes[amrn_i])
            self.target_sites_set_up = False

    def reset_all_abundant_nodes(self):  # TODO test
        for each_node in self.get_all_abundant_nodes():
            each_node.reset_node()
        for aka_reset_4 in range(len(self.micro_rna_nodes)):
            self.micro_rna_nodes[aka_reset_4].reset_node()
        self.current_abundances_list = []
        self.current_micro_rna_abundances_list = []

    def from_abundant_network(self, abundant_boolean_network: AbundantBooleanNetwork):
        # TODO and all this should be deepcopy?  09-07 yea or else it will alter the state of the from copy
        super().from_boolean_network(abundant_boolean_network)  # ??? 08-18-2024
        self.abundant_nodes_1 = abundant_boolean_network.abundant_nodes_1
        self.abundant_nodes_2 = abundant_boolean_network.abundant_nodes_2
        # self.graded_abundant_nodes = abundant_boolean_network.graded_abundant_nodes  ### tedious error
        self.additional_abundant_nodes = []
        # self.node_inputs_assignments # not correct assignments, this is for nodes, in super()
        self.abundant_n_assignments = (
            abundant_boolean_network.abundant_n_assignments
        )  # list of int
        self.nested_abundant_n_assignments = (
            abundant_boolean_network.nested_abundant_n_assignments
        )  # list[int]
        self.micro_rna_nodes = []
        self.target_sites_set_up = False

    def from_boolean_network(self, boolean_network: BooleanNetwork):
        super().from_boolean_network(boolean_network)
        self.current_micro_rna_abundances_list = None
        self.additional_abundant_nodes = (
            []
        )  # these are all assigned True or some step calculated function
        self.micro_rna_nodes = []
        self.endogenous_micro_rna_nodes = []
        self.target_sites_set_up = False
        # TODO redo from_ methods as overloaded constructor 09-07-2024

    def setup_target_site_lists(self):
        all_nodes = self.get_plottable_nodes()
        for i in range(len(all_nodes)):
            # abundant nodes includes miRNA nodes (making intronic might have some combinatorial effects)
            all_nodes[i].target_sites = get_three_prime_utr_target_sites(
                self.micro_rna_nodes, all_nodes[i]
            )

        for i in range(len(all_nodes)):
            for each_site in all_nodes[i].target_sites:
                self.micro_rna_nodes[
                    self.micro_rna_nodes.index(each_site.micro_rna_node_reference)
                ].target_sites.append(each_site)
        self.target_sites_set_up = True

    def knock_rates(self):
        # 3'UTR
        for each_node in self.get_plottable_nodes():
            for each_site in each_node.target_sites:
                # if three_prime_utr_target_site_to_scalar(each_site) > 1:
                #     print("Error, scalar evaluated to: " + str(three_prime_utr_target_site_to_scalar(each_site)))
                # each_node.current_degradation_rate = each_node.current_degradation_rate**(1 - three_prime_utr_target_site_to_scalar(each_site))
                # print(three_prime_utr_target_site_to_scalar(each_site))
                each_node.current_degradation_rate *= (
                    three_prime_utr_target_site_to_scalar(each_site)
                )

    def advance_abundant_state(self):
        self.current_abundances_list.append([])
        for aas_i in range(len(self.abundant_nodes_1)):
            self.abundant_nodes_1[aas_i].update(
                self.current_states_list[-1][self.abundant_n_assignments[aas_i]]
            )
            self.current_abundances_list[-1].append(
                self.abundant_nodes_1[aas_i].current_abundance
            )
        for aas_j in range(len(self.nested_abundant_nodes)):
            self.nested_abundant_nodes[aas_j].update(
                self.current_states_list[-1][self.nested_abundant_n_assignments[aas_j]]
            )
        for aas_k in range(len(self.abundant_nodes_2)):
            self.current_abundances_list[-1].append(
                self.abundant_nodes_2[aas_k].current_abundance
            )
        # for acsl_g in range(len(self.graded_abundant_nodes)):  # this method is unfinished, not all nodes, do isinsta...
        #     self.graded_abundant_nodes[acsl_g].update_transcription_rate(
        #         [self.current_states_list[-1][self.graded_abundant_n_assignments[acsl_g][acsl_g_2]] for acsl_g_2 in
        #          range(len(self.graded_abundant_n_assignments[acsl_g]))])
        #     self.graded_abundant_nodes[acsl_g].update(True)
        #     self.current_abundances_list[-1].append(self.graded_abundant_nodes[acsl_g].current_abundance)

    def advance_mirna_abundances(self):
        self.current_micro_rna_abundances_list.append(
            [[0, 0, 0, 0] for aka_v in range(len(self.micro_rna_nodes))]
        )
        for aka_c, each_micro_rna_node in zip(
            range(len(self.micro_rna_nodes)), self.micro_rna_nodes
        ):
            self.micro_rna_nodes[aka_c].update(True)
            self.current_micro_rna_abundances_list[-1][aka_c][
                0
            ] = each_micro_rna_node.current_abundance
            self.current_micro_rna_abundances_list[-1][aka_c][
                1
            ] = each_micro_rna_node.current_pre
            self.current_micro_rna_abundances_list[-1][aka_c][
                2
            ] = each_micro_rna_node.current_mirmir
            self.current_micro_rna_abundances_list[-1][aka_c][
                3
            ] = each_micro_rna_node.current_ago2_loaded

    def animate_boolean_states_list(
        self,
        states_to_animate: list[list[bool]],
        number_of_cycles: int,
        reset_abundances_if_true: bool,
    ):
        """For all AbundantNode s of this instance, over the given set of states, animate abundances with knockdown."""
        num_steps = len(states_to_animate) * number_of_cycles + 1
        self.current_states_list = []
        if reset_abundances_if_true:
            self.reset_all_abundant_nodes()
        self.current_abundances_list = []
        self.current_micro_rna_abundances_list = []
        for step_i in range(num_steps):
            self.current_states_list.append(
                states_to_animate[step_i % len(states_to_animate)]
            )
            self.advance_mirna_abundances()  #
            self.knock_rates()  #
            self.advance_abundant_state()  # Euler
