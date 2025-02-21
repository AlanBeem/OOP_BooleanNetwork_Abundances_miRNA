import copy
from random import shuffle
from matplotlib import cm
from package.boolean_networks import *


# As a starting point for realistic numerical representation of RNA abundance, distributions other facts TBD. 08-07-2024
# From Copy_08_06_2024_CausalNetworkAndMiRNA_Models_Current.py, w/ modification
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Qiagen -- https://www.qiagen.com/us/resources/faq?id=06a192c2-e72d-42e8-9b40-3171e1eb4cb8&lang=en:
# "The RNA content and RNA make up of a cell depend very much on its developmental stage and the type of cell. To
# estimate the approximate yield of RNA that can be expected from your starting material, we usually calculate that a
# typical mammalian cell contains 10–30 pg total RNA.
#
# The majority of RNA molecules are tRNAs and rRNAs. mRNA accounts for only 1–5% of the total cellular RNA although the
# actual amount depends on the cell type and physiological state. Approximately 360,000 mRNA molecules are present in a
# single mammalian cell, made up of approximately 12,000 different transcripts with a typical length of around 2 kb.
# Some mRNAs comprise 3% of the mRNA pool whereas others account for less than 0.1%. These rare or low-abundance mRNAs
# may have a copy number of only 5–15 molecules per cell."
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#       Therefore, for each abundant RNA, a maximum abundance of [5, 15] * [1, 30] = [5, 450]
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#   Consider:   Matrix multiplication computation of RNA degradation, W * A * G. W is some weight function (such as from
#               a Laplace transform). A is abundances. G is a composite function, and can include
#               Boolean statements, where the result is an arbitrary data structure or a matrix of values that must be
#               interpreted across e.g. data objects each a subset of the columns, with each entry a scalar.
#               Additionally, let some function define transcription; A(t+1) = A(t) + Transcription(t, A(t)) - WAG(t);
#               basically, we can compute this programmatically.
#
# Regarding G:
# From reading Python documentation about in/out and strings and printing
# "See also pickle - the pickle module
# Contrary to JSON, pickle is a protocol which allows the serialization of arbitrarily complex Python objects. As such,
# it is specific to Python and cannot be used to communicate with applications written in other languages. It is also
# insecure by default: deserializing pickle data coming from an untrusted source can execute arbitrary code, if the data
# was crafted by a skilled attacker."
# #  So, G as contemplated could be a "pickled" collection of Python objects, for example.
#
#   Such an Euler-ian model could be an object-oriented program consistent with known models of aspects of RNA abundance
#   and micro-RNAs.
#
#   For general purposes a more 'numerical' approach would be more efficient, but for now, "Extreme OOP"
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# Classes from Copy_08_06_2024_CausalNetworkAndMiRNA_Models_Current.py    w/ modification                              #
class AbundantNode:
    def __init__(self):
        self.maximum_abundance = 0
        self.transcription_rate = 0
        self.current_degradation_rate = 0
        self.base_deg_rate = 0
        self.current_abundance = 0
        self.abundance_velocity = 0
        # self.location_abundances = []
        #
        self.genomic_5_prime = ["", ""]  # reverse complements of the same sequences
        self.genomic_3_prime = [
            "",
            "",
        ]  # reverse complements of the same sequences  # 09-06
        #
        self.five_prime_utr = None
        self.orf = None
        self.three_prime_utr = None
        self.rna_binding_protein_sites = []
        self.target_sites = []
        self.three_prime_utr_accessibility = None

    def __str__(self):
        out_string = self.__repr__()
        out_string += "\nSequences:"
        out_string += "\n" + self.three_prime_utr
        return out_string

    def random_initialization(self, given_cycle_length: int):
        # Average human transcription rate is 2 - 6 kb/min, 3.5kb/min average (Bionumbers)
        # Average human transcript length is: ~2000nt (Bionumbers) # TODO double check source
        # Rate of transcription: 3500nt/min (Bionumbers) # TODO double check source
        # A reported density of RNA polymerase on DNA is 1/270nt # TODO find source
        num_on_steps_to_maximum = SystemRandom().gauss(300, 60) / (
            24 * 60 / given_cycle_length
        )
        num_off_steps_to_5 = SystemRandom().gauss(600, 60) / (
            24 * 60 / given_cycle_length
        )
        self.maximum_abundance = SystemRandom().randrange(
            5, 16, 1
        ) * SystemRandom().randrange(1, 30, 1)
        self.transcription_rate = self.maximum_abundance / num_on_steps_to_maximum
        self.current_degradation_rate = math.pow(0.05, 1 / num_off_steps_to_5)
        self.base_deg_rate = math.pow(
            0.05, 1 / num_off_steps_to_5
        )  # self.current_degradation_rate

    def random_setup(self, given_cycle_length: int):
        self.random_initialization(given_cycle_length)

    def reset_node(self):
        self.current_abundance = 0
        self.current_degradation_rate = self.base_deg_rate

    def set_abundance(self, new_abundance):
        # self.abundance_velocity = int(min(self.maximum_abundance, new_abundance)) - self.current_abundance
        # # So, there is subsumed causality relating to maximum abundance, but it is not abundance_velocity
        # self.current_abundance = int(min(self.maximum_abundance, new_abundance))
        self.current_abundance = max(0, int(new_abundance))

    def update(self, current_state: bool):
        self.set_abundance(  # for Euler's method
            self.current_abundance * self.current_degradation_rate
            + self.transcription_rate * int(current_state)
        )
        self.current_degradation_rate = self.base_deg_rate  # 08-25-2024


class BooleanAbundantNode(AbundantNode):
    def __init__(
        self,
        boolean_network: BooleanNetwork,
        input_node_indices: list[int],
        output_function: list[bool],
    ):
        super().__init__()
        self.boolean_network = boolean_network
        self.input_node_indices = input_node_indices
        self.input_rows = get_all_list_bool(len(input_node_indices), [False, True])
        self.output_function = output_function
        self.current_state = False
        #
        self.deg_input_node_indices = input_node_indices
        self.deg_input_rows = self.input_rows
        self.deg_output_functions = []

    def setup_graded_degradation_rates(self, number_of_deg_output_functions: int):
        for sgdr_i in range(number_of_deg_output_functions):
            self.deg_output_functions.append(
                get_random_list(2 ** len(self.input_node_indices), [False, True])
            )
        # TODO Make graded on both ends (transcription and degradation)

    def update_current_state(self):
        same_i = 0  # 08-26-2024
        # same_boolean = False
        # this finds the first matching input row (state of inputs)
        for each_row in self.input_rows:
            same_boolean = True
            for same_i, each_element in zip(range(len(each_row)), each_row):
                if (
                    self.boolean_network.current_states_list[-1][
                        self.input_node_indices[same_i]
                    ]
                    != each_element
                ):
                    same_boolean = False
                    break
            if same_boolean:
                self.current_state = self.output_function[same_i]
                break
            same_i += 1

    # can modulate sending True to each abundant node randomly from central loop, keep with format as used
    def update(self, current_state: bool):
        if self.current_state:
            self.set_abundance(  # for Euler's method
                self.current_abundance * self.current_degradation_rate
                + self.transcription_rate * int(current_state)
            )
        self.current_degradation_rate = self.base_deg_rate  # 08-25-2024


class NestedAbundantNode:  # from an older version, with changes
    """NestedAbundantNode represents a node with an assigned state, for each step, the states of the nested nodes are
    determined by the state of the Boolean node and the truth table of the nested nodes states: starting from the
    """

    def __init__(
        self,
        num_nested_nodes: int,
        given_cycle_length: int,
        random_if_true: bool,
        num_possible_states_to_average_per_update: int,
    ):
        self.random_if_true = random_if_true
        self.input_rows = get_all_list_bool(num_nested_nodes, [False, True])
        self.output_column = get_random_list(2**num_nested_nodes, [False, True])
        while all(
            [
                self.output_column[setup_i] == self.output_column[0]
                for setup_i in range(len(self.output_column))
            ]
        ):
            self.output_column = get_random_list(2**num_nested_nodes, [False, True])
        self.abundant_nodes = []
        for i in range(num_nested_nodes):
            self.abundant_nodes.append(AbundantNode())
            self.abundant_nodes[-1].random_setup(given_cycle_length)
        self.num_to_average_per_update = (
            num_possible_states_to_average_per_update  # Vary over steps (over BN Hamm.)
        )
        self.current_abundance = 0
        self.substate_index_list = list(range(len(self.input_rows)))
        shuffle(self.substate_index_list)

    def display(self) -> None:
        disp_string = "inputs".ljust(len(self.input_rows[0])) + "\toutputs\n\n"
        for i in range(len(self.input_rows)):
            disp_string += (
                "".join([str(int(b)) for b in self.input_rows[i]])
                + f"\t{self.output_column[i]}\n"
            )
        print(disp_string)

    def update(
        self, current_state: bool
    ):  #                                displaying the abundance of the nested nodes was also good, this morphed a lot over time
        if not self.random_if_true:  # updated 09-07-2024 pun not intended
            # u_index = -1  # moved outside loop
            for u_i in range(len(self.abundant_nodes)):
                nan_temp_current_abundance = self.abundant_nodes[u_i].current_abundance
                nan_temp_abundances = []
                u_index = -1
                for u_k in range(
                    self.num_to_average_per_update
                ):  # these loops could be more efficient
                    u_bool = True
                    while u_bool:
                        u_index += 1
                        if (
                            self.output_column[
                                self.substate_index_list[
                                    u_index % len(self.output_column)
                                ]
                            ]
                            == current_state
                        ):
                            self.abundant_nodes[u_i].current_abundance = (
                                nan_temp_current_abundance  # this averages together multiple possible next steps
                            )
                            self.abundant_nodes[u_i].update(
                                self.input_rows[
                                    self.substate_index_list[
                                        u_index % len(self.input_rows)
                                    ]
                                ][u_i]
                            )
                            nan_temp_abundances.append(
                                self.abundant_nodes[u_i].current_abundance
                            )
                            u_bool = False
                self.abundant_nodes[u_i].current_abundance = (
                    sum(nan_temp_abundances) / self.num_to_average_per_update
                )
            self.current_abundance = sum(
                [node.current_abundance for node in self.abundant_nodes]
            ) / len(
                self.abundant_nodes
            )  # added 2025
        else:
            for u_j in range(len(self.abundant_nodes)):
                nan_temp_current_abundance = self.abundant_nodes[u_j].current_abundance
                nan_temp_sum_abundances = 0
                for u_k in range(
                    self.num_to_average_per_update
                ):  # this averages possible input states given output state
                    u_bool = True
                    while u_bool:
                        u_index = SystemRandom().randrange(len(self.output_column))
                        if self.output_column[u_index] == current_state:
                            self.abundant_nodes[u_j].current_abundance = (
                                nan_temp_current_abundance
                            )
                            self.abundant_nodes[u_j].update(
                                self.input_rows[u_index][u_j]
                            )
                            nan_temp_sum_abundances += self.abundant_nodes[
                                u_j
                            ].current_abundance
                            u_bool = False
                self.abundant_nodes[u_j].current_abundance = (
                    nan_temp_sum_abundances / self.num_to_average_per_update
                )
            self.current_abundance = sum(
                [node.current_abundance for node in self.abundant_nodes]
            ) / len(
                self.abundant_nodes
            )  #


# state encoding (1, 0) to continuous (vector of linearly increasing or decreasing values eg transcription_rate * 0.9 ^ n )


class AbundantBooleanNetwork(BooleanNetwork):
    def __init__(
        self,
        num_nodes,
        possible_functions_indices_list: list[int],
        maximum_inputs_number: int,
        iterations_limit: int,
        setup_add_cycles: int,
    ):
        super().__init__(
            num_nodes,
            possible_functions_indices_list,
            maximum_inputs_number,
            iterations_limit,
        )
        self.abundant_nodes_1 = []  # list[AbundantNode]
        self.abundant_nodes_2 = (
            []
        )  # list[AbundantNode] (each from a NestedAbundantNode) (? TODO check these)
        self.boolean_abundant_nodes = []
        self.nested_abundant_nodes = []
        self.abundant_n_assignments = []
        self.nested_abundant_n_assignments = []
        self.current_abundances_list = None
        self.setup(setup_add_cycles)
        self.num_abundant_nodes = 0

    def setup(self, setup_add_cycles: int):
        for s_i in range(setup_add_cycles):
            self.add_cycle()

    def reset_all_abundant_nodes(self):
        for aka_reset_1 in range(len(self.abundant_nodes_1)):
            self.abundant_nodes_1[aka_reset_1].reset_node()
        for aka_reset_2 in range(len(self.abundant_nodes_2)):
            self.abundant_nodes_2[aka_reset_2].reset_node()
        for aka_reset_b in range(len(self.boolean_abundant_nodes)):
            self.boolean_abundant_nodes[aka_reset_b].reset_node()

    def add_abundant_nodes(self, number_of_abundant_nodes: int):
        aan_average_cycle_length = 0
        for each_cycle_record in self.bn_collapsed_cycles.cycle_records:
            aan_average_cycle_length += len(each_cycle_record)
        if len(self.bn_collapsed_cycles) > 0:
            aan_average_cycle_length = aan_average_cycle_length / len(
                self.bn_collapsed_cycles
            )
        else:
            aan_average_cycle_length = 1
        for aan_i in range(number_of_abundant_nodes):
            self.abundant_nodes_1.append(AbundantNode())
            if aan_average_cycle_length != 1:
                self.abundant_nodes_1[-1].random_setup(self.longest_cycle_length())
            self.abundant_n_assignments.append(SystemRandom().randrange(len(self)))
        self.num_abundant_nodes += number_of_abundant_nodes

    def add_nested_nodes(
        self,
        number_of_nested_nodes: int,
        nodes_per_node: int,
        random_if_true: bool,
        num_to_average_per_update: int,
        deterministic_average: bool = False,
    ):
        ann_average_cycle_length = 0
        for each_cycle_record in self.bn_collapsed_cycles.cycle_records:
            ann_average_cycle_length += len(each_cycle_record)
        ann_average_cycle_length = ann_average_cycle_length / len(
            self.bn_collapsed_cycles
        )
        for ann_i in range(number_of_nested_nodes):
            self.nested_abundant_nodes.append(
                NestedAbundantNode(
                    nodes_per_node,
                    int(ann_average_cycle_length),
                    random_if_true,
                    num_to_average_per_update,
                )
            )
            for ann_L in range(len(self.nested_abundant_nodes[-1].abundant_nodes)):
                self.abundant_nodes_2.append(
                    self.nested_abundant_nodes[-1].abundant_nodes[ann_L]
                )
                self.nested_abundant_n_assignments.append(
                    SystemRandom().randrange(len(self))
                )
        self.num_abundant_nodes += number_of_nested_nodes

    def add_a_boolean_abundant_node(
        self, input_node_indices: list[int], output_function: list[bool]
    ):
        self.boolean_abundant_nodes.append(
            BooleanAbundantNode(self, input_node_indices, output_function)
        )

    def animate_boolean_states_list(
        self,
        states_to_animate: list[list[bool]],
        number_of_cycles: int = 1,
        reset_abundances_if_true: bool = False,
    ):
        """Gives values to current_abundances_list, use for setting up abundances.
        For all AbundantNode s of this instance, over the given set of states, animate abundance and display a plot
        using matplotlib."""
        num_steps = len(states_to_animate) * number_of_cycles + 1
        if reset_abundances_if_true:
            self.reset_all_abundant_nodes()
        self.current_abundances_list = [
            [0 for _ in range(self.get_number_of_abundances())]
            for __ in range(num_steps)
        ]
        for asl_i in range(num_steps):
            for asl_j in range(len(self.abundant_nodes_1) - 1):
                self.abundant_nodes_1[asl_j].update(
                    states_to_animate[asl_i % len(states_to_animate)][
                        self.abundant_n_assignments[asl_j]
                    ]
                )
            for asl_k in range(
                len(self.nested_abundant_nodes)
            ):  # field of ab. nodes: abundant_nodes_2
                self.nested_abundant_nodes[asl_k].update(
                    states_to_animate[asl_i % len(states_to_animate)][
                        self.nested_abundant_n_assignments[asl_k]
                    ]
                )
            for acsl_b in range(len(self.boolean_abundant_nodes)):
                self.boolean_abundant_nodes[acsl_b].update_current_state()
                self.boolean_abundant_nodes[acsl_b].update(True)
            for ab_i, each_node in zip(
                range(self.get_number_of_abundances()), self.get_plottable_nodes()
            ):
                self.current_abundances_list[asl_i][ab_i] = each_node.current_abundance

    def initialize_abundances(self, cycle_to_animate: CycleRecord | None):
        number_of_cycles_for_initial_values = 10
        self.animate_boolean_states_list(
            cycle_to_animate.cycle_states_list,
            number_of_cycles_for_initial_values,
            True,
        )
        initial_abundances = [
            get_auc_list(self.current_abundances_list)[ia_i]
            / (number_of_cycles_for_initial_values * len(cycle_to_animate))
            for ia_i in range(self.get_number_of_abundances())
        ]
        for ia_j, each_ab_node in zip(
            range(self.get_number_of_abundances()), self.get_all_abundant_nodes()
        ):
            each_ab_node.current_abundance = initial_abundances[ia_j]

    def animate_a_cycle(
        self, cycle_to_animate: CycleRecord, number_of_cycle_lengths: int | float
    ):
        self.initialize_abundances(cycle_to_animate)
        self.animate_boolean_states_list(
            cycle_to_animate.cycle_states_list[0 : len(cycle_to_animate)],
            int(number_of_cycle_lengths),
            False,
        )

    def calculate_abundances_from_given_abundances(
        self,
        current_abundances: list[float],
        states_to_animate: list[list[bool]],
        number_of_cycles: int,
        plot_title: str,
        reset_colors_if_true: bool,
    ):
        # TODO finish this method  ### currently in plotting in jupyter notebook
        pass

    def get_number_of_abundances(self):
        # return len(self.get_all_abundant_nodes())
        return self.num_abundant_nodes

    def get_all_abundant_nodes(self):
        return (
            self.abundant_nodes_1 + self.abundant_nodes_2 + self.boolean_abundant_nodes
        )

    def get_plottable_nodes(self):
        return self.abundant_nodes_1 + self.nested_abundant_nodes

    def from_boolean_network(self, boolean_network: BooleanNetwork):
        self.total_advance_states = boolean_network.total_advance_states
        self.iterations_limit = boolean_network.iterations_limit
        self.current_states_list = copy.deepcopy(boolean_network.current_states_list)
        self.list_of_all_run_ins = copy.deepcopy(boolean_network.list_of_all_run_ins)
        self.list_of_all_indices_of_first_cycle_state = copy.deepcopy(
            boolean_network.list_of_all_indices_of_first_cycle_state
        )
        self.nodes = copy.deepcopy(boolean_network.nodes)
        self.node_inputs_assignments = copy.deepcopy(
            boolean_network.node_inputs_assignments
        )
        self.bn_collapsed_cycles = copy.deepcopy(boolean_network.bn_collapsed_cycles)
        self.bn_trajectories = copy.deepcopy(boolean_network.bn_trajectories)
        self.total_unit_perturbations_transition_matrix = copy.deepcopy(
            boolean_network.total_unit_perturbations_transition_matrix
        )
        self.cycles_unit_perturbations_transition_matrix = copy.deepcopy(
            boolean_network.cycles_unit_perturbations_transition_matrix
        )
        self.cycles_unit_perturbations_records = copy.deepcopy(
            boolean_network.cycles_unit_perturbations_records
        )
        self.t_trajectories_unit_perturbations_records = copy.deepcopy(
            boolean_network.t_trajectories_unit_perturbations_records
        )
        self.u_trajectories_unit_perturbations_records = copy.deepcopy(
            boolean_network.u_trajectories_unit_perturbations_records
        )
