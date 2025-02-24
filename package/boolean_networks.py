from abc import ABC, abstractmethod
import copy
from package.bn_mir_helper_functions_V1 import *


# _BooleanFunction objects for k ≤ 2
class _BooleanFunction1:  # index: 0, k: 0
    @staticmethod
    def get_boolean(a, b):
        return False

    @staticmethod
    def get_k():
        return 0

    @staticmethod
    def get_number():
        return 1

    @staticmethod
    def get_expression_string():
        return "False"


class _BooleanFunction2:  # index: 1, k: 2
    @staticmethod
    def get_boolean(a, b):
        return a and b

    @staticmethod
    def get_k():
        return 2

    @staticmethod
    def get_number():
        return 2

    @staticmethod
    def get_expression_string():
        return "a and b"


class _BooleanFunction3:  # index: 2, k: 2
    @staticmethod
    def get_boolean(a, b):
        return a and (not b)

    @staticmethod
    def get_k():
        return 2

    @staticmethod
    def get_number():
        return 3

    @staticmethod
    def get_expression_string():
        return "a and (not b)"


class _BooleanFunction4:  # index: 3, k: 1
    @staticmethod
    def get_boolean(a, b):
        return a

    @staticmethod
    def get_k():
        return 1

    @staticmethod
    def get_number():
        return 4

    @staticmethod
    def get_expression_string():
        return "a"


class _BooleanFunction5:  # index: 4, k: 2
    @staticmethod
    def get_boolean(a, b):
        return (not a) and b

    @staticmethod
    def get_k():
        return 2

    @staticmethod
    def get_number():
        return 5

    @staticmethod
    def get_expression_string():
        return "(not a) and b"


class _BooleanFunction6:  # index: 5, k: 1
    @staticmethod
    def get_boolean(a, b):
        return b

    @staticmethod
    def get_k():
        return 1

    @staticmethod
    def get_number():
        return 6

    @staticmethod
    def get_expression_string():
        return "b"


class _BooleanFunction7:  # index: 6, k: 2
    @staticmethod
    def get_boolean(a, b):
        return a is not b

    @staticmethod
    def get_k():
        return 2

    @staticmethod
    def get_number():
        return 7

    @staticmethod
    def get_expression_string():
        return "a is not b"


class _BooleanFunction8:  # index: 7, k: 2
    @staticmethod
    def get_boolean(a, b):
        return a or b

    @staticmethod
    def get_k():
        return 2

    @staticmethod
    def get_number():
        return 8

    @staticmethod
    def get_expression_string():
        return "a or b"


class _BooleanFunction9:  # index: 8, k: 2
    @staticmethod
    def get_boolean(a, b):
        return (not a) and (not b)

    @staticmethod
    def get_k():
        return 2

    @staticmethod
    def get_number():
        return 9

    @staticmethod
    def get_expression_string():
        return "(not a) and (not b)"


class _BooleanFunction10:  # index: 9, k: 2
    @staticmethod
    def get_boolean(a, b):
        return a is b

    @staticmethod
    def get_k():
        return 2

    @staticmethod
    def get_number():
        return 10

    @staticmethod
    def get_expression_string():
        return "a is b"


class _BooleanFunction11:  # index: 10, k: 1
    @staticmethod
    def get_boolean(a, b):
        return not b

    @staticmethod
    def get_k():
        return 1

    @staticmethod
    def get_number():
        return 11

    @staticmethod
    def get_expression_string():
        return "not b"


class _BooleanFunction12:  # index: 11, k: 2
    @staticmethod
    def get_boolean(a, b):
        return (a is b) or a

    @staticmethod
    def get_k():
        return 2

    @staticmethod
    def get_number():
        return 12

    @staticmethod
    def get_expression_string():
        return "(a is b) or a"


class _BooleanFunction13:  # index: 12, k: 1
    @staticmethod
    def get_boolean(a, b):
        return not a

    @staticmethod
    def get_k():
        return 1

    @staticmethod
    def get_number():
        return 13

    @staticmethod
    def get_expression_string():
        return "not a"


class _BooleanFunction14:  # index: 13, k: 2
    @staticmethod
    def get_boolean(a, b):
        return (not a) or b  # a implies b

    @staticmethod
    def get_k():
        return 2

    @staticmethod
    def get_number():
        return 14

    @staticmethod
    def get_expression_string():
        return "(not a) or b"


class _BooleanFunction15:  # index: 14, k: 2
    @staticmethod
    def get_boolean(a, b):
        return (not a) or (not b)

    @staticmethod
    def get_k():
        return 2

    @staticmethod
    def get_number():
        return 15

    @staticmethod
    def get_expression_string():
        return "(not a) or (not b)"


class _BooleanFunction16:  # index: 15, k: 0
    @staticmethod
    def get_boolean(a, b):
        return True

    @staticmethod
    def get_k():
        return 0

    @staticmethod
    def get_number():
        return 16

    @staticmethod
    def get_expression_string():
        return "True"


#


class _BooleanFunctionRandom:  # index: 16, k: 0
    @staticmethod
    def get_boolean(a, b):
        return SystemRandom().choice([True, False])

    @staticmethod
    def get_k():
        return 0

    @staticmethod
    def get_number():
        return 17

    @staticmethod
    def get_expression_string():
        return "SystemRandom().choice([True, False])"


class _BooleanFunctionAlternating:  # index: 17, k: 0
    """for this alternating series of {True, False}, the minimum cycle length is 2."""

    def __init__(self, half_cycle_length: int = 10, initial_boolean: bool = False):
        self.half_cycle_length = half_cycle_length
        self.accumulator = 0
        self.current_boolean = initial_boolean

    def get_boolean(self, a, b):
        self.accumulator += 1
        if self.accumulator % self.half_cycle_length == 0:
            self.current_boolean = not self.current_boolean
        return self.current_boolean

    @staticmethod
    def get_k():
        return 0

    @staticmethod
    def get_number():
        return 18

    def get_expression_string(self):
        return f"Alternating(cycle length: {2 * self.half_cycle_length})"


class _BooleanFunctionSequence:  # index: 18, k: 0
    """this sequence is treated as circular"""

    def __init__(self, sequence: list[bool] = [True]):
        self.sequence = sequence
        self.t = 0

    def get_boolean(self, a, b):
        got_bool = self.sequence[self.t % len(self.sequence)]
        self.t += 1
        return got_bool

    @staticmethod
    def get_k():
        return 0

    @staticmethod
    def get_number():
        return 19

    def get_expression_string(self):
        return f"Sequence of cycle length: {len(self.sequence)})"


class _BooleanFunction_K:  # index: 19, k: 0
    """takes K inputs, output function is the column along a truth table:
    \nT\tT\toutput
    \nT\tF
    \nF\tT
    \nF\tF"""

    def __init__(self, bn, input_indices: list[int], output_function: list[bool] = []):
        if input_indices == []:
            input_indices = bn.input_node_assignments[bn.nodes.index(self)]
        elif output_function == []:
            output_function = list(
                SystemRandom().choices([True, False]), k=len(input_indices)
            )
        else:
            if not len(input_indices) == len(output_function):
                raise ValueError
        self.input_indices = input_indices
        self.bn = bn
        self.state_t = dict()
        truths = [""]
        for _ in range(len(input_indices)):
            temp_truths = []
            for t in truths:
                for tf in ["1", "0"]:
                    temp_truths.append(t + tf)
            truths = temp_truths
        for i in range(len(output_function)):
            self.state_t.update({truths[i]: output_function[i]})

    def get_boolean(self, a, b):
        return self.state_t[
            "".join(
                [
                    str(int(self.bn.current_states_list[-1][i]))
                    for i in self.input_indices
                ]
            )
        ]

    def get_k(self):
        return len(self.input_indices)

    @staticmethod
    def get_number():
        return 19

    def get_expression_string(self):
        return f"Boolean function of {len(self.input_indices)} inputs)"


functions = [
    _BooleanFunction1(),
    _BooleanFunction2(),
    _BooleanFunction3(),
    _BooleanFunction4(),
    _BooleanFunction5(),
    _BooleanFunction6(),
    _BooleanFunction7(),
    _BooleanFunction8(),
    _BooleanFunction9(),
    _BooleanFunction10(),
    _BooleanFunction11(),
    _BooleanFunction12(),
    _BooleanFunction13(),
    _BooleanFunction14(),
    _BooleanFunction15(),
    _BooleanFunction16(),
    _BooleanFunctionRandom(),
    _BooleanFunctionAlternating(),
    _BooleanFunctionSequence(),
    _BooleanFunction_K,
]  # len = 20


# Each BooleanFunctionNode gets 1 of the _BooleanFunction s
class BooleanFunctionNode:
    def __init__(
        self, boolean_expression_index: int
    ):  # sort of an observation of the Factory Design pattern, but only in form, and kind of atomically
        self.function = functions[boolean_expression_index]
        self.k = self.function.get_k()
        self.expression_index = boolean_expression_index

    def get_boolean(self, input_a: bool, input_b: bool):
        return self.function.get_boolean(
            input_a, input_b
        )  # TODO replace with __bool__ method


# Observer Subject design pattern for updating records over additional observations
class Observer(ABC):
    def __init__(self):
        self.subjects = []

    @abstractmethod
    def update(self):
        pass


class Subject:
    def __init__(self):
        self.observers = []

    def attach(self, observer: Observer):
        if self.observers.count(observer) == 0:
            self.observers.append(observer)
            observer.subjects.append(self)

    def detach(self, observer):
        if observer in self.observers:
            self.observers.remove(observer)
        if self in observer.subjects:
            observer.subjects.remove(self)

    def notify(self):
        """notifies all observers (calls observer.update())"""
        for each_observer in self.observers:
            each_observer.update()


#


class CycleRecord:
    def __init__(self, cycle_states_list: list[list[bool]], num_observations: int):
        self.cycle_states_list = cycle_states_list
        self.num_observations = num_observations
        self.zero_distance = None

    def __len__(self):
        return len(self.cycle_states_list) - 1

    def __getitem__(self, index):
        return self.cycle_states_list[index]

    def __lt__(self, other):
        return (
            len(self) < len(other) or self.num_observations < other.num_observations
        )  # < maybe remove this if not used; no need to pick a primary meaning, here

    def set_zero_distance(self, zero_distance: float):
        self.zero_distance = zero_distance

    # def __int__(self) -> int:
    #     self.index_function(self)


class CollapsedCycles(Subject):
    def __init__(self):
        super().__init__()
        self.most_recent_cycle_index = None
        self.cycle_records = []
        self.hashed_states = dict()
        self.current_sort = ""

    def __len__(self):
        return len(self.cycle_records)

    def get_maximum_cycle_length(self):
        return max([len(self.cycle_records[gmcl_i]) for gmcl_i in range(len(self))])

    def add_observation(self, observed_cycle: list[list[bool]]):
        """accepts a cycle observation e.g. A B A, rejected inputs result in
        a self.most_recent_cycle_index of None."""
        # if (len(observed_cycle) == 1) or (observed_cycle[0] != observed_cycle[-1]):
        #     self.most_recent_cycle_index = None
        # else:
        #     cc_ao_index = self.get_index(observed_cycle[0])
        #     if cc_ao_index is None:
        #         cc_ao_index = len(self.cycle_records)
        #         cycle_record = CycleRecord(observed_cycle, 1)
        #         self.cycle_records.append(cycle_record)
        #     else:
        #         cycle_record = self.cycle_records[cc_ao_index]
        #         self.cycle_records[cc_ao_index].num_observations += 1
        #     self.most_recent_cycle_index = cc_ao_index
        #     for each_state_observed in observed_cycle[0:len(observed_cycle):1]:  # state[0] == state[-1]
        #         self.hashed_states.update({''.join([str(int(b)) for b in each_state_observed]):cc_ao_index})
        if (len(observed_cycle) == 1) or (observed_cycle[0] != observed_cycle[-1]):
            self.most_recent_cycle_index = None
        else:
            cc_ao_index = self.get_index(observed_cycle[0])
            if cc_ao_index is None:
                cc_ao_index = len(self.cycle_records)
                self.cycle_records.append(CycleRecord(observed_cycle, 1))
            else:
                self.cycle_records[cc_ao_index].num_observations += 1
            self.most_recent_cycle_index = cc_ao_index
            for each_state_observed in observed_cycle[0 : len(observed_cycle) : 1]:
                self.hashed_states.update(
                    {
                        "".join(
                            [str(int(b)) for b in each_state_observed]
                        ): self.cycle_records[-1]
                    }
                )

    def get_index(self, query_state: list[bool]):  # TODO more hash tables
        #
        # Idea: use a dictionary to get a CycleRecord, return index(CycleRecord)
        #
        # got_index = self.hashed_states.get(''.join([str(int(b)) for b in query_state]), None)
        # if got_index is not None:
        #     got_index = self.cycle_records.index(got_index)
        # return got_index
        # return self.hashed_states.get(''.join([str(int(b)) for b in query_state]), None)

        cc_return_index = None
        cc_gi_bool = False
        cc_gi_accumulator = -1
        if len(self) > 0:
            for each_cycle in self.cycle_records:
                cc_gi_accumulator += 1
                for each_state in each_cycle[0 : len(each_cycle)]:
                    if query_state == each_state:
                        cc_return_index = cc_gi_accumulator
                        cc_gi_bool = True
                        break
                if cc_gi_bool:
                    break
        return cc_return_index

    def sort_cycle_records_by_num_observations(self):
        self.cycle_records.sort(
            key=lambda p: 0 - p.num_observations
        )  # reverse=True keyword is also an option
        self.notify()
        self.current_sort = "num. observations"

    def sort_cycle_records_by_cycle_length(self):
        self.cycle_records.sort(key=lambda p: 0 - len(p))
        self.notify()
        self.current_sort = "cycle length"

    def sort_by_cycle_length_and_num_observations(self):
        self.cycle_records = sorted(self.cycle_records, reverse=True)
        self.notify()
        self.current_sort = "cycle length and num. observations"

    def sort_cycle_records_by_hamming_diameter(self):
        self.cycle_records.sort(
            key=lambda p: 0 - get_hamming_diameter(p.cycle_states_list)
        )
        self.notify()
        self.current_sort = "Hamming diameter"

    def sort_by_cycle_similarity(self, zero_cycle: CycleRecord):
        for each_record in self.cycle_records:
            if isinstance(each_record, CycleRecord):
                each_record.set_zero_distance(
                    per_node_sequence_similarity(
                        zero_cycle.cycle_states_list,
                        each_record.cycle_states_list,
                        None,
                    )
                )
        self.cycle_records.sort(key=lambda p: 0 - p.zero_distance)
        self.notify()
        self.current_sort = "cycle similarity"

    def get_hamm_cons_zero_cycle(self):
        pass

    def get_reference(self, query_list_bool: list[bool]):
        qblb_temp_index = self.get_index(query_list_bool)
        qblb_temp_record_reference = None
        if qblb_temp_index is not None:
            qblb_temp_record_reference = self.cycle_records[qblb_temp_index]
        return qblb_temp_record_reference

    def get_nearest_reference(self, point: list[bool]):
        gnr_nearest_ref = None
        gnr_nearest_hamm = 1e8
        if len(self) > 0:
            for each_cycle_record in self.cycle_records:
                for each_state in each_cycle_record.cycle_states_list[
                    0 : len(each_cycle_record)
                ]:
                    if get_hamming_distance(each_state, point) < gnr_nearest_hamm:
                        gnr_nearest_ref = self.cycle_records[
                            self.cycle_records.index(each_cycle_record)
                        ]
                        gnr_nearest_hamm = get_hamming_distance(each_state, point)
        return gnr_nearest_ref

    def get_furthest_reference(self, point: list[bool]):
        gfr_furthest_ref = None
        gfr_furthest_hamm = -1
        if len(self) > 0:
            for each_cycle_record in self.cycle_records:
                for each_state in each_cycle_record.cycle_states_list[
                    0 : len(each_cycle_record)
                ]:
                    if get_hamming_distance(each_state, point) > gfr_furthest_hamm:
                        gfr_furthest_ref = self.cycle_records[
                            self.cycle_records.index(each_cycle_record)
                        ]
                        gfr_furthest_hamm = get_hamming_distance(each_state, point)
        return gfr_furthest_ref

    def get_consensus_state(self):
        gcs_all_states = []
        for gcs_i in range(len(self)):
            for gcs_j in range(len(self.cycle_records[gcs_i])):
                gcs_all_states.append(
                    self.cycle_records[gcs_i].cycle_states_list[gcs_j]
                )
        return get_consensus_sequence_list(gcs_all_states, [False, True])


class TrajectoryRecord:
    def __init__(
        self, start_index: int | None, end_index: int | None, run_in: list[list[bool]]
    ):
        self.start_index = start_index
        self.end_index = end_index
        self.run_in = copy.deepcopy(run_in)

    def __len__(self):
        return len(self.run_in)


class TrajectoriesManager(Observer):
    """Basically just lists of TrajectoryRecords, for now..."""

    def __init__(self, collapsed_cycles: CollapsedCycles):
        super().__init__()
        self.collapsed_cycles = collapsed_cycles
        collapsed_cycles.attach(
            self
        )  # This step is required by the Observer Subject Design Pattern
        self.u_records = []
        self.t_records = []

    def __len__(self):
        return len(self.u_records) + len(self.t_records)

    def add_record(self, trajectory_record: TrajectoryRecord):
        if trajectory_record.end_index is None:
            self.u_records.append(trajectory_record)
        else:
            self.t_records.append(trajectory_record)

    def update(self):
        """Update end indices of all terminated trajectory records, and update previously unterminated records,
        according to the returned index for the last state in the recorded run-in."""
        for each_t_trajectory_record in self.t_records:
            each_t_trajectory_record.end_index = self.collapsed_cycles.get_index(
                each_t_trajectory_record.run_in[-1]
            )
        for each_u_trajectory_record in self.u_records:
            tm_update_temp_start_index = self.collapsed_cycles.get_index(
                each_u_trajectory_record.run_in[0]
            )
            tm_update_temp_end_index = self.collapsed_cycles.get_index(
                each_u_trajectory_record.run_in[-1]
            )
            if tm_update_temp_end_index is not None:
                self.collapsed_cycles.cycle_records[
                    tm_update_temp_end_index
                ].num_observations += 1
                self.add_record(
                    TrajectoryRecord(
                        tm_update_temp_start_index,
                        tm_update_temp_end_index,
                        each_u_trajectory_record.run_in.copy(),
                    )
                )
                self.u_records.remove(each_u_trajectory_record)


class UnitPerturbationRecord:
    def __init__(
        self,
        reference_state: list[bool],
        start_index: int | None,
        end_index: int | None,
        run_in: list[list[bool]],
        perturbed_node_index: int,
    ):
        self.reference_state = reference_state
        self.start_index = start_index
        self.end_index = end_index
        self.run_in = run_in
        self.perturbed_node_index = perturbed_node_index

    def __str__(self) -> str:
        out_string = f"reference state: {self.reference_state}\n"
        out_string += f"start_index: {self.start_index}\n"
        out_string += f"end_index: {self.end_index}\n"
        out_string += f"perturbed_node_index: {self.perturbed_node_index}\n"
        out_string += "run_in:\n"
        for each in self.run_in:
            out_string += str(each) + "\n"
        return out_string

    def display(self, boolean_network) -> str:
        # Builds rows of text output  ### TODO in a further version, try a single bytearray as representation of each run-in
        test_bn_string_list = [
            str(f'{"node " + str(tbn_k + 1):>12s}: ')
            + str(f"{str(self.reference_state[tbn_k]):^12s}-")
            for tbn_k in range(len(self.run_in[0]))
        ]  # one for each node
        for tbn_h in range(len(self.run_in[0])):
            if self.perturbed_node_index == tbn_h:
                test_bn_string_list[tbn_h] += str(
                    f'{"(" + str(self.run_in[0][tbn_h]) + ")":^12s}'
                )
            else:
                test_bn_string_list[tbn_h] += str(f"{str(self.run_in[0][tbn_h]):^12s}")
        for each_list_1 in self.run_in[1 : len(self.run_in)]:
            for tbn_l in range(len(each_list_1)):
                test_bn_string_list[tbn_l] += str(f"{str(each_list_1[tbn_l]):^12s}")
        print(
            "\ncycle index: "
            + "".join(
                [
                    str(
                        f"{str(boolean_network.bn_collapsed_cycles.get_index(self.reference_state)):^12s}"
                    )
                ]
                + [
                    str(
                        f"{str(boolean_network.bn_collapsed_cycles.get_index(each_state)):^12s}"
                    )
                    for each_state in self.run_in
                ]
            )
            + "\n"
        )
        print("\n".join(test_bn_string_list))


class CompositePerturbation:
    def __init__(self, perturb_node_indices: list[int]):
        self.perturb_node_indices = perturb_node_indices

    def get_perturbed_state(self, in_state: list[bool]):
        gps_state = in_state.copy()
        for each_int in self.perturb_node_indices:
            gps_state[each_int] = not gps_state[each_int]
        return gps_state


class CompositePerturbationRecord:
    def __init__(
        self,
        reference_state: list[bool],
        start_index: int | None,
        end_index: int | None,
        run_in: list[list[bool]],
        perturbed_node_indices: list[int],
    ):
        self.reference_state = reference_state
        self.start_index = start_index
        self.end_index = end_index
        self.run_in = run_in
        self.perturbed_node_indices = perturbed_node_indices


# class RunIn:
#     def __init__(self):
#         self.offset = 0
#         self.data = bytearray()

#     def add_state(self, state):
#         self.data.extend(state)
#         self.offset += len(state)

#     def get_ints(self) -> tuple[int]:
#         for i in range(len(self.data)):
#             b = '0b' + ''.join([str(int(s[i])) for i in range(len(s))])
#             net_states.extend((int(b, base=2) net.bn_collapsed_cycles.get_index(s)))


class BooleanNetwork:
    def __init__(
        self,
        num_nodes,
        possible_functions_indices_list=list(range(16)),
        maximum_inputs_number: int = 2,
        iterations_limit: int = 400,
        initial_sample_size: int = 0,
    ):
        """Constructs a random Boolean network."""
        self.num_nodes = num_nodes
        self.possible_functions_indices_list = possible_functions_indices_list
        self.maximum_inputs_number = maximum_inputs_number
        #
        self.set_of_initial_conditions = set()
        self.total_advance_states = 0
        self.iterations_limit = iterations_limit
        self.current_states_list = []
        self.list_of_all_run_ins = []
        self.list_of_all_indices_of_first_cycle_state = []
        self.nodes = []
        self.node_inputs_assignments = (
            []
        )  # This class controls execution of any behaviors that are f(inputs)
        self.bn_collapsed_cycles = CollapsedCycles()
        self.bn_trajectories = TrajectoriesManager(self.bn_collapsed_cycles)
        #
        self.total_unit_perturbations_transition_matrix = None
        self.cycles_unit_perturbations_transition_matrix = None
        self.t_unit_perturbation_matrix = None
        self.cycles_unit_perturbations_records = None
        self.t_trajectories_unit_perturbations_records = None
        self.u_trajectories_unit_perturbations_records = None
        self.cycle_states_int_set = None
        self.u_trajectories_composite_perturbations_records = None
        self.t_trajectories_composite_perturbations_records = None
        self.cycles_composite_perturbations_matrix = None
        self.cycles_composite_perturbations_records = None
        self.total_composite_perturbations_transition_matrix = None
        #
        self.permute_cycle_tuples = []
        self.permute_total_tuples = []
        self.initial_sample_size = initial_sample_size
        # Network setup
        #
        # setup node input assignments
        for bn_i in range(num_nodes):  # Assumes K is bounded
            init_random_ints_temp = [
                SystemRandom().randrange(0, num_nodes, 1)
                for bn_j in range(maximum_inputs_number)
            ]
            self.node_inputs_assignments.append(
                init_random_ints_temp
            )  # The nodes don't know
        #
        # setup nodes
        for bn_k in range(num_nodes):
            self.nodes.append(
                BooleanFunctionNode(
                    possible_functions_indices_list[
                        SystemRandom().randrange(
                            0, len(possible_functions_indices_list), 1
                        )
                    ]
                )
            )
        #
        # initial sample
        for _ in range(initial_sample_size):
            self.add_cycle()

    def __len__(self):
        return len(self.nodes)

    def __str__(self):
        return (
            "BooleanNetwork: N="
            + str(len(self))
            + ". avg_k= "
            + str(self.get_avg_k())
            + ". Run-ins: "
            + str(len(self.list_of_all_run_ins))
            + ". Observed cycles: "
            + str(len(self.bn_collapsed_cycles))
            + ". t_records: "
            + str(len(self.bn_trajectories.t_records))
            + ". u_records: "
            + str(len(self.bn_trajectories.u_records))
            + "\n"
            + "Cycle lengths: "
            + str(
                [
                    len(self.bn_collapsed_cycles.cycle_records[str_i])
                    for str_i in range(len(self.bn_collapsed_cycles))
                ]
            )
        )  # + bn_output_string_transitions_matrix)

    def add_random_node(self):
        """replaces a random node with a node with function: _BooleanFunctionRandom (function index: 16).
        If all nodes are random, no changes are made."""
        index_list = list(range(len(self)))
        shuffle(index_list)
        for i in index_list:
            if not isinstance(self.nodes[i].function, _BooleanFunctionRandom):
                self.nodes[i] = BooleanFunctionNode(16)
                break

    def new_boolean_functions(self):
        self.node_inputs_assignments = []
        self.nodes = []
        self.reset_network_data()
        # Network setup
        for bn_i in range(self.num_nodes):
            self.node_inputs_assignments.append(
                [SystemRandom().randrange(0, self.num_nodes, 1) for bn_j in range(10)]
            )  # The nodes don't know
        for bn_k in range(self.num_nodes):
            self.nodes.append(
                BooleanFunctionNode(
                    self.possible_functions_indices_list[
                        SystemRandom().randrange(
                            0, len(self.possible_functions_indices_list), 1
                        )
                    ]
                )
            )

    def longest_cycle_length(self):
        return self.bn_collapsed_cycles.get_maximum_cycle_length()

    def reset_network_data(self):
        self.set_of_initial_conditions = set()
        self.total_advance_states = 0
        self.current_states_list = []
        self.list_of_all_run_ins = []
        self.list_of_all_indices_of_first_cycle_state = []
        self.bn_collapsed_cycles = CollapsedCycles()
        self.bn_trajectories = TrajectoriesManager(self.bn_collapsed_cycles)
        self.total_unit_perturbations_transition_matrix = None
        self.cycles_unit_perturbations_records = None
        self.t_trajectories_unit_perturbations_records = None
        self.u_trajectories_unit_perturbations_records = None
        self.cycle_states_int_set = None
        self.u_trajectories_composite_perturbations_records = None
        self.t_trajectories_composite_perturbations_records = None
        self.cycles_composite_perturbations_matrix = None
        self.cycles_composite_perturbations_records = None
        self.total_composite_perturbations_transition_matrix = None

    def get_avg_k(self):
        return sum(
            [
                (
                    each_node.k
                    if self.node_inputs_assignments[gak_i][0]
                    != self.node_inputs_assignments[gak_i][1]
                    else min(each_node.k, 1)
                )
                for gak_i, each_node in zip(range(len(self.nodes)), self.nodes)
            ]
        ) / len(self)

    def sort_cycle_records_by_num_observations(self):
        if len(self.bn_collapsed_cycles) > 1:
            self.bn_collapsed_cycles.sort_cycle_records_by_num_observations()

    def systematic_sample(
        self, initial_condition_list: list[list[bool]] = [[]]
    ) -> None:
        if initial_condition_list == [[]]:
            initial_condition_list = get_all_list_bool(len(self))
        for cond in initial_condition_list:
            self.run_in_from_conditions(cond)

    def run_ins_from_homogeneous_states(self):
        # Run-in's from special case initial conditions of all-False and all-True:
        self.run_in_from_conditions([False for _ in self.nodes])
        self.run_in_from_conditions([True for _ in self.nodes])

    def set_conditions(self, conditions: list[bool]):
        """set_conditions assigns [conditions] to self.current_states_list and is present for readability"""
        # self.current_states_list = [conditions.copy()]
        self.current_states_list = [bytearray(conditions)]

    def advance_state(self):
        """appends to current_state_list a list according to each Boolean function, and the previous state, advancing
        and recording the Boolean states of the conceptual nodes."""
        # self.current_states_list.append([])
        self.current_states_list.append(bytearray())
        for as_i in range(len(self.nodes)):
            self.current_states_list[-1].append(
                self.nodes[as_i].function.get_boolean(
                    bool(
                        self.current_states_list[-2][
                            self.node_inputs_assignments[as_i][0]
                        ]
                    ),
                    bool(
                        self.current_states_list[-2][
                            self.node_inputs_assignments[as_i][1]
                        ]
                    ),
                )
            )
        self.total_advance_states += 1
        # self.current_states_df.apply()

    def get_random_conditions(self):
        """get_random_initial_conditions returns a list of bool chosen by SystemRandom of length len(self.nodes)"""
        return [
            SystemRandom().choice([True, False]) for gric_i in range(len(self.nodes))
        ]

    def run_in_from_random(self, dont_resample_if_true: bool = False):
        """run_in_from_random passes self.get_random_conditions() to self.run_in_from_conditions( ... ).
        dont_resample_if_true is not recommended for sampling > __% state space"""
        rifr_conditions = self.get_random_conditions()
        if len(self.set_of_initial_conditions) < 2 ** len(self):
            if dont_resample_if_true:
                while (
                    "".join([str(rifr_conditions[c_i]) for c_i in range(len(self))])
                    in self.set_of_initial_conditions
                ):
                    rifr_conditions = self.get_random_conditions()
        self.run_in_from_conditions(rifr_conditions)

    def run_in_from_conditions(self, conditions: list[bool]):
        """run_in_from_conditions handles all communication with CollapsedCycles and TrajectoriesManager field
        references. run_in_from_conditions uses self.set_conditions(conditions) and self.advance_state() to generate a
        run-in and executes logic to route run-in results according to cycle detection, self.iterations_limit.
        Post-condition: The contents of the current run-in are added to records of cycles, and/or trajectories.
        """
        self.set_conditions(conditions)
        self.set_of_initial_conditions.add(  # used to facilitate run-in without resample  # todo make plural and have it stop when all states sampled, maybe also systematic sample, but that could be client to this and use run-in from conditions, providing its own, use for Experiment model stuff
            "".join([str(int(conditions[rifr_k])) for rifr_k in range(len(conditions))])
        )
        # detect cycles with a dictionary, for each run-in: run_set <- dict() ### update: (state string, run-in index)
        run_d = dict()
        index = 0  # initial condition has index==0
        # while rifc_bool_cycle_incomplete:
        for _ in range(self.iterations_limit):
            if (
                not (
                    state_str := "".join(
                        [str(int(s)) for s in self.current_states_list[-1]]
                    )
                )
                in run_d
            ):
                run_d.update({state_str: index})  # e.g. '001101':2
                #
                self.advance_state()  #
                #
                index += 1
            else:
                # current state has been observed previously (cycle detected)
                # add cycle observation, and (terminated) trajectory record
                self.bn_collapsed_cycles.add_observation(
                    self.current_states_list[run_d.get(state_str) :]
                )
                self.bn_trajectories.add_record(
                    TrajectoryRecord(
                        self.bn_collapsed_cycles.get_index(self.current_states_list[0]),
                        self.bn_collapsed_cycles.most_recent_cycle_index,
                        copy.deepcopy(self.current_states_list),
                    )
                )
                self.list_of_all_indices_of_first_cycle_state.append(
                    self.bn_collapsed_cycles.most_recent_cycle_index
                )
                break
        else:
            # otherwise, add an unterminated trajectory record
            self.bn_trajectories.add_record(
                TrajectoryRecord(None, None, copy.deepcopy(self.current_states_list))
            )
            self.list_of_all_indices_of_first_cycle_state.append(None)
        self.list_of_all_run_ins.append(copy.deepcopy(self.current_states_list))
        # while rifc_bool_cycle_incomplete:  # brute force solution
        #     self.advance_state()
        #     rifc_j = len(self.current_states_list) - 1 - 1
        #     # Compare most recent to preceding, from second-most recent to initial conditions
        #     while rifc_j > -1:
        #         # Cycle detection
        #         if self.current_states_list[-1] == self.current_states_list[rifc_j]:
        #             rifc_bool_cycle_incomplete = False
        #             # add cycle observation, and (terminated) trajectory record
        #             self.bn_collapsed_cycles.add_observation(
        #                 self.current_states_list[rifc_j:len(self.current_states_list) + 1:1])
        #             self.bn_trajectories.add_record(
        #                 TrajectoryRecord(self.bn_collapsed_cycles.get_index(self.current_states_list[0]),
        #                                  self.bn_collapsed_cycles.most_recent_cycle_index,
        #                                  self.current_states_list[0:rifc_j + 1]))
        #             self.list_of_all_indices_of_first_cycle_state.append(rifc_j)
        #             break
        #         rifc_j -= 1
        #     if len(self.current_states_list) >= self.iterations_limit and rifc_bool_cycle_incomplete:
        #         # otherwise, add an unterminated record
        #         self.bn_trajectories.add_record(TrajectoryRecord(None, None, self.current_states_list))
        #         rifc_bool_cycle_incomplete = False
        #         self.list_of_all_indices_of_first_cycle_state.append(None)
        # self.list_of_all_run_ins.append(self.current_states_list.copy())
        # intermediate optimized solution
        # run_set = set()
        # while rifc_bool_cycle_incomplete:
        #     self.advance_state()
        #     if not (state_str:=''.join([str(int(s)) for s in self.current_states_list[-1]])) in run_set:
        #         run_set.add(state_str)  # e.g. '001101'
        #     else:
        #         rifc_j = len(self.current_states_list) - 1 - 1  # (the one before the most recent)
        #         # Compare most recent to preceding, from second-most recent to initial conditions
        #         while rifc_j > -1:
        #             # Cycle detection
        #             if self.current_states_list[-1] == self.current_states_list[rifc_j]:
        #                 rifc_bool_cycle_incomplete = False
        #                 # add cycle observation, and (terminated) trajectory record
        #                 self.bn_collapsed_cycles.add_observation(
        #                     self.current_states_list[rifc_j:len(self.current_states_list) + 1:1])
        #                 self.bn_trajectories.add_record(
        #                     TrajectoryRecord(self.bn_collapsed_cycles.get_index(self.current_states_list[0]),
        #                                     self.bn_collapsed_cycles.most_recent_cycle_index,
        #                                     self.current_states_list[0:rifc_j + 1]))
        #                 self.list_of_all_indices_of_first_cycle_state.append(rifc_j)
        #                 break
        #             rifc_j -= 1
        #         if len(self.current_states_list) >= self.iterations_limit and rifc_bool_cycle_incomplete:
        #             # otherwise, add an unterminated record
        #             self.bn_trajectories.add_record(TrajectoryRecord(None, None, self.current_states_list))
        #             rifc_bool_cycle_incomplete = False
        #             self.list_of_all_indices_of_first_cycle_state.append(None)
        # self.list_of_all_run_ins.append(self.current_states_list.copy())

    def unrecorded_states_with_noise(
        self,
        number_of_steps: int,
        initial_condition: list[bool],
        unit_perturb_noise: float = 0,
        unit_perturb_bv_noise: float = 0,
    ):
        """Post-condition: len(self.current_states_list) == number_of_steps"""
        uswn_perturbed_nodes = []
        uswn_perturbed_steps = []
        uswn_run_in = []
        self.set_conditions(initial_condition)
        self.advance_state()
        for uswn_i in range(number_of_steps - 1):
            if (
                SystemRandom().random()
                < unit_perturb_bv_noise
                * get_boolean_velocity_list_per_node(
                    [self.current_states_list[-2], self.current_states_list[-1]]
                )
            ):
                # perturbation
                uswn_perturbed_steps.append(uswn_i)
                uswn_perturbed_nodes.append(
                    SystemRandom().randrange(0, len(self.nodes))
                )
                self.current_states_list[-1][uswn_perturbed_nodes[-1]] = (
                    not self.current_states_list[-1][uswn_perturbed_nodes[-1]]
                )
            if SystemRandom().random() < unit_perturb_noise:
                # perturbation
                uswn_perturbed_steps.append(uswn_i)
                uswn_perturbed_nodes.append(
                    SystemRandom().randrange(0, len(self.nodes))
                )
                self.current_states_list[-1][uswn_perturbed_nodes[-1]] = (
                    not self.current_states_list[-1][uswn_perturbed_nodes[-1]]
                )
            # step from perturbed state
            self.advance_state()
        uswn_run_in = copy.deepcopy(self.current_states_list)
        return [uswn_run_in, uswn_perturbed_steps, uswn_perturbed_nodes]

    # def unrecorded_states_with_noise(self, number_of_steps: int, initial_condition: list[bool],
    #                                  prob_unit_perturb: float):
    #     """Post-condition: len(self.current_states_list) == number_of_steps
    #     returns [uswn_run_in, uswn_perturbed_steps, uswn_perturbed_nodes]"""
    #     uswn_perturbed_nodes = []
    #     uswn_perturbed_steps = []
    #     uswn_run_in = []
    #     self.set_conditions(initial_condition)
    #     for uswn_i in range(number_of_steps - 1):
    #         if SystemRandom().random() > prob_unit_perturb:
    #             # perturbation
    #             uswn_perturbed_steps.append(uswn_i)
    #             uswn_perturbed_nodes.append(SystemRandom().randrange(0, len(self.nodes)))
    #             self.current_states_list[-1][uswn_perturbed_nodes[-1]] = not self.current_states_list[-1][
    #                 uswn_perturbed_nodes[-1]]
    #         # step from perturbed state
    #         self.advance_state()
    #     uswn_run_in = copy.deepcopy(self.current_states_list)
    #     return [uswn_run_in, uswn_perturbed_steps, uswn_perturbed_nodes]

    # def unrecorded_states_with_bv_noise(self, number_of_steps: int, initial_condition: list[bool],
    #                                     prob_unit_perturb: float):
    #     """Post-condition: len(self.current_states_list) == number_of_steps"""
    #     uswn_perturbed_nodes = []
    #     uswn_perturbed_steps = []
    #     uswn_run_in = []
    #     self.set_conditions(
    #         initial_condition)
    #     self.advance_state()
    #     for uswn_i in range(
    #             number_of_steps - 1):
    #         if SystemRandom().random() < prob_unit_perturb * get_boolean_velocity_list_per_node(
    #                 [self.current_states_list[-2], self.current_states_list[-1]]):
    #             # perturbation
    #             uswn_perturbed_steps.append(uswn_i)
    #             uswn_perturbed_nodes.append(SystemRandom().randrange(0, len(self.nodes)))
    #             self.current_states_list[-1][uswn_perturbed_nodes[-1]] = not self.current_states_list[-1][
    #                 uswn_perturbed_nodes[-1]]
    #         # step from perturbed state
    #         self.advance_state()
    #     uswn_run_in = copy.deepcopy(self.current_states_list)
    #     return [uswn_run_in, uswn_perturbed_steps, uswn_perturbed_nodes]

    def run_async(self, num_steps: int, update_prob: float) -> None:
        arrays = [
            bytearray([SystemRandom().choice([True, False])]) for _ in self.nodes
        ]  # initial condition
        for _ in range(num_steps):
            update_index_list = list(range(len(arrays)))
            SystemRandom().shuffle(update_index_list)
            for i in update_index_list:  # for each node
                if SystemRandom().random() <= update_prob:
                    a = self.node_inputs_assignments[i][0]  # int
                    b = self.node_inputs_assignments[i][1]  # int
                    arrays[i].append(
                        self.nodes[i].get_boolean(arrays[a][-1], arrays[b][-1])
                    )
                else:
                    arrays[i].append(arrays[i][-1])
            # all arrays in arrays are the same length
        self.current_states_list = [
            [arrays[i][j] for i in range(len(arrays))] for j in range(len(arrays[0]))
        ]

    def add_cycle(self):
        """add_cycle attempts to add a cycle by generating a run-in from random conditions."""
        self.run_in_from_random(dont_resample_if_true=False)

    def add_cycles(self, num_runs: int | None = None):
        """add_cycles generates run-ins from random conditions.
        when num_runs is None, this generates a number of run-ins equal to the constructed initial sample size
        """
        if num_runs is None:
            num_runs = self.initial_sample_size
        for _ in range(num_runs):
            self.run_in_from_random(dont_resample_if_true=False)

    def add_cycle_without_resample(self):
        self.run_in_from_random(dont_resample_if_true=True)

    def get_avg_cycle_length(self):
        if len(self.bn_collapsed_cycles) > 0:
            return sum(
                [
                    len(self.bn_collapsed_cycles.cycle_records[gacl_i])
                    for gacl_i in range(len(self.bn_collapsed_cycles))
                ]
            ) / len(self.bn_collapsed_cycles)

    def _permute_transitions(self) -> None:
        """given a list of tuples: (lambda, lambda, int),
        assigns new values to cycles transitions matrix."""
        for t in self.permute_tuples:  #                   ## call two lambda functions
            self.cycles_unit_perturbations_transition_matrix[t[0]()][t[1]()] = t[2]
        pass

    # unit perturbations and Mersenne primes? (looking at only states 1 edit away from each state of trajectory, cycle states)
    def compute_unit_perturbations_matrix(
        self, sort_selection: int = 1, compute_over_t_u: bool = False
    ):
        """Not row-normalized. For each of the observed states (== reference state):

        set_conditions(the perturbed reference state), advance_state() until a previously observed cycle is detected or
        iterations limit is reached, then, make a UnitPerturbationRecord:

        reference state = an observed state

        start_index = index of reference state (a loop variable)

        end_index = index of last state in current_states_list (returned by CollapsedCycles.get_index(list[bool])

        run_in = current_states_list

        perturbed_node_index = the index of the perturbed node (a loop variable).

        For large computations, all cycle states are sampled, and some of u_records and t_records states are sampled.

        sort_selection:

        0: no sort

        1: cycle length

        2: number of observations

        3: cycle length or number of observations

        4: Hamming "diameter" (maximum intra-cycle Hamming distance)"""
        if sort_selection == 1:
            self.bn_collapsed_cycles.sort_cycle_records_by_cycle_length()
        if sort_selection == 2:
            self.bn_collapsed_cycles.sort_cycle_records_by_num_observations()
        if sort_selection == 3:
            self.bn_collapsed_cycles.sort_by_cycle_length_and_num_observations()
            self.bn_collapsed_cycles.sort_cycle_records_by_cycle_length()
            self.bn_collapsed_cycles.sort_by_cycle_length_and_num_observations()  # TODO not sure why that works ...
        if sort_selection == 4:
            self.bn_collapsed_cycles.sort_cycle_records_by_hamming_diameter()
        self.total_unit_perturbations_transition_matrix = [
            [0 for cupm_k in range(len(self.bn_collapsed_cycles) + 1)]
            for cupm_l in range(len(self.bn_collapsed_cycles) + 1)
        ]
        self.cycles_unit_perturbations_transition_matrix = [
            [0 for cupm_v in range(len(self.bn_collapsed_cycles) + 1)]
            for cupm_u in range(len(self.bn_collapsed_cycles) + 1)
        ]
        self.cycles_unit_perturbations_records = []
        self.t_trajectories_unit_perturbations_records = []
        self.u_trajectories_unit_perturbations_records = []
        if len(self.bn_collapsed_cycles) > 0:
            for c_i in range(len(self.bn_collapsed_cycles)):  # each cycle record
                for each_state_c in self.bn_collapsed_cycles.cycle_records[c_i][
                    0 : len(self.bn_collapsed_cycles.cycle_records[c_i])
                ]:  # each cycle starts / ends with same state
                    #  reference state: each_state_c
                    for c_j in range(len(self)):
                        c_perturbed_state = each_state_c.copy()
                        c_perturbed_state[c_j] = not c_perturbed_state[c_j]
                        # c_perturbed_state[c_j] = not each_state_c[c_j]  # regarding .copy(), this also works  ### see comment in unittest
                        # c_perturbed_state = []
                        # for c_j_2 in range(len(self)):
                        #     if c_j_2 == c_j:
                        #         c_perturbed_state.append(not each_state_c[c_j_2])
                        #     else:
                        #         c_perturbed_state.append(each_state_c[c_j_2])
                        c_bool_cycle_incomplete = True
                        self.set_conditions(c_perturbed_state)  # sets state
                        c_temp_index = self.bn_collapsed_cycles.get_index(
                            self.current_states_list[-1]
                        )
                        if (c_temp_index is not None) or (
                            len(self.current_states_list) > self.iterations_limit
                        ):
                            c_bool_cycle_incomplete = False
                        while c_bool_cycle_incomplete:
                            self.advance_state()  # sets state to next state
                            c_temp_index = self.bn_collapsed_cycles.get_index(
                                self.current_states_list[-1]
                            )
                            if (c_temp_index is not None) or (
                                len(self.current_states_list) > self.iterations_limit
                            ):
                                c_bool_cycle_incomplete = False
                        self.cycles_unit_perturbations_records.append(
                            UnitPerturbationRecord(
                                each_state_c,
                                c_i,
                                c_temp_index,
                                self.current_states_list.copy(),
                                c_j,
                            )
                        )
                        # c_i is the start index, c_temp_index is the end index, c_j is the index of the perturbed node
        #
        # for terminated:= end in cycle state, trajectory records (1)
        # considering states labeled by end_index without multiplicity is consistent with calculation above regarding perturbed cycle states
        if len(self.bn_trajectories.t_records) > 0 and compute_over_t_u:
            t_dict_list = [
                dict() for _ in self.bn_collapsed_cycles.cycle_records
            ]  # index by each_t_trajectory_record.end_index
            for each_t_trajectory_record in self.bn_trajectories.t_records:
                for each_state_t in each_t_trajectory_record.run_in:
                    t_dict_list[each_t_trajectory_record.end_index].update(
                        {
                            "".join([str(int(s)) for s in each_state_t]): (
                                each_t_trajectory_record.end_index,
                                each_state_t.copy(),
                            )
                        }
                    )  # try without copy, also
            # there is 1 value for each unique state (measuring otherwise would include effect of pseudorandom sampling)
            for i in range(len(self.bn_collapsed_cycles.cycle_records)):
                # print(f"t_dict_list[i].values(): {t_dict_list[i].values()}")
                for each_tuple in t_dict_list[i].values():
                    if (
                        self.bn_collapsed_cycles.get_index(each_tuple[1]) is None
                    ):  # excludes cycle states, data for which is present as a copy
                        t_ref_index = each_tuple[0]
                        for t_j in range(len(self)):
                            t_perturbed_state = bytearray()
                            for t_j_2 in range(len(self)):
                                if t_j_2 == t_j:
                                    t_perturbed_state.append(not each_tuple[1][t_j_2])
                                else:
                                    t_perturbed_state.append(each_tuple[1][t_j_2])
                            t_bool_cycle_incomplete = True
                            self.set_conditions(t_perturbed_state)
                            t_temp_index = self.bn_collapsed_cycles.get_index(
                                self.current_states_list[-1]
                            )
                            if (t_temp_index is not None) or (
                                len(self.current_states_list) > self.iterations_limit
                            ):
                                t_bool_cycle_incomplete = False
                            while t_bool_cycle_incomplete:
                                self.advance_state()
                                t_temp_index = self.bn_collapsed_cycles.get_index(
                                    self.current_states_list[-1]
                                )
                                if (t_temp_index is not None) or (
                                    len(self.current_states_list)
                                    > self.iterations_limit
                                ):
                                    t_bool_cycle_incomplete = False
                            self.t_trajectories_unit_perturbations_records.append(
                                UnitPerturbationRecord(
                                    each_state_t,
                                    t_ref_index,
                                    t_temp_index,
                                    self.current_states_list.copy(),
                                    t_j,
                                )
                            )
        #
        # for unterminated:= does not end in cycle state, trajectory records (same~ as (1))
        if len(self.bn_trajectories.u_records) > 0 and compute_over_t_u:
            u_dict = dict()
            for i in range(len(self.bn_trajectories.u_records)):
                u_ref_index = None
                for each_state_u in self.bn_trajectories.u_records[i].run_in:
                    u_dict.update(
                        {
                            "".join(
                                [str(int(s)) for s in each_state_u]
                            ): each_state_u.copy()
                        }
                    )
            # print(f"u_dict:{u_dict}")
            for each_state_u in u_dict.values():
                # TODO Add normalization totals for all ... probably needs graph degree- or something like that (may be relevant )
                # consider: distribution of number of observations of each cycle over trajectory states (like car counters)
                for u_j in range(len(self)):
                    u_perturbed_state = []
                    for u_j_2 in range(len(self)):
                        if u_j_2 == u_j:
                            u_perturbed_state.append(not each_state_u[u_j_2])
                        else:
                            u_perturbed_state.append(each_state_u[u_j_2])
                    u_bool_cycle_incomplete = True
                    self.set_conditions(u_perturbed_state)
                    u_temp_index = self.bn_collapsed_cycles.get_index(
                        self.current_states_list[-1]
                    )
                    if (u_temp_index is not None) or (
                        len(self.current_states_list) > self.iterations_limit
                    ):
                        u_bool_cycle_incomplete = False
                    while u_bool_cycle_incomplete:
                        self.advance_state()
                        u_temp_index = self.bn_collapsed_cycles.get_index(
                            self.current_states_list[-1]
                        )
                        if (u_temp_index is not None) or (
                            len(self.current_states_list) > self.iterations_limit
                        ):
                            u_bool_cycle_incomplete = False
                    self.u_trajectories_unit_perturbations_records.append(
                        UnitPerturbationRecord(
                            each_state_u,
                            u_ref_index,
                            u_temp_index,
                            self.current_states_list.copy(),
                            u_j,
                        )
                    )

        # count them up, if present
        if (
            len(self.cycles_unit_perturbations_records)
            + len(self.u_trajectories_unit_perturbations_records)
            + len(self.t_trajectories_unit_perturbations_records)
            > 0
        ):
            # count cycle transitions
            cupm_none_index = len(self.bn_collapsed_cycles)
            for each_cycle_perturbation in self.cycles_unit_perturbations_records:
                cupm_temp_start_index = each_cycle_perturbation.start_index
                cupm_temp_end_index = each_cycle_perturbation.end_index
                if cupm_temp_start_index is None:
                    cupm_temp_start_index = cupm_none_index
                if cupm_temp_end_index is None:
                    cupm_temp_end_index = cupm_none_index
                #
                self.cycles_unit_perturbations_transition_matrix[cupm_temp_start_index][
                    cupm_temp_end_index
                ] += 1  # ADDS VALUE TO cycles_unit_...
                #
            # total transitions from sampled and resulting states labeled by terminal state index at cycle detection (or iterations limit)
            for cupm_q in range(len(self.cycles_unit_perturbations_transition_matrix)):
                for cupm_r in range(
                    len(self.cycles_unit_perturbations_transition_matrix[0])
                ):
                    self.total_unit_perturbations_transition_matrix[cupm_q][
                        cupm_r
                    ] += self.cycles_unit_perturbations_transition_matrix[cupm_q][
                        cupm_r
                    ]
            for each_perturbation_record in (
                self.u_trajectories_unit_perturbations_records
                + self.t_trajectories_unit_perturbations_records
            ):
                cupm_temp_start_index = each_perturbation_record.start_index
                cupm_temp_end_index = each_perturbation_record.end_index
                if cupm_temp_start_index is None:
                    cupm_temp_start_index = cupm_none_index
                if cupm_temp_end_index is None:
                    cupm_temp_end_index = cupm_none_index
                self.total_unit_perturbations_transition_matrix[cupm_temp_start_index][
                    cupm_temp_end_index
                ] += 1
            # # save values for permutation wrt indices of cycle records (use get_index from str (dict))
            # for i in range(len(self.cycles_unit_perturbations_transition_matrix) - 1):  # last row,
            #     for j in range(len(self.cycles_unit_perturbations_transition_matrix[0]) - 1):  # column have an index not in cycle records
            #         self.permute_cycle_tuples.append((''.join(str(s) for s in self.bn_collapsed_cycles.cycle_records[i].cycle_states_list[0]),
            #                                           ''.join(str(s) for s in self.bn_collapsed_cycles.cycle_records[j].cycle_states_list[0]),
            #                                           self.cycles_unit_perturbations_transition_matrix[i][j]))
            #     else:
            #         self.permute_cycle_tuples.append((''.join(str(s) for s in self.bn_collapsed_cycles.cycle_records[i].cycle_states_list[0]),
            #                                           'None',
            #                                           self.cycles_unit_perturbations_transition_matrix[i][j - 1]))  # None's included: matrix is square, though we could compute longer and resolve more None indexes
            # else:
            #     self.permute_cycle_tuples.append(('None',
            #                                       ''.join(str(s) for s in self.bn_collapsed_cycles.cycle_records[i - 1].cycle_states_list[0]),
            #                                       self.cycles_unit_perturbations_transition_matrix[i][j]))

    def compute_composite_perturbations_matrix(
        self,
        sort_selection: int | None,
        composite_perturbations: list[CompositePerturbation],
        compute_over_u_t_records_if_true: bool,
    ):
        """For each of the *cycle* states (== reference state):  [terminated totals look like cycle totals]

        set_conditions(the perturbed reference state), advance_state() until a previously observed cycle is detected or
        iterations limit is reached, then, make a UnitPerturbationRecord:

        reference state = an observed state

        start_index = index of reference state (a loop variable)

        end_index = index of last state in current_states_list (returned by CollapsedCycles.get_index(list[bool])

        run_in = current_states_list

        perturbed_node_index = the index of the perturbed node (a loop variable).

        For large computations, all cycle states are sampled, and some of u_records and t_records states are sampled.

        sort_selection:

        1: cycle length

        2: number of observations

        3: cycle length or number of observations

        4: Hamming diameter: Maximum Hamming distance of states in argument"""
        if sort_selection is not None:
            if sort_selection == 1:
                self.bn_collapsed_cycles.sort_cycle_records_by_cycle_length()
            elif sort_selection == 2:
                self.bn_collapsed_cycles.sort_cycle_records_by_num_observations()
            elif sort_selection == 3:
                self.bn_collapsed_cycles.sort_by_cycle_length_and_num_observations()
                self.bn_collapsed_cycles.sort_cycle_records_by_cycle_length()
                self.bn_collapsed_cycles.sort_by_cycle_length_and_num_observations()  # TODO not sure why that works ...
            elif sort_selection == 4:
                self.bn_collapsed_cycles.sort_cycle_records_by_hamming_diameter()
            elif sort_selection == 5:
                # combined sort, primarily by cycle length then by Hamming diameter or avg. Bv TODO
                pass
        self.total_composite_perturbations_transition_matrix = [
            [0 for cupm_k in range(len(self.bn_collapsed_cycles) + 1)]
            for cupm_l in range(len(self.bn_collapsed_cycles) + 1)
        ]
        self.cycles_composite_perturbations_matrix = [
            [0 for cupm_v in range(len(self.bn_collapsed_cycles) + 1)]
            for cupm_u in range(len(self.bn_collapsed_cycles) + 1)
        ]
        self.cycles_composite_perturbations_records = []
        self.t_trajectories_composite_perturbations_records = []
        self.u_trajectories_composite_perturbations_records = []
        self.cycle_states_int_set = (
            set()
        )  # Computed to compare hash of states as binary numbers for t, u
        if len(self.bn_collapsed_cycles) > 0:
            for c_i, each_cycle in zip(
                range(len(self.bn_collapsed_cycles)),
                self.bn_collapsed_cycles.cycle_records,
            ):
                for each_state_c in each_cycle[
                    0 : len(each_cycle)
                ]:  # each cycle starts / ends with same state
                    self.cycle_states_int_set.add(bool_state_to_int(each_state_c))
                    #  reference state: each_state_c
                    for each_composite_perturbation in composite_perturbations:
                        c_perturbed_state = (
                            each_composite_perturbation.get_perturbed_state(
                                each_state_c
                            )
                        )
                        c_bool_cycle_incomplete = True
                        self.set_conditions(c_perturbed_state)  # sets state
                        c_temp_index = self.bn_collapsed_cycles.get_index(
                            self.current_states_list[-1]
                        )
                        if (c_temp_index is not None) or (
                            len(self.current_states_list) > self.iterations_limit
                        ):
                            c_bool_cycle_incomplete = False
                        while c_bool_cycle_incomplete:
                            self.advance_state()  # sets state
                            c_temp_index = self.bn_collapsed_cycles.get_index(
                                self.current_states_list[-1]
                            )
                            if (c_temp_index is not None) or (
                                len(self.current_states_list) > self.iterations_limit
                            ):
                                c_bool_cycle_incomplete = False
                        self.cycles_composite_perturbations_records.append(
                            CompositePerturbationRecord(
                                each_state_c,
                                c_i,
                                c_temp_index,
                                self.current_states_list.copy(),
                                each_composite_perturbation.perturb_node_indices,
                            )
                        )
                        # c_i is the start index, c_temp_index is the end index
        #
        #
        if compute_over_u_t_records_if_true:
            if len(self.bn_trajectories.t_records) > 0:
                for each_t_trajectory_record in self.bn_trajectories.t_records:
                    for each_state_t in each_t_trajectory_record.run_in:
                        if self.bn_collapsed_cycles.get_index(each_state_t) is None:
                            t_ref_index = each_t_trajectory_record.end_index
                            for each_composite_perturbation in composite_perturbations:
                                t_perturbed_state = (
                                    each_composite_perturbation.get_perturbed_state(
                                        each_state_t
                                    )
                                )
                                t_bool_cycle_incomplete = True
                                self.set_conditions(t_perturbed_state)
                                t_temp_index = self.bn_collapsed_cycles.get_index(
                                    self.current_states_list[-1]
                                )
                                if (t_temp_index is not None) or (
                                    len(self.current_states_list)
                                    > self.iterations_limit
                                ):
                                    t_bool_cycle_incomplete = False
                                while t_bool_cycle_incomplete:
                                    self.advance_state()
                                    t_temp_index = self.bn_collapsed_cycles.get_index(
                                        self.current_states_list[-1]
                                    )
                                    if (t_temp_index is not None) or (
                                        len(self.current_states_list)
                                        > self.iterations_limit
                                    ):
                                        t_bool_cycle_incomplete = False
                                self.t_trajectories_composite_perturbations_records.append(
                                    CompositePerturbationRecord(
                                        each_state_t,
                                        t_ref_index,
                                        t_temp_index,
                                        self.current_states_list.copy(),
                                        each_composite_perturbation.perturb_node_indices,
                                    )
                                )
            if len(self.bn_trajectories.u_records) > 0:
                for each_u_trajectory_record in self.bn_trajectories.u_records:
                    u_ref_index = None
                    for each_state_u in each_u_trajectory_record.run_in:
                        for each_composite_perturbation in composite_perturbations:
                            u_perturbed_state = (
                                each_composite_perturbation.get_perturbed_state(
                                    each_state_u
                                )
                            )
                            u_bool_cycle_incomplete = True
                            self.set_conditions(u_perturbed_state)
                            u_temp_index = self.bn_collapsed_cycles.get_index(
                                self.current_states_list[-1]
                            )
                            if (u_temp_index is not None) or (
                                len(self.current_states_list) > self.iterations_limit
                            ):
                                u_bool_cycle_incomplete = False
                            while u_bool_cycle_incomplete:
                                self.advance_state()
                                u_temp_index = self.bn_collapsed_cycles.get_index(
                                    self.current_states_list[-1]
                                )
                                if (u_temp_index is not None) or (
                                    len(self.current_states_list)
                                    > self.iterations_limit
                                ):
                                    u_bool_cycle_incomplete = False
                            self.u_trajectories_composite_perturbations_records.append(
                                CompositePerturbationRecord(
                                    each_state_u,
                                    u_ref_index,
                                    u_temp_index,
                                    self.current_states_list.copy(),
                                    each_composite_perturbation.perturb_node_indices,
                                )
                            )
        #
        #
        #
        if (
            len(self.cycles_composite_perturbations_records)
            + len(self.u_trajectories_composite_perturbations_records)
            + len(self.t_trajectories_composite_perturbations_records)
            > 0
        ):
            cupm_none_index = len(self.bn_collapsed_cycles)
            for each_cycle_perturbation in self.cycles_composite_perturbations_records:
                cupm_temp_start_index = each_cycle_perturbation.start_index
                cupm_temp_end_index = each_cycle_perturbation.end_index
                if cupm_temp_start_index is None:
                    cupm_temp_start_index = cupm_none_index
                if cupm_temp_end_index is None:
                    cupm_temp_end_index = cupm_none_index
                self.cycles_composite_perturbations_matrix[cupm_temp_start_index][
                    cupm_temp_end_index
                ] += 1
            if compute_over_u_t_records_if_true:
                for cupm_q in range(len(self.cycles_composite_perturbations_matrix)):
                    for cupm_r in range(
                        len(self.cycles_composite_perturbations_matrix[0])
                    ):
                        self.total_composite_perturbations_transition_matrix[cupm_q][
                            cupm_r
                        ] += self.cycles_composite_perturbations_matrix[cupm_q][cupm_r]
                for each_perturbation_record in (
                    self.u_trajectories_composite_perturbations_records
                    + self.t_trajectories_composite_perturbations_records
                ):
                    cupm_temp_start_index = each_perturbation_record.start_index
                    cupm_temp_end_index = each_perturbation_record.end_index
                    if cupm_temp_start_index is None:
                        cupm_temp_start_index = cupm_none_index
                    if cupm_temp_end_index is None:
                        cupm_temp_end_index = cupm_none_index
                    self.total_composite_perturbations_transition_matrix[
                        cupm_temp_start_index
                    ][cupm_temp_end_index] += 1


def align_cycle_record(cycle_record: CycleRecord, origin_sequence: list[bool]):
    """align_cycle_record is a mutator method that alters the state of the CycleRecord parameter,
    Post-condition: the cycle_states_list of the CycleRecord parameter starts/ends with the cycle state least distant from origin_sequence
    """
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


# Which direction are they going? ...
def align_cycle_states_lists_of_cc_to_cons_seq(collapsed_cycles: CollapsedCycles):
    """align_cycle_states_lists_of_cc_to_cons_seq is a mutator method that alters the state of the parameter.
    Post-condition: all cycle records of reference CollapsedCycles start / end on the cycle state least distant from the consensus sequence of all cycle states
    """
    acslcc_all_states = []
    for acslcc_i in range(len(collapsed_cycles)):
        for acslcc_j in range(len(collapsed_cycles.cycle_records[acslcc_i])):
            acslcc_all_states.append(
                collapsed_cycles.cycle_records[acslcc_i].cycle_states_list[acslcc_j]
            )
    acslcc_cons_seq = get_consensus_sequence_list(acslcc_all_states, [False, True])
    for acslcc_k in range(len(collapsed_cycles)):
        align_cycle_record(collapsed_cycles.cycle_records[acslcc_k], acslcc_cons_seq)
