import unittest
from package import *


# testing boolean_networks.py
# 01-18-2025 TODO Update line numbers (try to automate)  # TODO testing of added Function nodes (random, alternating, sequence)
# in making some revisions at the start of 2025, this unittest was very useful.
# 08-02-2024, 08-03-2024, 08-04-2024, 08-05-2024, 08-06-2024
# Revisions:
# 08-05-2024 -- Revision of get_avg_k adds conditionality such that actual number of edges is counted (such as, when a
# node inputs a, b are a is b, up to 1 is counted (0 if tautology or contradiction))
# 08-05-2024 -- Revision to t_records (not adding trajectory records for trajectories starting in an indexed cycle),
# replacement of total_run_in_s: Integer with a list of all run-ins: some test statements assert wrt length
# 08-06-2024 -- One more look at transitions matrix totals


# BooleanFunctionNode(boolean_expression_index: int)     line 5 through line 218
# 08-02-2024, 08-03-2024 (added assertions about tautology and contradiction (indices 0, 15, respectively))
class TestBooleanNetworks(unittest.TestCase):
    def test_BooleanFunctionNode__BooleanFunction_s(self):
        list_bfn_1 = [boolean_networks.BooleanFunctionNode(i) for i in range(16)]
        list_bfn_2 = [boolean_networks.BooleanFunctionNode(j) for j in range(1, 15, 1)]
        list_input_pairs_list_1 = [
            [False, False],
            [False, True],
            [True, False],
            [True, True],
        ]
        list_bfn_k_1 = [list_bfn_1[k].function.get_k() for k in range(len(list_bfn_1))]
        list_bfn_k_2 = [list_bfn_2[L].function.get_k() for L in range(len(list_bfn_2))]
        for bfn, i in zip(list_bfn_1, range(1, 16 + 1)):
            self.assertEqual(bfn.function.get_number(), i)
        for m, each_bfn in zip(range(len(list_bfn_1)), list_bfn_1):
            global from_expression, a, b
            from_expression = None
            expression = (
                "from_expression = " + each_bfn.function.get_expression_string()
            )
            for each_input_pair in list_input_pairs_list_1:
                self.assertTrue(
                    isinstance(
                        list_bfn_1[m].function.get_boolean(
                            each_input_pair[0], each_input_pair[1]
                        ),
                        bool,
                    )
                )
                a = each_input_pair[0]
                b = each_input_pair[1]
                exec(expression, globals())
                self.assertEqual(
                    from_expression,
                    list_bfn_1[m].get_boolean(each_input_pair[0], each_input_pair[1]),
                )
        for each_input_pair in list_input_pairs_list_1:
            self.assertFalse(
                list_bfn_1[0].function.get_boolean(
                    each_input_pair[0], each_input_pair[1]
                )
            )
            self.assertTrue(
                list_bfn_1[15].function.get_boolean(
                    each_input_pair[0], each_input_pair[1]
                )
            )
        for each_element_1 in list_bfn_k_1:
            self.assertTrue(isinstance(each_element_1, int))
            self.assertTrue(each_element_1 >= 0)
            self.assertTrue(each_element_1 <= 2)
        for each_element_2 in list_bfn_k_2:
            self.assertTrue(isinstance(each_element_2, int))
            self.assertTrue(each_element_2 >= 1)
            self.assertTrue(each_element_2 <= 2)


# class CycleRecord:     line 247 through line 268
# def __init__(self, cycle_states_list: list[list[bool]], num_observations: int):
# 08-02-2024
class TestCycleRecord(unittest.TestCase):
    def test_all_methods_of_CycleRecord(self):
        list_list_bool_1 = (
            [[False for i in range(100)] for L in range(100)]
            + [[True for j in range(100)] for m in range(200)]
            + [[False for k in range(100)] for n in range(100)]
        )
        list_list_bool_2 = [[False], [False]]
        list_list_bool_3 = [[False], [False], [True]]
        test_cycle_record_1 = boolean_networks.CycleRecord(list_list_bool_1, 0)
        test_cycle_record_2 = boolean_networks.CycleRecord(list_list_bool_2, 1)
        test_cycle_record_3 = boolean_networks.CycleRecord(list_list_bool_3, 10000)
        self.assertEqual(
            test_cycle_record_1.cycle_states_list[0],
            test_cycle_record_1.cycle_states_list[-1],
        )
        self.assertNotEqual(
            test_cycle_record_1.cycle_states_list[0],
            test_cycle_record_1.cycle_states_list[200],
        )
        self.assertEqual(test_cycle_record_1[0], test_cycle_record_1[-1])
        self.assertNotEqual(test_cycle_record_1[0], test_cycle_record_1[200])
        self.assertEqual(len(test_cycle_record_3) - len(test_cycle_record_2), 1)
        self.assertEqual(len(test_cycle_record_1), 399)


# class CollapsedCycles(Subject):    263 through 317      (229 through 244)  [not covering notify(self) 242-244]
# def __init__(self):
# 08-02-2024, 08-03-2024, 08-06-2024
# Revision 08-06-2024: implementation of a tree to decrease index lookup times is such that direct assignment of cycle
# records to a CollapsedCycles without direct assignment of entries to its tree results in an index of None; further, a
# binary search tree of natural ordering of magnitude of integers equal to the state as a binary number requires the
# length be the same or known
class TestCollapsedCyclesAndSubject(unittest.TestCase):
    def test_Subject(self):
        cc_as_subject_1 = boolean_networks.CollapsedCycles()
        self.assertTrue(isinstance(cc_as_subject_1.observers, list))
        list_of_observers = []
        for i in range(100):
            cc_as_subject_1.attach(
                boolean_networks.TrajectoriesManager(cc_as_subject_1)
            )
            list_of_observers.append(cc_as_subject_1.observers[-1])
        for j in range(200):
            cc_as_subject_1.detach(list_of_observers[j % len(list_of_observers)])

    # Initial results were Errors, added conditions controlling execution of statements in attach and detach

    def test_CollapsedCycles_N3_K2_Ground_Truth_Known(self):
        cc_1 = boolean_networks.CollapsedCycles()  # To remain as constructed
        cc_2 = (
            boolean_networks.CollapsedCycles()
        )  # To accumulate observations of cycles
        cc_3 = (
            boolean_networks.CollapsedCycles()
        )  # The same as cc_2 except: add 10x duplicates for each observation
        cc_4 = (
            boolean_networks.CollapsedCycles()
        )  # The same as cc_2 except: add observations that fail [0] == [-1],
        # and add observations that fail len(observed_cycle)!=1
        cc_5 = (
            boolean_networks.CollapsedCycles()
        )  # The same as cc_1 except: add observations that fail [0] == [-1],
        # and add observations that fail len(observed_cycle)!=1
        cc_6 = (
            boolean_networks.CollapsedCycles()
        )  # The same as cc_1 except: add one observation of observed_cycle_8
        cc_7 = (
            boolean_networks.CollapsedCycles()
        )  # The same as cc_2 except: add one observation of observed_cycle_8
        # Consider: run-in's of len(list(list(bool)))==1 pass [0]==[-1] ... Now: len==1 results in no added observation
        #
        # Starting with hard-coded data for an N=3 K=2 network of Boolean functions:
        # node: A             # node: B             # node: C
        # B   C   A           # A   C   B           # A   B   C
        # 0   0   1           # 0   0   1           # 0   0   1
        # 0   1   1           # 0   1   1           # 0   1   0
        # 1   0   0           # 1   0   0           # 1   0   1
        # 1   1   0           # 1   1   0           # 1   1   0
        #
        # There are 2^3 possible states of [A,B,C]: [0,0,0] [0,0,1] [0,1,0] [0,1,1] [1,0,0] [1,0,1] [1,1,0] [1,1,1]
        #
        # initial condition: [0,0,0]
        # t=0
        # A: 0    B: 0    C: 0
        # t=1
        # A: 1    B: 1    C: 1
        # t=2
        # A: 0    B: 0    C: 0    # Cycle length: 2    [ ### cycle index: 0 ]    [ num observations: 1 ]
        run_in_1 = [[False, False, False], [True, True, True], [False, False, False]]
        observed_cycle_1 = run_in_1  # Emulating logic in class BooleanNetwork
        #
        # initial condition: [0,0,1]
        # t=0
        # A: 0    B: 0    C: 1
        # t=1
        # A: 1    B: 1    C: 1
        # t=2
        # A: 0    B: 0    C: 0
        # t=3
        # A: 1    B: 1    C: 1    # Cycle length: 2    [ ### cycle index: 0 ]    [ num observations: 2 ]
        run_in_2 = [
            [False, False, True],
            [True, True, True],
            [False, False, False],
            [True, True, True],
        ]
        observed_cycle_2 = run_in_2[1:4:1]  # Emulating logic in class BooleanNetwork
        #
        # initial condition: [0,1,0]
        # t=0
        # A: 0    B: 1    C: 0
        # t=1
        # A: 0    B: 1    C: 0    # Cycle length: 1 (eq. state)    [ ### cycle index: 1 ]    [ num observations: 1 ]
        run_in_3 = [[False, True, False], [False, True, False]]
        observed_cycle_3 = run_in_3  # Emulating logic in class BooleanNetwork
        #
        # initial condition: [0,1,1]
        # t=0
        # A: 0    B: 1    C: 1
        # t=1
        # A: 0    B: 1    C: 0
        # t=2
        # A: 0    B: 1    C: 0    # Cycle length: 1 (eq. state)    [ ### cycle index: 1 ]    [ num observations: 2 ]
        run_in_4 = [[False, True, True], [False, True, False], [False, True, False]]
        observed_cycle_4 = run_in_4[1:3:1]  # Emulating logic in class BooleanNetwork
        #
        # initial condition: [1,0,0]
        # t=0
        # A: 1    B: 0    C: 0
        # t=1
        # A: 1    B: 0    C: 1
        # t=2
        # A: 1    B: 0    C: 1    # Cycle length: 1 (eq. state)    [ ### cycle index: 2 ]    [ num observations: 1 ]
        run_in_5 = [[True, False, False], [True, False, True], [True, False, True]]
        observed_cycle_5 = run_in_5[1:3:1]  # Emulating logic in class BooleanNetwork
        #
        # initial condition: [1,0,1]
        # t=0
        # A: 1    B: 0    C: 1
        # t=1
        # A: 1    B: 0    C: 1    # Cycle length: 1 (eq. state)    [ ### cycle index: 2 ]    [ num observations: 2 ]
        run_in_6 = [[True, False, True], [True, False, True]]
        observed_cycle_6 = run_in_6  # Emulating logic in class BooleanNetwork
        #
        # initial condition: [1,1,0]
        # t=0
        # A: 1    B: 1    C: 0
        # t=1
        # A: 0    B: 0    C: 0
        # t=2
        # A: 1    B: 1    C: 1
        # t=3
        # A: 0    B: 0    C: 0    # Cycle length: 2    [ ### cycle index: 0 ]    [ num observations: 3 ]
        run_in_7 = [
            [True, True, False],
            [False, False, False],
            [True, True, True],
            [False, False, False],
        ]
        observed_cycle_7 = run_in_7[1:4:1]  # Emulating logic in class BooleanNetwork
        #
        # initial condition: [1,1,1]
        # t=0
        # A: 1    B: 1    C: 1
        # t=1
        # A: 0    B: 0    C: 0
        # t=2
        # A: 1    B: 1    C: 1    # Cycle length: 2    [ ### cycle index: 0 ]    [ num observations: 4 ]
        run_in_8 = [[True, True, True], [False, False, False], [True, True, True]]
        observed_cycle_8 = run_in_8  # Emulating logic in class BooleanNetwork
        list_of_observed_cycles = [
            observed_cycle_1,
            observed_cycle_2,
            observed_cycle_3,
            observed_cycle_4,
            observed_cycle_5,
            observed_cycle_6,
            observed_cycle_7,
            observed_cycle_8,
        ]
        obs_fail_first_last = [
            [True, True, True],
            [False, False, False],
            [True, True, False],
        ]
        obs_fail_length_1 = [[True, True, True]]
        obs_fail_length_2 = [[]]  # Fails suggested typing but code runs
        # adds all observed cycles to each CollapsedCycles, attempts to add duplicates, and observations that are not cycles
        # cc_1 does not get any observations
        for each_obs_1 in list_of_observed_cycles:
            cc_2.add_observation(each_obs_1)
            cc_3.add_observation(each_obs_1)
            cc_4.add_observation(each_obs_1)
            cc_7.add_observation(each_obs_1)
        self.assertEqual(cc_7.cycle_records[0].num_observations, 4)
        for _ in range(10):
            for each_obs_2 in list_of_observed_cycles:
                cc_3.add_observation(each_obs_2)
        for _ in range(10):
            cc_1.add_observation(obs_fail_first_last)
            cc_4.add_observation(obs_fail_first_last)
            cc_5.add_observation(obs_fail_first_last)
        for _ in range(10):
            cc_1.add_observation(obs_fail_length_1)
            cc_4.add_observation(obs_fail_length_1)
            cc_5.add_observation(obs_fail_length_1)
        for _ in range(10):
            cc_1.add_observation(obs_fail_length_2)
            cc_4.add_observation(obs_fail_length_2)
            cc_5.add_observation(obs_fail_length_2)
        cc_6.add_observation(observed_cycle_8)
        cc_7.add_observation(observed_cycle_8)
        # assertions regarding number of observed cycles
        self.assertEqual(len(cc_1), 0)
        self.assertEqual(len(cc_2), 3)
        self.assertEqual(len(cc_3), 3)
        self.assertEqual(len(cc_4), 3)
        self.assertEqual(len(cc_7), 3)
        # test sorting by number of observations
        for _ in range(10):
            cc_7.add_observation(list_of_observed_cycles[-1])
        cc_7.sort_cycle_records_by_num_observations()
        self.assertEqual(cc_7.cycle_records[0].cycle_states_list, observed_cycle_1)
        # TODO fix problem with eg T F T not equal to F T F (same cycle?)
        for i in range(len(cc_7) - 1):
            # a >= b
            self.assertGreaterEqual(
                cc_7.cycle_records[i].num_observations,
                cc_7.cycle_records[i + 1].num_observations,
            )

    def test_CollapsedCycles_Test_Data(self):
        cc_1 = boolean_networks.CollapsedCycles()
        cc_2 = boolean_networks.CollapsedCycles()
        cc_3 = boolean_networks.CollapsedCycles()
        cc_4 = boolean_networks.CollapsedCycles()
        cc_5 = boolean_networks.CollapsedCycles()
        cycle_1 = [
            [False],
            [False],
        ]  # For example, a cycle of a net of N=1 of contradiction, or _BooleanFunction5
        cycle_2 = [
            [False],
            [True],
            [False],
        ]  # Another feasible cycle of a net of N=1, e.g. _BooleanFunction9
        cycle_3 = [
            [False],
            [False],
            [False],
        ]  # cycles 1-3 are equivalent to conditions for adding an observation
        cycle_4 = [
            [True]
        ]  # cycle_4 is not equivalent to cycles 1-3 to the same, but fails a condition
        list_list_bool_1 = (
            [[False for i in range(100)] for L in range(100)]
            + [[True for j in range(100)] for m in range(200)]
            + [[False for k in range(100)] for n in range(100)]
        )
        cycle_5 = (
            list_list_bool_1 + list_list_bool_1
        )  # cycle_5 is not equivalent to cycles 1-4 to the same
        cycle_6_fail_length = [
            list_list_bool_1[0]
        ]  # cycle_6 is not equivalent to cycles 1-5, but fails a condition
        #
        for k in range(100):
            cc_2.add_observation(cycle_1)
            cc_5.add_observation(cycle_1)  #
            cc_3.add_observation(cycle_2)
            cc_5.add_observation(cycle_2)  #
            if k > 49:
                cc_2.add_observation(cycle_3)
                cc_2.add_observation(cycle_4)
                cc_5.add_observation(cycle_3)  #
                cc_5.add_observation(cycle_4)  #
                cc_3.add_observation(cycle_3)
                cc_3.add_observation(cycle_4)
                cc_5.add_observation(cycle_3)  #
                cc_5.add_observation(cycle_4)  #
        for m in range(100):
            cc_4.add_observation(cycle_5)
            cc_5.add_observation(cycle_5)  #
        cc_4.add_observation(cycle_6_fail_length)
        #
        self.assertEqual(len(cc_1), 0)
        self.assertEqual(len(cc_2), 1)
        self.assertEqual(len(cc_3), 1)
        self.assertEqual(len(cc_4), 1)
        self.assertEqual(len(cc_5), 2)  #
        self.assertEqual(cc_5.most_recent_cycle_index, 1)
        self.assertEqual(cc_5.most_recent_cycle_index, 1)
        self.assertIsNone(cc_4.most_recent_cycle_index)
        # TODO add assertions for numbers of observations, any other fields

    def test_CollapsedCycles_get_index(self):
        cycle_1 = [[False], [False]]
        cycle_2 = [[True], [True]]
        cycle_3 = [[False, False], [True, True], [False, False]]
        cycle_state_1 = [True, True, False]
        cc_1 = boolean_networks.CollapsedCycles()
        cc_2 = boolean_networks.CollapsedCycles()
        cc_1.cycle_records = [
            boolean_networks.CycleRecord(cycle_1, 100),
            boolean_networks.CycleRecord(cycle_2, 10),
            boolean_networks.CycleRecord(cycle_3, 1),
        ]
        # for gi_i in range(len(cc_1.cycle_records)):
        #     for gi_j in range(len(cc_1.cycle_records[gi_i].cycle_states_list) - 1):
        #         cc_1.tree.add(cc_1.cycle_records[gi_i].cycle_states_list[gi_j], cc_1.cycle_records[gi_i])
        self.assertEqual(cc_1.get_index(cycle_1[0]), 0)
        self.assertEqual(cc_1.get_index(cycle_1[1]), 0)
        self.assertIsNone(cc_2.get_index(cycle_1[0]))
        self.assertIsNone(cc_1.get_index(cycle_state_1))
        self.assertIsNone(cc_2.get_index(cycle_state_1))
        self.assertIsNone(cc_2.get_index(cycle_state_1))

    def test_CollapsedCycles__get_index__get_reference(self):
        # sort cycles by num observations also calls update() of Subject, designed to prompt a TrajectoriesManager to
        # update its cycle indices for trajectory records terminating in a cycle, this behavior is tested below in class
        # TestTrajectoriesManagerObserver.test_as_Observer
        cycle_1 = [[False], [False]]
        cycle_2 = [[True], [True]]
        query_state_1 = [False]
        query_state_2 = [True]
        cc_1 = boolean_networks.CollapsedCycles()
        cc_2 = boolean_networks.CollapsedCycles()
        cycle_record_2 = boolean_networks.CycleRecord(cycle_2, 10)
        cc_1.cycle_records = [
            boolean_networks.CycleRecord(cycle_1, 100),
            cycle_record_2,
        ]
        # for each_record_1 in cc_1.cycle_records:
        #     for each_state_1 in each_record_1.cycle_states_list[0:len(each_record_1.cycle_states_list) - 1:1]:
        #         cc_1.tree.add(each_state_1, each_record_1)
        cc_2.cycle_records = [boolean_networks.CycleRecord(cycle_1, 1), cycle_record_2]
        # for each_record_2 in cc_2.cycle_records:
        #     for each_state_2 in each_record_2.cycle_states_list[0:len(each_record_2.cycle_states_list) - 1:1]:
        #         cc_2.tree.add(each_state_2, each_record_2)
        cc_2.sort_cycle_records_by_num_observations()
        self.assertEqual(cc_1.get_index(query_state_1), 0)
        self.assertEqual(cc_1.get_index(query_state_2), 1)
        self.assertEqual(cc_2.get_index(query_state_1), 1)
        self.assertEqual(cc_2.get_index(query_state_2), 0)
        self.assertIs(
            cc_1.get_reference(query_state_2), cc_2.get_reference(query_state_2)
        )
        self.assertIsNot(
            cc_1.get_reference(query_state_1), cc_2.get_reference(query_state_1)
        )


# class TrajectoryRecord    line 320 through line 324
# 08-03-2024
class TestTrajectoryRecord(unittest.TestCase):
    def test___init__(self):
        trajectory_record_1 = boolean_networks.TrajectoryRecord(
            None, None, [[False], [True], [False]]
        )
        trajectory_record_2 = boolean_networks.TrajectoryRecord(-100, -100, [[True]])
        self.assertIsNone(trajectory_record_1.start_index)
        self.assertIsNone(trajectory_record_1.end_index)
        self.assertEqual(trajectory_record_2.start_index, -100)


# class TrajectoriesManager(Observer)    line 327 through line 344      (line 221 through 226)
# def __init__(self, collapsed_cycles: CollapsedCycles):
# 08-03-2024, 08-06-2024
class TestTrajectoriesManagerObserver(unittest.TestCase):
    def test_all(self):
        cc_1 = boolean_networks.CollapsedCycles()
        tm_as_observer_1 = boolean_networks.TrajectoriesManager(cc_1)
        self.assertIsInstance(tm_as_observer_1.subjects, list)
        cc_1.add_observation([[False, True], [False, True]])
        cc_1.cycle_records[-1].num_observations = 100
        cc_1.add_observation([[True, False], [True, False]])
        cc_1.cycle_records[-1].num_observations = 1000
        tm_as_observer_1.add_record(
            boolean_networks.TrajectoryRecord(
                None, 1, [[False, False], [True, True], [True, False], [True, False]]
            )
        )
        tm_as_observer_1.add_record(
            boolean_networks.TrajectoryRecord(
                None, 0, [[False, False], [True, True], [False, True], [False, True]]
            )
        )
        tm_as_observer_1.add_record(
            boolean_networks.TrajectoryRecord(
                None, None, [[False, False], [True, True]]
            )
        )
        tm_as_observer_1.add_record(
            boolean_networks.TrajectoryRecord(
                None, None, [[False, False], [False, False], [True, False]]
            )
        )
        self.assertEqual(len(tm_as_observer_1.u_records), 2)
        self.assertEqual(len(tm_as_observer_1.t_records), 2)
        self.assertIsNone(tm_as_observer_1.u_records[0].end_index)
        self.assertIsNone(tm_as_observer_1.u_records[1].end_index)
        self.assertEqual(tm_as_observer_1.t_records[0].end_index, 1)
        self.assertEqual(tm_as_observer_1.t_records[1].end_index, 0)
        cc_1.sort_cycle_records_by_num_observations()
        self.assertEqual(len(tm_as_observer_1.u_records), 1)
        self.assertEqual(len(tm_as_observer_1.t_records), 3)
        self.assertIsNone(tm_as_observer_1.u_records[0].end_index)
        self.assertEqual(tm_as_observer_1.t_records[0].end_index, 0)
        self.assertEqual(tm_as_observer_1.t_records[1].end_index, 1)
        self.assertEqual(tm_as_observer_1.t_records[2].end_index, 0)


# class UnitPerturbationRecord:    line 350 through line 357
# 08-03-2024
class TestUnitPerturbationRecord(unittest.TestCase):
    def test_record(self):
        upr_1 = boolean_networks.UnitPerturbationRecord(
            [True], 0, 0, [[False], [True], [True]], 0
        )
        self.assertIsInstance(upr_1, boolean_networks.UnitPerturbationRecord)


# class BooleanNetwork    line 364 through line 579
# def __init__(self, num_nodes, possible_functions_indices_list, maximum_inputs_number, iterations_limit):
# 08-04-2024, 08-05-2024, 08-06-2024
class TestBooleanNetwork(unittest.TestCase):
    def test_construction__and____len__(self):
        # bn_1 = boolean_networks.BooleanNetwork(1, [0], 0,
        #                                        100)
        bn_2 = boolean_networks.BooleanNetwork(20, [i for i in range(16)], 2, 400)
        bn_3 = boolean_networks.BooleanNetwork(200, [0], 100, 400)
        #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   # ##
        bn_4 = boolean_networks.BooleanNetwork(10000, [j for j in range(16)], 100, 400)
        # not indicative of potential use of BooleanNetwork, but, construction of this should be without any problems ##
        # self.assertEqual(bn_1.total_advance_states, 0)
        # self.assertEqual(len(bn_1.list_of_all_run_ins), 0)
        self.assertEqual(bn_2.total_advance_states, 0)
        self.assertEqual(len(bn_2.list_of_all_run_ins), 0)
        self.assertEqual(bn_3.total_advance_states, 0)
        self.assertEqual(len(bn_3.list_of_all_run_ins), 0)
        self.assertEqual(bn_4.total_advance_states, 0)
        self.assertEqual(len(bn_4.list_of_all_run_ins), 0)
        # self.assertEqual(bn_1.iterations_limit, 100)
        self.assertEqual(bn_2.iterations_limit, 400)
        self.assertEqual(bn_3.iterations_limit, 400)
        self.assertEqual(bn_4.iterations_limit, 400)
        # self.assertEqual(bn_1.current_states_list, [])
        self.assertEqual(bn_2.current_states_list, [])
        self.assertEqual(bn_3.current_states_list, [])
        self.assertEqual(bn_4.current_states_list, [])
        # self.assertIsNone(bn_1.cycles_unit_perturbations_transition_matrix)
        # self.assertIsNone(bn_1.cycles_unit_perturbations_records)
        # self.assertIsNone(bn_1.t_trajectories_unit_perturbations_records)
        # self.assertIsNone(bn_1.u_trajectories_unit_perturbations_records)
        self.assertIsNone(bn_2.cycles_unit_perturbations_transition_matrix)
        self.assertIsNone(bn_2.cycles_unit_perturbations_records)
        self.assertIsNone(bn_2.t_trajectories_unit_perturbations_records)
        self.assertIsNone(bn_2.u_trajectories_unit_perturbations_records)
        self.assertIsNone(bn_3.cycles_unit_perturbations_transition_matrix)
        self.assertIsNone(bn_3.cycles_unit_perturbations_records)
        self.assertIsNone(bn_3.t_trajectories_unit_perturbations_records)
        self.assertIsNone(bn_3.u_trajectories_unit_perturbations_records)
        self.assertIsNone(bn_4.cycles_unit_perturbations_transition_matrix)
        self.assertIsNone(bn_4.cycles_unit_perturbations_records)
        self.assertIsNone(bn_4.t_trajectories_unit_perturbations_records)
        self.assertIsNone(bn_4.u_trajectories_unit_perturbations_records)
        # self.assertEqual(len(bn_1.nodes), 1)
        self.assertEqual(len(bn_2.nodes), 20)
        self.assertEqual(len(bn_3.nodes), 200)
        self.assertEqual(len(bn_4.nodes), 10000)
        self.assertEqual(len(bn_2), 20)
        self.assertEqual(len(bn_3), 200)
        # self.assertEqual(bn_1.node_inputs_assignments, [None])
        self.assertTrue(len(bn_2.node_inputs_assignments) == len(bn_2))
        self.assertTrue(len(bn_3.node_inputs_assignments) == len(bn_3))
        self.assertTrue(len(bn_4.node_inputs_assignments) == len(bn_4))
        self.assertEqual(len(bn_2.node_inputs_assignments[0]), 2)
        self.assertEqual(len(bn_2.node_inputs_assignments[-1]), 2)
        self.assertEqual(len(bn_3.node_inputs_assignments[0]), 100)
        self.assertEqual(len(bn_3.node_inputs_assignments[-1]), 100)
        self.assertEqual(len(bn_4.node_inputs_assignments[0]), 100)
        self.assertEqual(len(bn_4.node_inputs_assignments[-1]), 100)
        # self.assertIsInstance(bn_1.nodes[0], boolean_networks.BooleanFunctionNode)
        # self.assertIsInstance(bn_1.nodes[0].function, boolean_networks._BooleanFunction1)
        self.assertIsInstance(bn_2.nodes[0], boolean_networks.BooleanFunctionNode)
        self.assertIsInstance(bn_2.nodes[-1], boolean_networks.BooleanFunctionNode)
        self.assertIsInstance(bn_3.nodes[0], boolean_networks.BooleanFunctionNode)
        self.assertIsInstance(
            bn_3.nodes[0].function, boolean_networks._BooleanFunction1
        )
        self.assertIsInstance(bn_3.nodes[-1], boolean_networks.BooleanFunctionNode)
        self.assertIsInstance(
            bn_3.nodes[-1].function, boolean_networks._BooleanFunction1
        )
        self.assertIsInstance(bn_4.nodes[0], boolean_networks.BooleanFunctionNode)
        self.assertIsInstance(bn_4.nodes[-1], boolean_networks.BooleanFunctionNode)

    def test_get_avg_k(self):
        # function index [0, 1, 2, 3, 4, 5, 6 ,7, 8, 9, 10, 11, 12, 13, 14, 15]
        # get_k return   [0, 2, 2, 1, 2, 1, 2, 2, 2, 2,  1,  2,  1,  2,  2,  0]
        # TODO get_k list (sequence) is palindromic; inputs a, b are randomly assigned, can't we use half of this list?
        bn_1 = boolean_networks.BooleanNetwork(200, [0, 15], 2, 200)
        bn_2 = boolean_networks.BooleanNetwork(200, [3, 5, 10, 12], 2, 200)
        bn_3_1 = boolean_networks.BooleanNetwork(
            200, [1, 2, 4, 6, 7, 8, 9, 11, 13, 14], 2, 200
        )
        bn_3_2 = boolean_networks.BooleanNetwork(
            200, [1, 2, 4, 6, 7, 8, 9, 11, 13, 14], 2, 200
        )
        for tgak_i, each_integer_pair in zip(
            range(len(bn_3_2.node_inputs_assignments)), bn_3_2.node_inputs_assignments
        ):
            if each_integer_pair[0] == each_integer_pair[1]:
                # This line doesn't always run
                bn_3_2.node_inputs_assignments[tgak_i][0] = (
                    bn_3_2.node_inputs_assignments[tgak_i][0] + 1
                ) % len(bn_3_2)
        bn_3_3 = boolean_networks.BooleanNetwork(
            200, [1, 2, 4, 6, 7, 8, 9, 11, 13, 14], 2, 200
        )
        for tgak_j in range(len(bn_3_3.node_inputs_assignments)):
            bn_3_3.node_inputs_assignments[tgak_j][1] = bn_3_3.node_inputs_assignments[
                tgak_j
            ][0]
        bn_4 = boolean_networks.BooleanNetwork(
            200, [0, 3], 2, 200
        )  # 0 <= avg num inputs <= 1
        bn_5 = boolean_networks.BooleanNetwork(
            200, [3, 1], 2, 200
        )  # 1 <= avg num inputs <= 2
        self.assertEqual(bn_1.get_avg_k(), 0)
        self.assertEqual(bn_2.get_avg_k(), 1)
        # Initial results: a number != 1, error was in class BooleanFunctionNode, a repeat of _BooleanFunction14, and
        # omission of _BooleanFunction7. In retrospect, keeping track of these would have been easier if named by index.
        self.assertTrue(bn_3_1.get_avg_k() <= 2)
        self.assertEqual(bn_3_2.get_avg_k(), 2)
        self.assertEqual(bn_3_3.get_avg_k(), 1)
        self.assertTrue(bn_4.get_avg_k() >= 0)
        self.assertTrue(bn_4.get_avg_k() <= 1)
        self.assertTrue(bn_5.get_avg_k() >= 1)
        self.assertTrue(bn_5.get_avg_k() <= 2)

    def test_set_conditions(self):
        bn_1 = boolean_networks.BooleanNetwork(1, [1], 1, 1)
        bn_2 = boolean_networks.BooleanNetwork(1, [1], 1, 1)
        initial_conditions_1 = bytearray([False, True])
        initial_conditions_2 = bytearray([False, False, False, False, False])
        not_ic_2 = bytearray([True, True, True, True, True])
        bn_1.set_conditions(initial_conditions_1)
        bn_2.set_conditions(initial_conditions_2)
        for bn_i in range(len(initial_conditions_2)):
            initial_conditions_2 = [
                (
                    initial_conditions_2[bn_j]
                    if bn_j != bn_i
                    else not initial_conditions_2[bn_j]
                )
                for bn_j in range(len(initial_conditions_2))
            ]
            bn_2.set_conditions(initial_conditions_2)
        self.assertEqual(bn_1.current_states_list[0], initial_conditions_1)
        self.assertEqual(bn_2.current_states_list[0], not_ic_2)
        self.assertEqual(bn_1.current_states_list[-1], initial_conditions_1)
        self.assertEqual(bn_2.current_states_list[-1], not_ic_2)

    def test_advance_state(self):
        # Using the same N=3 K=2 network as described above: directly assign each of 2^3 = 8 possible conditions to
        # current_states_list and advance state, assert equality of [-1] state and ground truth, additionally, for an
        # applicable initial condition, call advance state ~1000 times, and every 25 iterations assert one of two states
        # according to odd/even iterations
        # Starting with hard-coded data for an N=3 K=2 network of Boolean functions:
        # node: A                     # node: B                     # node: C
        # B   C   A                   # A   C   B                   # A   B   C
        # 0   0   1                   # 0   0   1                   # 0   0   1
        # 0   1   1                   # 0   1   1                   # 0   1   0
        # 1   0   0                   # 1   0   0                   # 1   0   1
        # 1   1   0                   # 1   1   0                   # 1   1   0
        #
        # _BooleanFunction13            _BooleanFunction13            _BooleanFunction11
        # index: 12                     index: 12                     index: 10
        bn_1 = boolean_networks.BooleanNetwork(3, [0], 2, 3)
        bn_1.nodes[0].function = boolean_networks._BooleanFunction13
        bn_1.nodes[1].function = boolean_networks._BooleanFunction13
        bn_1.nodes[2].function = boolean_networks._BooleanFunction11
        bn_1.node_inputs_assignments = [[1, 2], [0, 2], [0, 1]]
        #
        # There are 2^3 possible states of [A,B,C]: [0,0,0] [0,0,1] [0,1,0] [0,1,1] [1,0,0] [1,0,1] [1,1,0] [1,1,1]
        #
        # initial condition: [0,0,0]
        # t=0
        # A: 0    B: 0    C: 0
        # t=1
        # A: 1    B: 1    C: 1
        # t=2
        # A: 0    B: 0    C: 0    # Cycle length: 2    ### Cycle for even/odd loop assertions
        ic_1 = bytearray([False, False, False])
        ic_1_t1 = bytearray([True, True, True])
        ic_1_t_odd = bytearray([False, False, False])
        ic_1_t_even = bytearray([True, True, True])
        #
        # initial condition: [0,0,1]
        # t=0
        # A: 0    B: 0    C: 1
        # t=1
        # A: 1    B: 1    C: 1
        ic_2 = bytearray([False, False, True])
        ic_2_t1 = bytearray([True, True, True])
        #
        # initial condition: [0,1,0]
        # t=0
        # A: 0    B: 1    C: 0
        # t=1
        # A: 0    B: 1    C: 0    # Cycle length: 1 (eq. state)
        ic_3 = bytearray([False, True, False])
        ic_3_t1 = bytearray([False, True, False])
        #
        # initial condition: [0,1,1]
        # t=0
        # A: 0    B: 1    C: 1
        # t=1
        # A: 0    B: 1    C: 0
        ic_4 = bytearray([False, True, True])
        ic_4_t1 = bytearray([False, True, False])
        #
        # initial condition: [1,0,0]
        # t=0
        # A: 1    B: 0    C: 0
        # t=1
        # A: 1    B: 0    C: 1
        ic_5 = bytearray([False, False, False])
        ic_5_t1 = bytearray([True, True, True])
        #
        # initial condition: [1,0,1]
        # t=0
        # A: 1    B: 0    C: 1
        # t=1
        # A: 1    B: 0    C: 1    # Cycle length: 1 (eq. state)
        ic_6 = bytearray([True, False, True])
        ic_6_t1 = bytearray([True, False, True])
        #
        # initial condition: [1,1,0]
        # t=0
        # A: 1    B: 1    C: 0
        # t=1
        # A: 0    B: 0    C: 0
        ic_7 = bytearray([True, True, False])
        ic_7_t1 = bytearray([False, False, False])
        #
        # initial condition: [1,1,1]
        # t=0
        # A: 1    B: 1    C: 1
        # t=1
        # A: 0    B: 0    C: 0
        ic_8 = bytearray([True, True, True])
        ic_8_t1 = bytearray([False, False, False])
        num_advance_state_calls = 8
        ic_list = [ic_1, ic_2, ic_3, ic_4, ic_5, ic_6, ic_7, ic_8]
        c_t1_s = [
            ic_1_t1,
            ic_2_t1,
            ic_3_t1,
            ic_4_t1,
            ic_5_t1,
            ic_6_t1,
            ic_7_t1,
            ic_8_t1,
        ]
        for each_initial_condition, each_t1_condition in zip(ic_list, c_t1_s):
            bn_1.set_conditions(each_initial_condition)
            bn_1.advance_state()
            self.assertEqual(len(bn_1.current_states_list), 2)
            self.assertEqual(bn_1.current_states_list[0], each_initial_condition)
            self.assertEqual(bn_1.current_states_list[-1], each_t1_condition)
        self.assertEqual(bn_1.total_advance_states, num_advance_state_calls)
        bn_1.set_conditions(ic_1)
        for tas_i in range(1000):
            bn_1.advance_state()
            if tas_i % 25 == 0:
                if tas_i % 2 == 0:
                    self.assertEqual(bn_1.current_states_list[-1], ic_1_t_even)
                else:
                    self.assertEqual(bn_1.current_states_list[-1], ic_1_t_odd)
        self.assertEqual(bn_1.total_advance_states, 1008)
        self.assertEqual(len(bn_1.list_of_all_run_ins), 0)
        self.assertEqual(len(bn_1.bn_collapsed_cycles), 0)
        self.assertEqual(len(bn_1.bn_trajectories.u_records), 0)
        self.assertEqual(len(bn_1.bn_trajectories.t_records), 0)

    def test_get_random_conditions(self):
        bn_1 = boolean_networks.BooleanNetwork(1000, [0], 2, 400)
        bn_2 = boolean_networks.BooleanNetwork(0, [0], 2, 400)
        for _ in range(10):
            for each_bit in bn_1.get_random_conditions():
                self.assertIsInstance(each_bit, bool)
        self.assertEqual(bn_2.get_random_conditions(), [])

    def test_add_cycle(
        self,
    ):  # This also tests run_in_from_random, and run_in_from_conditions
        bn_1 = boolean_networks.BooleanNetwork(3, [0], 2, 200)
        bn_1.nodes[0].function = boolean_networks._BooleanFunction13
        bn_1.nodes[1].function = boolean_networks._BooleanFunction13
        bn_1.nodes[2].function = boolean_networks._BooleanFunction11
        bn_1.node_inputs_assignments = [[1, 2], [0, 2], [0, 1]]
        bn_2 = boolean_networks.BooleanNetwork(3, [0], 100, 200)
        bn_2.nodes[0].function = boolean_networks._BooleanFunction13
        bn_2.nodes[1].function = boolean_networks._BooleanFunction13
        bn_2.nodes[2].function = boolean_networks._BooleanFunction11
        bn_2.node_inputs_assignments = [[1, 2], [0, 2], [0, 1]]
        bn_3 = boolean_networks.BooleanNetwork(20, [i for i in range(16)], 2, 400)
        bn_4 = boolean_networks.BooleanNetwork(200, [0], 100, 400)
        # Starting with hard-coded data for an N=3 K=2 network of Boolean functions:
        # node: A             # node: B             # node: C
        # B   C   A           # A   C   B           # A   B   C
        # 0   0   1           # 0   0   1           # 0   0   1
        # 0   1   1           # 0   1   1           # 0   1   0
        # 1   0   0           # 1   0   0           # 1   0   1
        # 1   1   0           # 1   1   0           # 1   1   0
        #
        # There are 2^3 possible states of [A,B,C]: [0,0,0] [0,0,1] [0,1,0] [0,1,1] [1,0,0] [1,0,1] [1,1,0] [1,1,1]
        #
        # initial condition: [0,0,0]
        # t=0
        # A: 0    B: 0    C: 0
        # t=1
        # A: 1    B: 1    C: 1
        # t=2
        # A: 0    B: 0    C: 0    # Cycle length: 2    [ ### cycle index: 0 ]    [ num observations: 1 ]
        run_in_1 = [[False, False, False], [True, True, True], [False, False, False]]
        observed_cycle_1 = run_in_1  # Emulating logic in class BooleanNetwork
        #
        # initial condition: [0,0,1]
        # t=0
        # A: 0    B: 0    C: 1
        # t=1
        # A: 1    B: 1    C: 1
        # t=2
        # A: 0    B: 0    C: 0
        # t=3
        # A: 1    B: 1    C: 1    # Cycle length: 2    [ ### cycle index: 0 ]    [ num observations: 2 ]
        run_in_2 = [
            [False, False, True],
            [True, True, True],
            [False, False, False],
            [True, True, True],
        ]
        observed_cycle_2 = run_in_2[1:4:1]  # Emulating logic in class BooleanNetwork
        #
        # initial condition: [0,1,0]
        # t=0
        # A: 0    B: 1    C: 0
        # t=1
        # A: 0    B: 1    C: 0    # Cycle length: 1 (eq. state)    [ ### cycle index: 1 ]    [ num observations: 1 ]
        run_in_3 = [[False, True, False], [False, True, False]]
        observed_cycle_3 = run_in_3  # Emulating logic in class BooleanNetwork
        #
        # initial condition: [0,1,1]
        # t=0
        # A: 0    B: 1    C: 1
        # t=1
        # A: 0    B: 1    C: 0
        # t=2
        # A: 0    B: 1    C: 0    # Cycle length: 1 (eq. state)    [ ### cycle index: 1 ]    [ num observations: 2 ]
        run_in_4 = [[False, True, True], [False, True, False], [False, True, False]]
        observed_cycle_4 = run_in_4[1:3:1]  # Emulating logic in class BooleanNetwork
        #
        # initial condition: [1,0,0]
        # t=0
        # A: 1    B: 0    C: 0
        # t=1
        # A: 1    B: 0    C: 1
        # t=2
        # A: 1    B: 0    C: 1    # Cycle length: 1 (eq. state)    [ ### cycle index: 2 ]    [ num observations: 1 ]
        run_in_5 = [[True, False, False], [True, False, True], [True, False, True]]
        observed_cycle_5 = run_in_5[1:3:1]  # Emulating logic in class BooleanNetwork
        #
        # initial condition: [1,0,1]
        # t=0
        # A: 1    B: 0    C: 1
        # t=1
        # A: 1    B: 0    C: 1    # Cycle length: 1 (eq. state)    [ ### cycle index: 2 ]    [ num observations: 2 ]
        run_in_6 = [[True, False, True], [True, False, True]]
        observed_cycle_6 = run_in_6  # Emulating logic in class BooleanNetwork
        #
        # initial condition: [1,1,0]
        # t=0
        # A: 1    B: 1    C: 0
        # t=1
        # A: 0    B: 0    C: 0
        # t=2
        # A: 1    B: 1    C: 1
        # t=3
        # A: 0    B: 0    C: 0    # Cycle length: 2    [ ### cycle index: 0 ]    [ num observations: 3 ]
        run_in_7 = [
            [True, True, False],
            [False, False, False],
            [True, True, True],
            [False, False, False],
        ]
        observed_cycle_7 = run_in_7[1:4:1]  # Emulating logic in class BooleanNetwork
        #
        # initial condition: [1,1,1]
        # t=0
        # A: 1    B: 1    C: 1
        # t=1
        # A: 0    B: 0    C: 0
        # t=2
        # A: 1    B: 1    C: 1    # Cycle length: 2    [ ### cycle index: 0 ]    [ num observations: 4 ]
        run_in_8 = [[True, True, True], [False, False, False], [True, True, True]]
        observed_cycle_8 = run_in_8  # Emulating logic in class BooleanNetwork
        # ### The above cycle indices are not expected for finite state machine operation from random initial conditions
        list_of_observed_cycles = [
            observed_cycle_1,
            observed_cycle_2,
            observed_cycle_3,
            observed_cycle_4,
            observed_cycle_5,
            observed_cycle_6,
            observed_cycle_7,
            observed_cycle_8,
        ]
        list_of_run_ins = [
            run_in_1,
            run_in_2,
            run_in_3,
            run_in_4,
            run_in_5,
            run_in_6,
            run_in_7,
            run_in_8,
        ]
        # converting run-in lists to lists of bytearrays
        for i in range(len(list_of_run_ins)):
            for j in range(len(list_of_run_ins[i])):
                list_of_run_ins[i][j] = bytearray(list_of_run_ins[i][j])
        # for bn_1 and bn_2, each execution of run_ins_from_homogeneous_states calls run_in_from_conditions twice
        bn_1.run_ins_from_homogeneous_states()
        bn_2.run_ins_from_homogeneous_states()
        self.assertEqual(len(bn_1.bn_trajectories.t_records), 2)
        self.assertEqual(len(bn_2.bn_trajectories.t_records), 2)
        self.assertEqual(len(bn_1.bn_trajectories.u_records), 0)
        self.assertEqual(len(bn_2.bn_trajectories.u_records), 0)
        self.assertEqual(len(bn_1.bn_collapsed_cycles), 1)
        self.assertEqual(len(bn_2.bn_collapsed_cycles), 1)
        self.assertEqual(
            sum(
                [
                    each_cycle_record_1.num_observations
                    for each_cycle_record_1 in bn_1.bn_collapsed_cycles.cycle_records
                ]
            ),
            2,
        )
        self.assertEqual(
            sum(
                [
                    each_cycle_record_2.num_observations
                    for each_cycle_record_2 in bn_2.bn_collapsed_cycles.cycle_records
                ]
            ),
            2,
        )
        self.assertTrue(
            len(bn_1.list_of_all_run_ins)
            == bn_1.bn_collapsed_cycles.cycle_records[0].num_observations
        )
        self.assertTrue(
            len(bn_2.list_of_all_run_ins)
            == bn_2.bn_collapsed_cycles.cycle_records[-1].num_observations
        )
        self.assertEqual(bn_1.total_advance_states, 4)
        self.assertEqual(bn_2.total_advance_states, 4)
        # this loop (bn_1, bn_2) is excessive but is the same as that used below
        for each_cycle_record_1, each_cycle_record_2 in zip(
            bn_1.bn_collapsed_cycles.cycle_records,
            bn_2.bn_collapsed_cycles.cycle_records,
        ):
            self.assertTrue(
                (
                    each_cycle_record_1.cycle_states_list[-2]
                    == list_of_run_ins[tac_i][-2]
                    if each_cycle_record_1.cycle_states_list[-1]
                    == list_of_run_ins[tac_i][-1]
                    else True
                )
                for tac_i in range(len(list_of_run_ins))
            )
            self.assertTrue(
                (
                    each_cycle_record_2.cycle_states_list[-2]
                    == list_of_run_ins[tac_i][-2]
                    if each_cycle_record_2.cycle_states_list[-1]
                    == list_of_run_ins[tac_i][-1]
                    else True
                )
                for tac_i in range(len(list_of_run_ins))
            )
        self.assertTrue(
            sum(
                [
                    bn_1.bn_collapsed_cycles.cycle_records[tac_L_1].num_observations
                    for tac_L_1 in range(len(bn_1.bn_collapsed_cycles))
                ]
            )
            + len(bn_1.bn_trajectories.u_records)
            == len(bn_1.list_of_all_run_ins)
        )  # Revised 08-06-2024
        self.assertTrue(
            sum(
                [
                    bn_2.bn_collapsed_cycles.cycle_records[tac_m_1].num_observations
                    for tac_m_1 in range(len(bn_2.bn_collapsed_cycles))
                ]
            )
            + len(bn_2.bn_trajectories.u_records)
            == len(bn_2.list_of_all_run_ins)
        )  # Revised 08-06-2024
        self.assertTrue(
            len(bn_1.bn_trajectories.t_records)
            == sum(
                [
                    bn_1.bn_collapsed_cycles.cycle_records[tac_L_2].num_observations
                    for tac_L_2 in range(len(bn_1.bn_collapsed_cycles))
                ]
            )
        )  # Added 08-06-2024
        self.assertTrue(
            len(bn_1.bn_trajectories.t_records)
            == sum(
                [
                    bn_2.bn_collapsed_cycles.cycle_records[tac_L_3].num_observations
                    for tac_L_3 in range(len(bn_2.bn_collapsed_cycles))
                ]
            )
        )  # Added 08-06-2024
        bn_1.reset_network_data()
        bn_2.reset_network_data()
        for tac_j in range(len(list_of_run_ins)):
            bn_1.run_in_from_conditions(list_of_run_ins[tac_j][0])
            bn_2.run_in_from_conditions(list_of_run_ins[tac_j][0])
        self.assertEqual(bn_1.current_states_list[-1], run_in_8[-1])
        self.assertEqual(bn_2.current_states_list[-1], run_in_8[-1])
        # run_in_from_conditions calls advance_state and records data from each step
        # run_in_from_random calls run_in_from_conditions(conditions from get_random_conditions)
        # add_cycle calls run_in_from_random
        bn_1.reset_network_data()
        bn_2.reset_network_data()
        for tac_j in range(2):
            bn_1.add_cycle()
        for tac_k in range(10000):
            bn_2.add_cycle()
        for each_cycle_record_1_1 in bn_1.bn_collapsed_cycles.cycle_records:
            self.assertTrue(
                (
                    each_cycle_record_1_1.cycle_states_list[-2]
                    == list_of_run_ins[tac_i][-2]
                    if each_cycle_record_1_1.cycle_states_list[-1]
                    == list_of_run_ins[tac_i][-1]
                    else True
                )
                for tac_i in range(len(list_of_run_ins))
            )
        for each_cycle_record_2_1 in bn_2.bn_collapsed_cycles.cycle_records:
            self.assertTrue(
                (
                    each_cycle_record_2_1.cycle_states_list[-2]
                    == list_of_run_ins[tac_i][-2]
                    if each_cycle_record_2_1.cycle_states_list[-1]
                    == list_of_run_ins[tac_i][-1]
                    else True
                )
                for tac_i in range(len(list_of_run_ins))
            )
        self.assertTrue(
            len(bn_3.bn_trajectories.t_records) + len(bn_3.bn_trajectories.u_records)
            == len(bn_3.list_of_all_run_ins)
        )
        self.assertTrue(
            len(bn_4.bn_trajectories.t_records) + len(bn_4.bn_trajectories.u_records)
            == len(bn_4.list_of_all_run_ins)
        )
        self.assertTrue(len(bn_3.bn_trajectories) == len(bn_3.list_of_all_run_ins))
        self.assertTrue(len(bn_4.bn_trajectories) == len(bn_4.list_of_all_run_ins))
        self.assertTrue(
            sum(
                [
                    bn_3.bn_collapsed_cycles.cycle_records[tac_L].num_observations
                    for tac_L in range(len(bn_3.bn_collapsed_cycles))
                ]
            )
            + len(bn_3.bn_trajectories.u_records)
            == len(bn_3.list_of_all_run_ins)
        )
        self.assertTrue(
            sum(
                [
                    bn_4.bn_collapsed_cycles.cycle_records[tac_m].num_observations
                    for tac_m in range(len(bn_4.bn_collapsed_cycles))
                ]
            )
            + len(bn_4.bn_trajectories.u_records)
            == len(bn_4.list_of_all_run_ins)
        )

    def test_compute_unit_perturbations(self):
        # 08-06-2024 -- Debugging of compute_unit_perturbations: function was missing condition for None as index of
        # states in t_record run-in's, the same condition was also needed in t_cup_assertions for rsa_end_index_sum
        def get_all_list_bool(length: int, sequence_elements: list[bool]):
            """For a net of 3 Boolean functions:
            There are 2^3 possible states of [A,B,C]: [0,0,0] [0,0,1] [0,1,0] [0,1,1] [1,0,0] [1,0,1] [1,1,0] [1,1,1];
            this function returns all possible states."""
            galb_return_sequence_list = [[]]
            for _ in range(length):
                galb_temp_sequence_list = []
                for each_list in galb_return_sequence_list:
                    for each_element in sequence_elements:
                        galb_temp_sequence_list.append(each_list + [each_element])
                galb_return_sequence_list = galb_temp_sequence_list.copy()
            return galb_return_sequence_list

        self.assertEqual(
            get_all_list_bool(3, [False, True]),
            [
                [False, False, False],
                [False, False, True],
                [False, True, False],
                [False, True, True],
                [True, False, False],
                [True, False, True],
                [True, True, False],
                [True, True, True],
            ],
        )

        def t_cup_assertions(a_bn: boolean_networks.BooleanNetwork):
            """Where a unit_perturbations_matrix is not [],
            each row sums to the number of nodes * number of states in the cycle corresponding to that row
            each column sums to the number of perturbation records ending in that column (cycle index, or None).
            Where a unit_perturbations_matrix is computed after completely sampling all possible conditions of a net,
            the last row sums to 0 and the last column sums to 0."""
            # TODO add logic for row corresponding to starting in None state (TBD)
            if (
                a_bn.total_advance_states == 0
                and a_bn.cycles_unit_perturbations_transition_matrix is not None
            ):
                self.assertEqual(
                    a_bn.cycles_unit_perturbations_transition_matrix, [[0]]
                )
            elif a_bn.cycles_unit_perturbations_transition_matrix is not None:
                rsa_total_sum = 0
                # The total sum of all entries in the transition matrix should be equal to the number of perturbation records
                for rsa_i in range(
                    len(a_bn.cycles_unit_perturbations_transition_matrix)
                ):
                    for rsa_j in range(
                        len(a_bn.cycles_unit_perturbations_transition_matrix)
                    ):
                        rsa_total_sum += (
                            a_bn.cycles_unit_perturbations_transition_matrix[rsa_i][
                                rsa_j
                            ]
                        )
                tcup_all_perturbations_records = (
                    a_bn.cycles_unit_perturbations_records
                    + a_bn.t_trajectories_unit_perturbations_records
                    + a_bn.u_trajectories_unit_perturbations_records
                )
                self.assertTrue(rsa_total_sum == len(tcup_all_perturbations_records))
                # Each column should sum to the same number of record .end_index fields that indicate a transition ending on a cycle of that index
                for rsa_L in range(
                    len(a_bn.cycles_unit_perturbations_transition_matrix[0])
                ):
                    rsa_col_sum = sum(
                        [
                            a_bn.cycles_unit_perturbations_transition_matrix[rsa_k][
                                rsa_L
                            ]
                            for rsa_k in range(
                                len(a_bn.cycles_unit_perturbations_transition_matrix[0])
                            )
                        ]
                    )
                    if (
                        rsa_L
                        == len(a_bn.cycles_unit_perturbations_transition_matrix[0]) - 1
                    ):
                        rsa_end_index_sum = sum(
                            [
                                (
                                    1
                                    if tcup_all_perturbations_records[rsa_n].end_index
                                    is None
                                    else 0
                                )
                                for rsa_n in range(len(tcup_all_perturbations_records))
                            ]
                        )
                    else:
                        rsa_end_index_sum = sum(
                            [
                                (
                                    1
                                    if tcup_all_perturbations_records[rsa_n].end_index
                                    == rsa_L
                                    else 0
                                )
                                for rsa_n in range(len(tcup_all_perturbations_records))
                            ]
                        )
                    self.assertTrue(rsa_col_sum == rsa_end_index_sum)
                # over each row of the transition matrix, the row sum should be equal to num nodes * cycle states
                for rsa_m, each_row in zip(
                    range(len(a_bn.cycles_unit_perturbations_transition_matrix)),
                    a_bn.cycles_unit_perturbations_transition_matrix,
                ):
                    if (
                        rsa_m
                        != len(a_bn.cycles_unit_perturbations_transition_matrix) - 1
                    ):
                        rsa_row_sum = sum(each_row)
                        rsa_num_nodes = len(a_bn)
                        rsa_num_states = len(
                            a_bn.bn_collapsed_cycles.cycle_records[rsa_m]
                        )
                        self.assertTrue(rsa_row_sum == rsa_num_nodes * rsa_num_states)

        # start testing using helper function t_cup_assertions (t_cup: test cycles unit perturbations (transition matrix))
        bn_1 = boolean_networks.BooleanNetwork(3, [0], 2, 200)
        bn_1.nodes[0].function = boolean_networks._BooleanFunction13
        bn_1.nodes[1].function = boolean_networks._BooleanFunction13
        bn_1.nodes[2].function = boolean_networks._BooleanFunction11
        bn_1.node_inputs_assignments = [[1, 2], [0, 2], [0, 1]]
        bn_2 = boolean_networks.BooleanNetwork(3, [0], 2, 200)
        bn_2.nodes[0].function = (
            boolean_networks._BooleanFunction2
        )  # number of inputs is 2
        bn_2.nodes[1].function = (
            boolean_networks._BooleanFunction3
        )  # number of inputs is 2
        bn_2.nodes[2].function = (
            boolean_networks._BooleanFunction5
        )  # number of inputs is 2
        bn_2.node_inputs_assignments = [[1, 2], [0, 2], [0, 1]]
        bn_3 = boolean_networks.BooleanNetwork(
            3, [tcup_n for tcup_n in range(16)], 2, 200
        )
        bn_3.node_inputs_assignments = [[1, 2], [0, 2], [0, 1]]
        bn_4 = boolean_networks.BooleanNetwork(
            5, [tcup_m for tcup_m in range(16)], 2, 400
        )
        bn_5 = boolean_networks.BooleanNetwork(
            20, [tcup_o for tcup_o in range(16)], 2, 40
        )
        bn_6 = boolean_networks.BooleanNetwork(40, [0, 15], 2, 400)
        bn_7 = boolean_networks.BooleanNetwork(
            20, [tcup_o for tcup_o in range(16)], 2, 0
        )
        bn_8 = boolean_networks.BooleanNetwork(40, [0, 15], 2, 0)
        list_of_bn = [bn_1, bn_2, bn_3, bn_4, bn_5, bn_6, bn_7, bn_8]
        list_of_all_sequences_per_len_bn_less_than_10 = []
        len_bn_set = set()
        for each_bn in list_of_bn:
            if len(each_bn) < 10:
                if not len(each_bn) in len_bn_set:
                    len_bn_set.add(len(each_bn))
                    list_of_all_sequences_per_len_bn_less_than_10.append(
                        get_all_list_bool(len(each_bn), [False, True])
                    )
        # in all cases the sum of each row of the transition matrix is equal to num_nodes*num_states, including where
        # computation occurs before any states have been observed  ### for each row, row sum == |cycle states| * |nodes|
        for each_bn_empty in list_of_bn:
            each_bn_empty.compute_unit_perturbations_matrix()
            self.assertEqual(
                each_bn_empty.cycles_unit_perturbations_transition_matrix, [[0]]
            )
            each_bn_empty.reset_network_data()
        # compute after only computing run-ins from homogenous conditions
        for each_bn_0_1 in list_of_bn:
            each_bn_0_1.run_ins_from_homogeneous_states()
            each_bn_0_1.compute_unit_perturbations_matrix()
            t_cup_assertions(each_bn_0_1)
            each_bn_0_1.reset_network_data()
        # compute after run_in_from_conditions for all run_ins of N<10
        for each_bn_for_n10 in list_of_bn:
            if len(each_bn_for_n10) < 10:
                for each_list in list_of_all_sequences_per_len_bn_less_than_10:
                    # print(each_list)
                    if len(each_list[0]) == len(each_bn_for_n10):
                        # for each_condition in list_of_all_sequences_per_len_bn_less_than_10:
                        for each_condition in each_list:
                            each_bn_for_n10.run_in_from_conditions(each_condition)
            each_bn_for_n10.compute_unit_perturbations_matrix()
            self.assertEqual(
                sum(each_bn_for_n10.cycles_unit_perturbations_transition_matrix[-1]), 0
            )
            self.assertEqual(
                sum(
                    [
                        each_row[-1]
                        for each_row in each_bn_for_n10.cycles_unit_perturbations_transition_matrix
                    ]
                ),
                0,
            )
            t_cup_assertions(each_bn_for_n10)
            each_bn_for_n10.reset_network_data()
        # compute after run_in_from_random or add_cycle a times = (1, 2, 4, 8, 16, 32, 64)
        for each_boolean in [False, True]:
            for each_integer in [1, 2, 4, 8, 16, 32, 64]:
                for each_bn_for_integers in list_of_bn:
                    for tcup_i in range(each_integer):
                        if each_boolean:
                            each_bn_for_integers.run_in_from_random()
                        else:
                            each_bn_for_integers.add_cycle()
                    each_bn_for_integers.compute_unit_perturbations_matrix()
                    t_cup_assertions(each_bn_for_integers)
                    each_bn_for_integers.reset_network_data()
        # From earlier testing, revision:
        # for each_bn_A in list_of_bn_n3_A + list_of_bn_n20_n200_A:
        #     each_bn_A.run_ins_from_homogeneous_states()
        #     each_bn_A.compute_unit_perturbations_matrix()
        #     list_of_all_bn.append(each_bn_A)
        #     # Initially this produces a result of list_of_all_run_ins ==
        #     #                                           [[[T,T,T], [F,F,F], [F,F,F]], [[T,T,T], [F,F,F], [T,T,T]]],
        #     # but without the call to .compute_unit_perturbation_matrix() it is as expected for run-ins from
        #     # homogeneous states for this N=3 net  ### Made a few changes that fixed it, didn't pinpoint cause,
        #     # 08-05-24, then did pinpoint the cause:
        #     #   ### Sufficient to recreate error:  at line ~474, boolean_networks.py
        #     #   ###     c_perturbed_state = each_state_c;
        #     #   ###     c_perturbed_state[c_j_1] = not c_perturbed_state[c_j_1], rather than appending literals
        #     #   ###  also added a .copy(), earlier, to list appendage of current_states_list, in run_in_from_cond...
        #     #   ### Sufficient to prevent recreated error:
        #     #   ###     c_perturbed_state = each_state_c.copy()

    def test_compute_unit_perturbations_and_t_u(self):
        def get_all_list_bool(length: int, sequence_elements: list[bool]):
            """For a net of 3 Boolean functions:
            There are 2^3 possible states of [A,B,C]: [0,0,0] [0,0,1] [0,1,0] [0,1,1] [1,0,0] [1,0,1] [1,1,0] [1,1,1];
            this function returns all possible states."""
            galb_return_sequence_list = [[]]
            for _ in range(length):
                galb_temp_sequence_list = []
                for each_list in galb_return_sequence_list:
                    for each_element in sequence_elements:
                        galb_temp_sequence_list.append(each_list + [each_element])
                galb_return_sequence_list = galb_temp_sequence_list.copy()
            return galb_return_sequence_list

        self.assertEqual(
            get_all_list_bool(3, [False, True]),
            [
                [False, False, False],
                [False, False, True],
                [False, True, False],
                [False, True, True],
                [True, False, False],
                [True, False, True],
                [True, True, False],
                [True, True, True],
            ],
        )

        def t_cup_assertions(a_bn: boolean_networks.BooleanNetwork):
            """Where a unit_perturbations_matrix is not [],
            each row sums to the number of nodes * number of states in the cycle corresponding to that row
            each column sums to the number of perturbation records ending in that column (cycle index, or None).
            Where a unit_perturbations_matrix is computed after completely sampling all possible conditions of a net,
            the last row sums to 0 and the last column sums to 0."""
            # TODO add logic for row corresponding to starting in None state
            # = sum(length u_records run_in s) * |nodes|
            if a_bn.total_unit_perturbations_transition_matrix is not None:
                rsa_total_sum = 0
                # The sum of all entries in the transition matrix should be equal to |{observed states}| * |nodes|
                #                                                                    ### using .list_of_all_run_ins
                for rsa_i in range(
                    len(a_bn.total_unit_perturbations_transition_matrix)
                ):
                    for rsa_j in range(
                        len(a_bn.total_unit_perturbations_transition_matrix)
                    ):
                        rsa_total_sum += (
                            a_bn.total_unit_perturbations_transition_matrix[rsa_i][
                                rsa_j
                            ]
                        )
                observed_set = set(
                    [
                        "".join([str(int(s)) for s in state])
                        for run_in in a_bn.list_of_all_run_ins
                        for state in run_in
                    ]
                )
                self.assertTrue(rsa_total_sum == len(observed_set) * len(a_bn))
                # over each row of the transition matrix, the row sum should be equal to num nodes * num states
                #
                # for row:=matrix[-1], index==None, row sum = |{states labeled by end_index}| * |nodes|
                cycle_states_set = set()
                for cycle in a_bn.bn_collapsed_cycles.cycle_records:
                    for state in cycle.cycle_states_list:
                        cycle_states_set.add("".join([str(int(s)) for s in state]))
                of_cycles_sets_list = [
                    set() for _ in a_bn.bn_collapsed_cycles.cycle_records
                ] + [
                    set()
                ]  # [-1] is for None index
                # index of the last run-in state
                for run_in in a_bn.list_of_all_run_ins:
                    cycle_index = a_bn.bn_collapsed_cycles.get_index(run_in[-1])
                    # print(f"cycle_index {cycle_index}")
                    of_cycles_sets_list[
                        cycle_index if cycle_index is not None else -1
                    ].update(
                        ["".join([str(int(s)) for s in state]) for state in run_in]
                    )
                for rsa_m, each_row in zip(
                    range(len(a_bn.total_unit_perturbations_transition_matrix)),
                    a_bn.total_unit_perturbations_transition_matrix,
                ):
                    if (
                        rsa_m
                        != len(a_bn.total_unit_perturbations_transition_matrix) - 1
                    ):  # excluding last row (index=None)
                        rsa_row_sum = sum(each_row)
                        rsa_num_nodes = len(a_bn)
                        rsa_num_states = len(of_cycles_sets_list[rsa_m])
                        self.assertTrue(rsa_row_sum == rsa_num_nodes * rsa_num_states)

        # start testing using helper function t_cup_assertions (t_cup: test cycles unit perturbations (transition matrix))
        bn_1 = boolean_networks.BooleanNetwork(3, [0], 2, 200)
        bn_1.nodes[0].function = boolean_networks._BooleanFunction13
        bn_1.nodes[1].function = boolean_networks._BooleanFunction13
        bn_1.nodes[2].function = boolean_networks._BooleanFunction11
        bn_1.node_inputs_assignments = [[1, 2], [0, 2], [0, 1]]
        bn_2 = boolean_networks.BooleanNetwork(3, [0], 2, 200)
        bn_2.nodes[0].function = (
            boolean_networks._BooleanFunction2
        )  # number of inputs is 2
        bn_2.nodes[1].function = (
            boolean_networks._BooleanFunction3
        )  # number of inputs is 2
        bn_2.nodes[2].function = (
            boolean_networks._BooleanFunction5
        )  # number of inputs is 2
        bn_2.node_inputs_assignments = [[1, 2], [0, 2], [0, 1]]
        bn_3 = boolean_networks.BooleanNetwork(
            3, [tcup_n for tcup_n in range(16)], 2, 200
        )
        bn_3.node_inputs_assignments = [[1, 2], [0, 2], [0, 1]]
        bn_4 = boolean_networks.BooleanNetwork(
            5, [tcup_m for tcup_m in range(16)], 2, 400
        )
        bn_5 = boolean_networks.BooleanNetwork(
            20, [tcup_o for tcup_o in range(16)], 2, 40
        )
        bn_6 = boolean_networks.BooleanNetwork(40, [0, 15], 2, 400)
        bn_7 = boolean_networks.BooleanNetwork(
            20, [tcup_o for tcup_o in range(16)], 2, 0
        )
        bn_8 = boolean_networks.BooleanNetwork(40, [0, 15], 2, 0)
        list_of_bn = [bn_1, bn_2, bn_3, bn_4, bn_5, bn_6, bn_7, bn_8]
        list_of_all_sequences_per_len_bn_less_than_10 = []
        len_bn_set = set()
        for each_bn in list_of_bn:
            if len(each_bn) < 10:
                if not len(each_bn) in len_bn_set:
                    len_bn_set.add(len(each_bn))
                    list_of_all_sequences_per_len_bn_less_than_10.append(
                        get_all_list_bool(len(each_bn), [False, True])
                    )
        # in all cases the sum of each row of the transition matrix is equal to num_nodes*num_states, including where
        # computation occurs before any states have been observed
        for each_bn_empty in list_of_bn:
            each_bn_empty.compute_unit_perturbations_matrix(1, True)
            self.assertEqual(
                each_bn_empty.total_unit_perturbations_transition_matrix, [[0]]
            )
            each_bn_empty.reset_network_data()
        # compute after only computing run-ins from homogenous conditions
        # for each in list_of_bn:
        #     print(each)
        #     print(each.bn_collapsed_cycles.cycle_records)
        for each_bn_0_1 in list_of_bn:
            each_bn_0_1.run_ins_from_homogeneous_states()
            each_bn_0_1.compute_unit_perturbations_matrix(1, True)
            # print(each_bn_0_1.cycles_unit_perturbations_transition_matrix)
            # print(each_bn_0_1)
            # print(each_bn_0_1.bn_collapsed_cycles.cycle_records)
            # print(f"each_bn_0_1 cycle records [{list_of_bn.index(each_bn_0_1)}]: {each_bn_0_1.bn_collapsed_cycles.cycle_records}")
            t_cup_assertions(each_bn_0_1)
            each_bn_0_1.reset_network_data()
        # compute after run_in_from_conditions for all run_ins of N<10
        for each_bn_for_n10 in list_of_bn:
            if len(each_bn_for_n10) < 10:
                for each_list in list_of_all_sequences_per_len_bn_less_than_10:
                    # print(each_list)
                    if len(each_list[0]) == len(each_bn_for_n10):
                        # for each_condition in list_of_all_sequences_per_len_bn_less_than_10:
                        for each_condition in each_list:
                            each_bn_for_n10.run_in_from_conditions(each_condition)
            each_bn_for_n10.compute_unit_perturbations_matrix(1, True)
            self.assertEqual(
                sum(each_bn_for_n10.total_unit_perturbations_transition_matrix[-1]), 0
            )
            self.assertEqual(
                sum(
                    [
                        each_row[-1]
                        for each_row in each_bn_for_n10.total_unit_perturbations_transition_matrix
                    ]
                ),
                0,
            )
            t_cup_assertions(each_bn_for_n10)
            each_bn_for_n10.reset_network_data()
        # compute after run_in_from_random or add_cycle a times = (1, 2, 4, 8, 16, 32, 64)
        for each_boolean in [False, True]:
            for each_integer in [1, 2, 4, 8, 16, 32, 64]:
                for each_bn_for_integers in list_of_bn:
                    for tcup_i in range(each_integer):
                        if each_boolean:
                            each_bn_for_integers.run_in_from_random()
                        else:
                            each_bn_for_integers.add_cycle()
                    each_bn_for_integers.compute_unit_perturbations_matrix(1, True)
                    t_cup_assertions(each_bn_for_integers)
                    each_bn_for_integers.reset_network_data()


# # signature    line FIRST through line LAST
# # DATES
# class Name(unittest.TestCase):
#     def test_(self):
#         pass

#                                                                                         # this can also be run with coverage
if (
    __name__ == "__main__"
):  # this executes when this file is run by Python: in an ipynb: !python3 testing__boolean_networks.py
    unittest.main()  # Run this one to execute all test cases (must start with "test_..." )
