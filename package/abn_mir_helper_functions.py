from package.abundant_boolean_networks import *


# def binary_exponentiate(num: float | int, )
# for decimal exponent: same as for integer, but square root
# ex: 2^1.01 = 2^1. * 2^0.0 * 2^0.01
#            = 2    * 1     * 2^(1 / 100)
# consider using base-Phi (base = (sqrt(5) + 1) / 2)


# circa 08-18-2024
def random_abundant_nodes(abun_bn: AbundantBooleanNetwork, cycle_index: int):
    constant_node_index_list = []
    for bv_L in range(len(abun_bn.bn_collapsed_cycles.cycle_records[cycle_index][0])):
        bv_bool_1 = True
        for bv_m in range(len(abun_bn.bn_collapsed_cycles.cycle_records[cycle_index])):
            if (
                abun_bn.bn_collapsed_cycles.cycle_records[cycle_index][bv_m][bv_L]
                != abun_bn.bn_collapsed_cycles.cycle_records[cycle_index][0][bv_L]
            ):
                bv_bool_1 = False
                break
        if bv_bool_1:
            constant_node_index_list.append(bv_L)
    if len(constant_node_index_list) == 0:
        constant_node_index_list = [
            all_n for all_n in range(0, len(abun_bn.abundant_nodes_1))
        ]
    for bun_i, each_abun in zip(
        range(len(abun_bn.abundant_nodes_1)), abun_bn.abundant_nodes_1
    ):
        each_abun.random_setup(len(abun_bn.bn_collapsed_cycles.cycle_records[0]))
        counter_abun = 0
        while abun_bn.abundant_n_assignments[bun_i] in constant_node_index_list:
            abun_bn.abundant_n_assignments[bun_i] = SystemRandom().choice(
                constant_node_index_list
            )
            counter_abun += 1
            if counter_abun > len(abun_bn) * 2:
                break


# circa 09-17-2024
def random_abundant_nodes_2(abun_bn: AbundantBooleanNetwork, cycle_index: int):
    constant_node_index_list = []
    for bv_L in range(len(abun_bn.bn_collapsed_cycles.cycle_records[cycle_index][0])):
        bv_bool_1 = True
        for bv_m in range(len(abun_bn.bn_collapsed_cycles.cycle_records[cycle_index])):
            if (
                abun_bn.bn_collapsed_cycles.cycle_records[cycle_index][bv_m][bv_L]
                != abun_bn.bn_collapsed_cycles.cycle_records[cycle_index][0][bv_L]
            ):
                bv_bool_1 = False
                break
        if bv_bool_1:
            constant_node_index_list.append(bv_L)
    if len(constant_node_index_list) == 0:
        constant_node_index_list = [
            all_n for all_n in range(0, len(abun_bn.abundant_nodes_1))
        ]
    avg_transcription_rate = 0
    for bun_i, each_abun in zip(
        range(len(abun_bn.abundant_nodes_1)), abun_bn.abundant_nodes_1
    ):
        each_abun.random_setup(len(abun_bn.bn_collapsed_cycles.cycle_records[0]))
        avg_transcription_rate += each_abun.transcription_rate
        counter_abun = 0
        # while (abun_bn.abundant_n_assignments[bun_i] in constant_node_index_list):
        while constant_node_index_list.count(abun_bn.abundant_n_assignments[bun_i]):
            abun_bn.abundant_n_assignments[bun_i] = SystemRandom().choice(
                constant_node_index_list
            )
            counter_abun += 1
            if counter_abun > len(abun_bn) * 2:
                break
    avg_transcription_rate /= len(abun_bn.abundant_nodes_1)
    for each_abun in abun_bn.abundant_nodes_1:
        if each_abun.transcription_rate <= avg_transcription_rate:
            each_abun.transcription_rate *= 20
            each_abun.current_degradation_rate = each_abun.current_degradation_rate ** (
                1 / 2
            )
            each_abun.base_deg_rate = each_abun.base_deg_rate ** (1 / 2)


# def conditions_list_from_variants(length_of_variant_region: int, sequence_elements: list, backbone_sequence: list):  # DONE 08-18 Move to helper methods, maybe replace backbone_... with B.N. ref.
#     clfv_list_conditions = []
#     clfv_all_seq = get_all_list_bool(length_of_variant_region, sequence_elements)
#     clfv_start_indices = [0, int(len(backbone_sequence) / 2) - int(len(clfv_all_seq[0]) / 2) - 1, len(backbone_sequence) - int(len(clfv_all_seq[0])) - 1]
#     for clfv_i in range(3):
#         for clfv_j in range(len(clfv_all_seq)):
#             clfv_temp_list = []
#             clfv_k = 0
#             while clfv_k < len(backbone_sequence):
#                 if clfv_k == clfv_start_indices[clfv_i]:
#                     clfv_k += len(clfv_all_seq[0])
#                     for clfv_L in range(len(clfv_all_seq[0])):
#                         clfv_temp_list.append(clfv_all_seq[clfv_j][clfv_L])
#                         # clfv_k += 1
#                 else:
#                     clfv_temp_list.append(backbone_sequence[clfv_k])
#                     clfv_k += 1
#             clfv_list_conditions.append(clfv_temp_list.copy())  # Works 1-deep? See pinpointed error in first unittest
#     return clfv_list_conditions
