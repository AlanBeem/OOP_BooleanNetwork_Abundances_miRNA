# make TOC class with docstring of all filenames and descriptions (/ aggregate docstrings from each)
from package.abn_mir_plotting_functions import select_network
from random import SystemRandom
from package.boolean_networks import BooleanNetwork
from package.abundant_boolean_networks import AbundantBooleanNetwork
from package.abundant_boolean_networks_with_micro_rna import (
    MicroRNAAbundantBooleanNetwork,
    MicroRNANode,
    get_random_sequence,
    get_rna_reverse_complement,
)
from package.abn_mir_plotting_functions import mutate_and_return_2


def get_mabn_for_sites(seq_random: bool = True) -> MicroRNAAbundantBooleanNetwork:
    net_selection = select_network(
        20, 12, 25, 20, 30, 2000
    )  # -> [BooleanNetwork, list[BooleanNetwork], a_bn_index_of_maximum_length_cycle: int]  # maximum longest cycle length 45 can make for longer runtimes re perturbation transition matrix
    # for more_cycles in range(1000):
    #     net_selection[0].add_cycle_without_resample()
    net = _get_miR_abn_set_up(_get_abn_set_up(net_selection[0]))
    for node in net.get_plottable_nodes():
        if node.three_prime_utr is None:
            node.three_prime_utr = "".join(
                SystemRandom().choices(["A", "U", "G", "C"], k=500)
            )
    net.add_micro_rna_nodes(250)
    net.setup_target_site_lists()
    return net


def get_mir_abn_net(seq_random: bool = True) -> MicroRNAAbundantBooleanNetwork:
    net_selection = select_network(
        20, 12, 25, 20, 30, 2000
    )  # -> [BooleanNetwork, list[BooleanNetwork], a_bn_index_of_maximum_length_cycle: int]  # maximum longest cycle length 45 can make for longer runtimes re perturbation transition matrix
    for more_cycles in range(1000):
        net_selection[0].add_cycle_without_resample()
    net = _get_miR_abn_set_up(_get_abn_set_up(net_selection[0]))
    for node in net.get_plottable_nodes():
        if node.three_prime_utr is None:
            node.three_prime_utr = (
                "".join(SystemRandom().choices(["A", "U", "G", "C"], k=500))
                + get_rna_reverse_complement(
                    SystemRandom().choice(net.micro_rna_nodes).sequence[1:9]
                )
                + "".join(SystemRandom().choices(["A", "U", "G", "C"], k=20))
            )
    net.setup_target_site_lists()
    return net


def _get_abn_set_up(boolean_network: BooleanNetwork) -> AbundantBooleanNetwork:
    abundant_boolean_network = AbundantBooleanNetwork(
        1, [1], 2, 400, 0
    )  # these values are replaced
    abundant_boolean_network.from_boolean_network(
        boolean_network
    )  # with copy from net_selection
    abundant_boolean_network.add_abundant_nodes(15)
    abun_bn_cycle_length = abundant_boolean_network.longest_cycle_length()
    return abundant_boolean_network


# def random_nested_nodes(abn: AbundantBooleanNetwork, turn_off: bool =False) -> None:
#     for node in abn.abundant_nodes_2:
#         node.random_if_true = not turn_off
#         node.num_to_average_per_update = 64 if not turn_off else 1


def _get_miR_abn_set_up(
    abundant_boolean_network: AbundantBooleanNetwork,
    seq_random: bool = True,
    mirs_boolean: bool = False,
    num_mirs: int = 20,
) -> MicroRNAAbundantBooleanNetwork:
    a_bn_mir = MicroRNAAbundantBooleanNetwork(
        30, [setup_i for setup_i in range(1, 15)], 2, 400, 200
    )
    a_bn_mir.from_abundant_network(abundant_boolean_network)
    a_non_random_sequence = (
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "G"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "GG"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "GGG"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "GGGG"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "GGGGG"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "GGGGGG"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "GGGGGGG"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "GGGGGGGG"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "GGGGGGGGG"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "GGGGGGGGGG"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "UUUUUUUUUUU"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    )

    for abun_i, abun_node in zip(
        range(len(a_bn_mir.get_plottable_nodes())), a_bn_mir.get_plottable_nodes()
    ):
        if abun_i % 2 == 0 and not seq_random:
            abun_node.three_prime_utr = a_non_random_sequence
        else:
            abun_node.three_prime_utr = get_random_sequence(
                len(a_non_random_sequence), ["A", "U", "G", "C"]
            )

    mir_nodes = []
    for mir_i in range(num_mirs):
        mir_nodes.append(MicroRNANode())
        if mir_i % 2 == 0:
            mir_nodes[-1].sequence = (
                get_random_sequence(1, ["A", "U", "G", "C"])
                + "CCCCCC"
                + get_random_sequence(13, ["A", "U", "G", "C"])
            )
        else:
            # mir_nodes[-1].sequence = get_random_sequence(21, ["A", "U", "G", "C"])
            mir_nodes[-1].sequence = (
                get_random_sequence(1, ["A", "U", "G", "C"])
                + get_rna_reverse_complement(
                    (
                        node := SystemRandom().choice(a_bn_mir.get_plottable_nodes())
                    ).three_prime_utr[
                        (
                            i := SystemRandom().randrange(
                                0, len(node.three_prime_utr) - 6
                            )
                        ) : i
                        + 6
                    ]
                )
                + get_random_sequence(13, ["A", "U", "G", "C"])
            )
        mir_nodes[-1].random_setup(a_bn_mir.longest_cycle_length())
    a_bn_mir.add_micro_rna_nodes(mir_nodes)

    a_bn_mir.bn_collapsed_cycles.sort_cycle_records_by_cycle_length()
    cycle_length = len(a_bn_mir.bn_collapsed_cycles.cycle_records[0])
    mirs_cycle = a_bn_mir.bn_collapsed_cycles.cycle_records[0].cycle_states_list
    for each_mir_node in a_bn_mir.micro_rna_nodes:
        each_mir_node.state_schedule_index = 0
        if not mirs_boolean:
            each_mir_node.state_schedule = [False for mir_i in range(cycle_length)] + [
                True for mir_j in range(4 * cycle_length)
            ]
        else:
            node_index = SystemRandom().randrange(0, len(a_bn_mir))
            each_mir_node.state_schedule = [s[node_index] for s in mirs_cycle]
        each_mir_node.current_knockdown_rate = 0.99
        each_mir_node.base_knockdown_rate = 0.99

    return a_bn_mir
