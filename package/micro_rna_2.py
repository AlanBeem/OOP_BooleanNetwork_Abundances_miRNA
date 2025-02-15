# model of transfection would be cool to see
from package.bn_mir_helper_functions_V1 import *
from package.abundant_boolean_networks import *


# dinucleotide vectors, placeholder for e.g. data from McGeary 2019
dinuc_vectors = (dict(), dict())
for seq in get_all_sequences(2, nt):
    dinuc_vectors[0].update({seq: SystemRandom().random()})
    dinuc_vectors[1].update({seq: SystemRandom().random()})


class MicroRNANode(AbundantNode):
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3518396/pdf/nihms397915.pdf discusses 72 hour halflife of miRNA
    def __init__(self):
        super().__init__()
        self.state_schedule_index = None
        self.state_schedule = None
        self.base_transcription_rate = None
        self.knockdown_vector_N = None
        self.maximum_abundance = 0
        self.transcription_rate = 0
        self.base_transcription_rate = 0
        self.current_abundance = 0  # pri
        self.current_pre = 0
        self.current_mirmir = 0
        self.current_ago2_loaded = 0
        self.pri_to_pre_rate = 0
        self.pre_to_mirmir_rate = 0
        self.ago2_loading_rate = 0
        self.current_degradation_rate = 0
        self.current_pri_degradation_rate = 0
        self.current_pre_degradation_rate = 0
        self.current_mirmir_degradation_rate = 0
        self.current_ago_loaded_degradation_rate = 0
        # add perturbation of above by target directed degradation
        self.base_deg_rate = 0
        self.sequence = ""  # TODO strand selection (can make an accounting of relative gc content at each 5' end, or use NUPACK)
        self.pre_sequence = ""
        self.pri_sequence = ""
        self.current_knockdown_rate = 1
        self.base_knockdown_rate = 1
        # Snub: relation of binding regex to effects magnitude (weights defined over these, neural network) ### ML, regex as finite state machines, MC Boomer
        self.setup_default_1(10, 2500, 40)  # TODO dial this in, BN

    def setup_default_1(
        self,
        num_steps_to_maximum: int,
        maximum_abundance: int | float,
        num_steps_to_5: int,
    ):
        """setup_default_1 sets values as used in earlier code development in a simplified model of knockdown,
        and as completely made up regarding coupled 0.5, 0.6, 0.7 rates"""
        # For the future, with a name like this probably it should accept no arguments? maybe
        self.maximum_abundance = maximum_abundance
        self.current_abundance = 0
        self.current_pre = 0
        self.current_mirmir = 0
        self.current_ago2_loaded = 0
        self.transcription_rate = self.maximum_abundance / num_steps_to_maximum
        self.base_transcription_rate = self.transcription_rate
        self.base_deg_rate = math.pow(0.05, 1 / num_steps_to_5)
        self.current_degradation_rate = self.base_deg_rate
        self.pri_to_pre_rate = 0.5
        self.pre_to_mirmir_rate = 0.6  # ?
        self.ago2_loading_rate = (
            0.7  # TODO What are these rates, and how do they relate to sequence?
        )
        self.base_knockdown_rate = (
            0.67  # see page 11 Bartel 2009 / 2013: 0.67 as a per site knockdown
        )
        self.current_knockdown_rate = self.base_knockdown_rate

    def __str__(self):
        return (
            "MicroRNA node:\nsequence: "
            + self.sequence
            + "\nmaximum_abundance: "
            + str(self.maximum_abundance)
            + "\ncurrent_abundance: "
            + str(self.current_abundance)
            + "\ncurrent_pre: "
            + str(self.current_pre)
            + "\ncurrent_mirmir: "
            + str(self.current_mirmir)
            + "\ncurrent_ago2_loaded: "
            + str(self.current_ago2_loaded)
            + "\ntranscription_rate: "
            + str(self.transcription_rate)
            + "\nbase_deg_rate: "
            + str(self.base_deg_rate)
            + "\ncurrent_degradation_rate: "
            + str(self.current_degradation_rate)
            + "\npri_to_pre_rate: "
            + str(self.pri_to_pre_rate)
            + "\npre_to_mirmir_rate: "
            + str(self.pre_to_mirmir_rate)
            + "\nago2_loading_rate: "
            + str(self.ago2_loading_rate)
            + "\nbase_knockdown_rate: "
            + str(self.base_knockdown_rate)
            + "\ncurrent_knockdown_rate: "
            + str(self.current_knockdown_rate)
        )

    def update(self, state: bool):  # ODE's, for ~Euler's method
        if self.state_schedule is not None:
            state = self.state_schedule[
                self.state_schedule_index % len(self.state_schedule)
            ]
            self.state_schedule_index += 1
        self.current_ago2_loaded = (
            self.current_ago2_loaded * self.current_degradation_rate
            + self.current_mirmir * self.ago2_loading_rate
        )
        self.current_mirmir = (
            (self.current_mirmir - (self.current_mirmir * self.ago2_loading_rate))
            * self.current_degradation_rate
            + self.current_pre * self.pre_to_mirmir_rate
        )
        self.current_pre = (
            (self.current_pre - (self.current_pre * self.pre_to_mirmir_rate))
            * self.current_degradation_rate
            + self.current_abundance * self.pri_to_pre_rate
        )
        self.current_abundance = (
            self.current_abundance - (self.current_abundance * self.pri_to_pre_rate)
        ) * self.current_degradation_rate + int(state) * self.transcription_rate
        # TODO Make each of these a function of secondary structure and primary structure and represent each molecule

    def reset_node(self):
        self.current_knockdown_rate = self.base_knockdown_rate
        self.transcription_rate = self.base_transcription_rate
        self.current_ago2_loaded = 0
        self.current_mirmir = 0
        self.current_pre = 0
        self.current_abundance = 0


class ThreePrimeUTRTargetSite:
    """ThreePrimeUTRTargetSite is consistent with the assignment of target sites with a hierarchy of binding regions by
    length of contiguous bound subsequence, like in TargetScan. Contiguous seed pairing is calculated at evaluation or
    as setup logic in __init__."""

    # TODO define *= operation (dunder) and make product sum
    def __init__(
        self,
        start_index: int,
        end_index: int,
        mir_node_ref: MicroRNANode,
        abundant_node_ref: AbundantNode,
    ):
        self.start_index = start_index  # 5' to 3' on regulated RNA
        self.end_index = end_index  # 5' to 3' on regulated RNA
        # self.binding_nt = self.end_index - self.start_index + 1
        self.micro_rna_node_reference = mir_node_ref
        self.abundant_node_reference = abundant_node_ref
        self.upstream_distance = None  # To be filled centrally
        self.downstream_distance = None
        # adjust for contiguous matching nucleotides
        nt_comp = {"A": "U", "U": "A", "G": "C", "C": "G"}  # dictionary
        i = 7  # starts as a 6nt match  # miR: 0 [1 2 3 4 5 6] 7 ...
        while (
            self.start_index - 1 >= 0
            and abundant_node_ref.three_prime_utr[self.start_index - 1]
            == nt_comp[mir_node_ref.sequence[i]]
        ):
            i += 1
            self.start_index -= 1
        self.binding_nt = self.end_index - self.start_index + 1

        # assign site type  # "X-mer" notation: X = number of interacting target nt (A1 counts as an interaction)  (TODO full scheme seen in McGeary 2019)
        # as a binary encoding  # https://www.targetscan.org/docs/7mer.html TargetScan predicts 8mer, 7mer-m8, and 7mer-A1 (8mer = 7mer-m8-A1, or, I'd argue it should be written as 6mer-m8-A1)
        self.site_type = bytearray()  # [A1, >6, >7, ]
        # A1
        if (
            self.end_index == len(self.abundant_node_reference.three_prime_utr) - 1
            or self.abundant_node_reference.three_prime_utr[self.end_index + 1] == "A"
        ):
            self.site_type.append(True)
        else:
            self.site_type.append(False)
        # binding nt > 6
        if self.binding_nt > 6:
            self.site_type.append(True)
        else:
            self.site_type.append(False)
        # binding nt > 7
        if self.binding_nt > 7:
            self.site_type.append(True)
        else:
            self.site_type.append(False)

        # encoding:
        #           A1  nt>6    nt>7
        #       T
        #       F

        #                                     # could similarly encode local motifs, and site features, ...
        if all([not b for b in self.site_type]):  # [A1, >6, >7, ] = self.site_type
            type_string = "6mer"
        elif self.site_type[0] and not self.site_type[1]:
            type_string = "7mer-A1"
        elif not self.site_type[0] and self.site_type[1]:
            type_string = "7mer-m8"
        elif self.site_type[0] and self.site_type[1]:
            type_string = "8mer"
        self.type_string = type_string

    def __str__(self):
        disp_mir = "".join(
            reversed(self.micro_rna_node_reference.sequence[1 : self.binding_nt + 1])
        )
        disp_mir = f"3'...{disp_mir}.-5'"  # see size of sets of seeds at N nt
        seq_window_half = 20
        m = self.start_index
        #                3'-...NNNNNN.-5'
        # '5'-...NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN...-3'      site type: {site type}
        #                      |{start index}
        disp_mrna = f"5'...{self.abundant_node_reference.three_prime_utr[max(0, m - seq_window_half):
                                                                         min(len(self.abundant_node_reference.three_prime_utr), m + seq_window_half)]}...3'"
        # outstring = ''
        # outstring += f"miR:  {' ' * (seq_window_half)}{disp_mir}\n"
        # #                                             3'...
        # outstring += f"mRNA: {disp_mrna}" + '\n'
        # #                    5'...
        # outstring += f"           {' ' * seq_window_half}|{m}" + '\n'
        # return outstring

        outstring = []
        outstring.append(f"miR:  {' ' * (seq_window_half)}{disp_mir}")
        #                                             3'...
        outstring.append(f"mRNA: {disp_mrna}\t\tsite type: {self.type_string}")
        #                    5'...
        outstring.append(f"           {' ' * seq_window_half}|{m}")

        return "\n".join(outstring)

    def __float__(self):
        return three_prime_utr_target_site_to_scalar(self)

    # def __hash__(self):


class ORFTargetSite(ThreePrimeUTRTargetSite):
    """Same as 3' UTR, but intended to be generated only for ORF sites"""

    def __init__(
        self,
        start_index: int,
        end_index: int,
        mir_node_ref: MicroRNANode,
        abundant_node_ref: AbundantNode,
    ):
        super().__init__(start_index, end_index, mir_node_ref, abundant_node_ref)


class FivePrimeUTRTargetSite(ThreePrimeUTRTargetSite):
    """Same as 3' UTR, but intended to be generated only for 5' UTR sites (which are treated as less likely to be e
    efficacious at a given moment than ORF sites)"""

    def __init__(
        self,
        start_index: int,
        end_index: int,
        mir_node_ref: MicroRNANode,
        abundant_node_ref: AbundantNode,
    ):
        super().__init__(start_index, end_index, mir_node_ref, abundant_node_ref)


def get_three_prime_utr_target_sites(
    mirna_nodes: list[MicroRNANode], abundant_node: AbundantNode
) -> list[ThreePrimeUTRTargetSite]:
    """intended for batch use where all miRNA sequence nodes are known, also assigns site distances."""
    # 5' to 3' on target sequence, for each nt: for each miR | reverse complementary interaction:
    # deposit a TargetSite
    # ribosome_shadow = 15  # nt (index==14)
    # a_sequence = abundant_node.three_prime_utr[ribosome_shadow:len(abundant_node.three_prime_utr)]
    sequences_to_match = [
        mirna_nodes[sm_i].sequence[1:7] for sm_i in range(len(mirna_nodes))
    ]  # 6nt: 1 2 3 4 5 6, for all miRNA nodes
    target_site_list = []
    for start_index in range(15, len(abundant_node.three_prime_utr) - 6, 1):
        rel_end_index = start_index + 5  # used as an inclusive bound
        # sub_seq = a_sequence[start_index:start_index + len(sequences_to_match[0])]
        sub_seq = abundant_node.three_prime_utr[start_index : rel_end_index + 1]
        # give miRs a specific "seed(s)" field, and evaluate strand preference as a function of localization,
        # think stoichiometry and condensates
        for mir_i, each_seq in zip(range(len(mirna_nodes)), sequences_to_match):
            # reverse and complement must be applied to sequence comparison
            # if each_seq.startswith(get_rna_reverse_complement(sub_seq)):
            if each_seq == get_rna_reverse_complement(sub_seq):
                target_site_list.append(
                    ThreePrimeUTRTargetSite(
                        start_index, rel_end_index, mirna_nodes[mir_i], abundant_node
                    )
                )  # ...TargetSite construction adjusts start_index to represent larger matches
    _assign_site_distances(
        target_site_list
    )  # This modifies target sites to represent intersite distance
    return target_site_list


def _assign_site_distances(
    target_site_list: list[
        ThreePrimeUTRTargetSite | ORFTargetSite | FivePrimeUTRTargetSite
    ],
) -> None:
    """Sorts target_sites, and modifies elements of target_sites."""
    target_site_list.sort(key=lambda p: p.start_index)
    if len(target_site_list) > 1:
        if len(target_site_list) == 2:
            target_site_list[0].downstream_distance = abs(
                target_site_list[0].end_index - target_site_list[1].start_index
            )
            target_site_list[-1].upstream_distance = abs(
                target_site_list[-2].end_index - target_site_list[-1].start_index
            )
        else:
            target_site_list[0].downstream_distance = abs(
                target_site_list[0].end_index - target_site_list[1].start_index
            )  # fencepost
            #
            for i in range(
                1, len(target_site_list) - 1
            ):  # assign distances to non-fence-post portion of target site list
                target_site_list[i].upstream_distance = abs(
                    target_site_list[i - 1].end_index - target_site_list[i].start_index
                )
                target_site_list[i].downstream_distance = abs(
                    target_site_list[i].end_index - target_site_list[i + 1].start_index
                )
            #
            target_site_list[-1].upstream_distance = abs(
                target_site_list[-2].end_index - target_site_list[-1].start_index
            )  # fencepost


def get_orf_target_sites(
    mirna_nodes: list[MicroRNANode],
    abundant_node: AbundantNode,
    basepair_bound_inclusive: int,
):
    temp_return_orf_sites = []
    temp_3_prime_sites = get_three_prime_utr_target_sites(mirna_nodes, abundant_node)
    for orf_i in range(len(temp_3_prime_sites)):
        if (
            temp_3_prime_sites[orf_i].end_index - temp_3_prime_sites[orf_i].start_index
            >= basepair_bound_inclusive
        ):
            temp_return_orf_sites.append(temp_3_prime_sites[orf_i])
    return temp_return_orf_sites


def get_five_prime_target_sites(
    mirna_nodes: list[MicroRNANode],
    abundant_node: AbundantNode,
    basepair_bound_inclusive: int,
):
    temp_return_five_sites = []
    temp_3_prime_sites = get_three_prime_utr_target_sites(mirna_nodes, abundant_node)
    for orf_i in range(len(temp_3_prime_sites)):
        if (
            temp_3_prime_sites[orf_i].end_index - temp_3_prime_sites[orf_i].start_index
            >= basepair_bound_inclusive
        ):
            temp_return_five_sites.append(temp_3_prime_sites[orf_i])
    return temp_return_five_sites


local_affinity_all_seqs_lists = []
# could calculate a space accounting also for the differential effects of expression of any probable-to-express genes
for laasl_i in range(1, 10):
    local_affinity_all_seqs_lists.append(
        get_all_sequences(laasl_i, ["A", "U", "G", "C"])
    )


def three_prime_utr_target_site_to_scalar(
    target_site_to_scalar: ThreePrimeUTRTargetSite,
):
    # this should ? be fit per miRNA, mRNA # was first implemented as returning a scalar = f(intermediate_exponent),
    # TODO try letting this define an activation function
    """this modifies the values of references of target site or evaluates each TargetSite to a scalar based on current expression values, or ."""
    # TODO something similar as a method of TargetSite # available abundance in a shuffled order  # add target directed miRNA degradation
    # # do this like Factory (?) where is set up a lambda expression taking the miR abundance and target abundance and returning a scalar
    intermediate_scalar = 1
    this_abundance = target_site_to_scalar.abundant_node_reference.current_abundance
    mirna_abundance = target_site_to_scalar.micro_rna_node_reference.current_ago2_loaded
    targeted_abundance = 0  # 100  # 0  # TODO start this at some "background" number, e.g. location specific (LocalizedCell)
    for each_site in target_site_to_scalar.micro_rna_node_reference.target_sites:
        targeted_abundance += each_site.abundant_node_reference.current_abundance
    if targeted_abundance == 0:
        targeted_abundance = 1
    # print(f"this_abundance = {target_site_to_scalar.abundant_node_reference.current_abundance}")
    # print(f"mirna_abundance = {target_site_to_scalar.micro_rna_node_reference.current_ago2_loaded}")
    if (
        this_abundance >= 1 and mirna_abundance >= 1
    ):  # else returns f(intermediate exponent)
        # if targeted_abundance >= 1:
        #     intermediate_scalar = mirna_abundance / targeted_abundance
        if (
            target_site_to_scalar.end_index - target_site_to_scalar.start_index + 1 > 17
        ):  # TODO make threshold a field in miRNA, RNA
            # subtract?                                                                   # also make the cleavability a function of location
            target_site_to_scalar.abundant_node_reference.set_abundance(
                target_site_to_scalar.abundant_node_reference.current_abundance
                - (
                    1
                    - target_site_to_scalar.abundant_node_reference.current_abundance
                    / target_site_to_scalar.abundant_node_reference.maximum_abundance
                )
                * (
                    1
                    - target_site_to_scalar.micro_rna_node_reference.current_ago2_loaded
                    / targeted_abundance
                )
                * target_site_to_scalar.abundant_node_reference.current_abundance
                * 100
            )  # TODO vary this number
        # ceRNA (competing endogenous RNA hypothesis)
        # ## This is handled by returning (scalar)**(target abundance / targeted abundance), which might be too strong
        # Localization

        # 2nt 5' and 2nt 3',
        # Re McGeary 2019: two 16 length vectors (dictionaries) of e.g. {'AU':0.5}  # placeholder (subsumed) more variance than these cause TODO
        # could vary this (all these calculations) by region in RNA and cell and state spaces of constituent molecules, also
        nt_5 = target_site_to_scalar.abundant_node_reference.three_prime_utr[
            target_site_to_scalar.start_index - 2 : target_site_to_scalar.start_index
        ]
        intermediate_scalar *= 1 - dinuc_vectors[0].get(nt_5, 1) / 5  # / 10
        # print(f"* {nt_5} == {intermediate_scalar}")
        nt_3 = target_site_to_scalar.abundant_node_reference.three_prime_utr[
            target_site_to_scalar.end_index
            + 1 : target_site_to_scalar.end_index
            + 2
            + 1
        ]
        intermediate_scalar *= 1 - dinuc_vectors[0].get(nt_3, 1) / 5  #  / 10
        # print(f"* {nt_3} == {intermediate_scalar}")

        # Site Type: Bartel 2009, Figure 4
        # 8mer:= (6mer + m8 + A1, total binding nucleotides: 7)
        # A1 ?
        if (
            target_site_to_scalar.end_index + 1
            < len(target_site_to_scalar.abundant_node_reference.three_prime_utr)
            and target_site_to_scalar.abundant_node_reference.three_prime_utr[
                target_site_to_scalar.end_index + 1
            ]
            == "A"
        ):
            # intermediate_scalar = math.pow(intermediate_scalar, 1/2)
            intermediate_scalar *= 1 / 2
        # print(f"* A1 == {intermediate_scalar}")
        # m8 ?
        if target_site_to_scalar.end_index - target_site_to_scalar.start_index + 1 > 6:
            # intermediate_scalar = math.pow(intermediate_scalar, 1/3)
            intermediate_scalar *= 2 / 5
        # print(f"* m8 == {intermediate_scalar}")

        # productive 3' end pairing
        i = 0
        j = 0
        possible_nt_mrna = (
            target_site_to_scalar.abundant_node_reference.three_prime_utr[
                max(
                    0,
                    target_site_to_scalar.start_index
                    - (
                        len(target_site_to_scalar.micro_rna_node_reference.sequence) + 5
                    ),
                ) : target_site_to_scalar.start_index
            ]
        )
        possible_nt_mrna = list(
            reversed(possible_nt_mrna)
        )  #                                                 ### m.n. 1
        possible_nt_mir = target_site_to_scalar.micro_rna_node_reference.sequence[
            1
            + target_site_to_scalar.end_index
            - target_site_to_scalar.start_index
            + 1 :
        ]  # maybe not all nt of sequence- but imagine it has 1 more nt
        possible_nt_mir = get_rna_reverse_complement(
            possible_nt_mir
        )  # this should be calculated in TargetSite setup TODO
        alignments = []
        for j in range(3):
            # for i in range(len(possible_nt_mrna)):  # simple gapped alignment (only gapped on mRNA) for 4-6 intervening nucleotides on miRNA
            i = 0  #         #
            j = (
                j + 4
            ) - 1  # indexes                                                                  ### m.n. 2, relates to m.n. 1
            alignments.append(0)
            while i < len(possible_nt_mrna) and j < len(possible_nt_mir):
                if possible_nt_mir[j] == possible_nt_mrna[i]:
                    alignments[-1] += 1
                    i += 1
                    j += 1
                else:
                    i += 1
                if alignments[-1] > 3:  # TODO look up
                    intermediate_scalar *= 8 / 11
                    break
        # if prod_bool:
        #     intermediate_scalar *= 3/2
        # print(f"* 3' pairing == {intermediate_scalar}")

        # site distance
        if target_site_to_scalar.upstream_distance is not None:
            if (
                target_site_to_scalar.upstream_distance <= 40
            ):  # TODO add penalty for "too close" sites
                intermediate_scalar *= 1.2
        if target_site_to_scalar.downstream_distance is not None:
            if target_site_to_scalar.downstream_distance <= 40:
                intermediate_scalar *= 1.2
                # print(target_site_to_scalar.downstream_distance)
        # print(f"* site distance == {intermediate_scalar}")

        # Site Position: Bartel 2009, Figure 4
        position = (
            (target_site_to_scalar.end_index + target_site_to_scalar.start_index) / 2
        ) / len(
            target_site_to_scalar.abundant_node_reference.three_prime_utr
        )  # normalized to 1
        mu = 1
        sig = 0.33
        scale = 1
        cent_pen = math.e ** (
            (-0.5) * ((position - mu) / sig) ** 2
        )  # (thought to be general occlusion of RNA further from the end of the 3'-UTR)
        # for i in range(10):
        # print(f"f({i/10}) -> {math.e**((-0.5) * ((i/10 - mu) / sig) ** 2)}")
        intermediate_scalar *= min(1 - scale * cent_pen, 1)
        # print(f"* site position == {intermediate_scalar}")

        # site accessibility
        if (
            target_site_to_scalar.abundant_node_reference.three_prime_utr_accessibility
            is not None
        ):
            intermediate_scalar *= target_site_to_scalar.abundant_node_reference.three_prime_utr_accessibility[
                int(
                    position
                    * len(target_site_to_scalar.abundant_node_reference.three_prime_utr)
                )
            ]
        # consider: applying a bunch of random hexamers, or ... ssRNAse in a cell while you lyse it, stopping the reaction
        # print(f"* site accessibility (should be the same) == {intermediate_scalar}")

        # Local AU Content (proxy for site accessibility): Bartel 2009, Figure 4
        # Gaussian for weights centered at site (make this scale with UTR size)  # replace with min(distance from start, end)
        ud_nt = 80
        au_up_down_nt = [
            target_site_to_scalar.start_index,
            target_site_to_scalar.end_index,
        ]  # was 40
        while (
            target_site_to_scalar.start_index - au_up_down_nt[0] < ud_nt
            and au_up_down_nt[0] > 0
        ):  # could count the A's here and get that 3' end is more effective
            au_up_down_nt[0] -= 1
        while (
            au_up_down_nt[1] - target_site_to_scalar.start_index < ud_nt
            and au_up_down_nt[0] > 0
        ):  # could count the A's here and get that 3' end is more effective
            au_up_down_nt[1] += 1
        au_seq_5 = target_site_to_scalar.abundant_node_reference.three_prime_utr[
            au_up_down_nt[0] : target_site_to_scalar.start_index
        ]
        au_seq_3 = target_site_to_scalar.abundant_node_reference.three_prime_utr[
            target_site_to_scalar.end_index + 1 : au_up_down_nt[1]
        ]
        # Gaussian
        mu_2 = 0
        sig_2 = 0.3
        d_nt = 1 / ud_nt  # ##
        au_local_weight = [
            (1 - x_i**2)
            * (1 / (sig_2 * math.sqrt(2 * math.pi)))
            * math.exp(-(1 / 2) * ((x_i - mu_2) / sig_2) ** 2)
            for x_i in [d_nt * x_j for x_j in range(ud_nt)]
        ]  # ## per nt
        #
        au_weighted_counts = [[0, 0], [0, 0]]
        for au_i in range(len(au_seq_5)):
            au_weighted_counts[0][0] += au_local_weight[au_i] * int(
                au_seq_5[au_i] == "A" or au_seq_5[au_i] == "U"
            )
            au_weighted_counts[0][1] += au_local_weight[au_i]
        for au_k in range(len(au_seq_3)):
            au_weighted_counts[1][0] += au_local_weight[au_k] * int(
                au_seq_3[au_k] == "A" or au_seq_3[au_k] == "U"
            )
            au_weighted_counts[1][1] += au_local_weight[au_k]
        if au_weighted_counts[0][1] > 0:
            intermediate_scalar *= (
                au_weighted_counts[0][1] - au_weighted_counts[0][0]
            ) / au_weighted_counts[0][1]
        if au_weighted_counts[1][1] > 0:
            intermediate_scalar *= (
                au_weighted_counts[1][1] - au_weighted_counts[1][0]
            ) / au_weighted_counts[1][1]
        #
        # add more features #
        #
    scalar = (intermediate_scalar) ** (
        target_site_to_scalar.abundant_node_reference.current_abundance
        / targeted_abundance
    )  # TODO what really makes sense here (physics?)
    # return (scalar)**(1/8)  ### and how to scale by miRNA abundance? -> model competitive Ago2 loading TODO
    return scalar  # **(target_site_to_scalar.micro_rna_node_reference.current_ago2_loaded / target_site_to_scalar.micro_rna_node_reference.maximum_abundance)


def utr_3_target_site_list_to_scalar(target_site_list: list[ThreePrimeUTRTargetSite]):
    if target_site_list is None or len(target_site_list) == 0:
        return 1
    else:
        temp_utr_3_list_scalar = 1
        for each_target_site in target_site_list:
            temp_utr_3_list_scalar = (
                temp_utr_3_list_scalar
                * three_prime_utr_target_site_to_scalar(each_target_site)
            )
        return temp_utr_3_list_scalar


#


#
# def orf_target_site_to_scalar(each_target_site: ORFTargetSite):
# if each_target_site.micro_rna_node_reference.current_ago2_loaded > 150 and binding of more than N nt:  # TODO figure this out
# return 0.25  # TODO how unrealistic is this?


# def orf_target_site_list_to_scalar(target_site_list: list[ORFTargetSite]):
#     if len(target_site_list) == 0:
#         return 1
#     else:
#         temp_orf_list_scalar = 1
#         for each_target_site in target_site_list:
#             temp_orf_list_scalar = temp_orf_list_scalar * orf_target_site_to_scalar(each_target_site)
#         return temp_orf_list_scalar
# #


#
# def five_prime_target_site_to_scalar(each_target_site):
#     if each_target_site.micro_rna_node_reference.current_ago2_loaded > 175 and binding greater than N nt:  # TODO figure this out
#         return 0.33  # TODO how unrealistic is this?


# def five_prime_target_site_list_to_scalar(target_site_list: list[ThreePrimeUTRTargetSite]):
#     if len(target_site_list) == 0:
#         return 1
#     else:
#         temp_five_prime_list_scalar = 1
#         for each_target_site in target_site_list:
#             temp_five_prime_list_scalar = temp_five_prime_list_scalar * five_prime_target_site_to_scalar(each_target_site)
#         return temp_five_prime_list_scalar
# #

# TODO McGeary 2019 says AGO-RBNS recovers sites with less binding nucleotides than canonical re Bartel 2009
# Figure out minimal sequence determinants (over some number of possible models, see MC Boomer) for site type
# differences (is it worth modeling this predictively?)
