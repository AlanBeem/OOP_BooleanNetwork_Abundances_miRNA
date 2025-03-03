from package.boolean_networks import BooleanNetwork
import matplotlib.pyplot as plt

# from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from random import SystemRandom as rand
import copy
from package.plotting import get_colors
import numpy as np


def control_report(
    net: BooleanNetwork,
    cycle_colors: list = [],
    total_steps: int = 100,
    progress_div: int = 1,
    with_noise: str | float | None = None,
    frequency_stacked: bool = False,
    show_legend: bool = True,
    goal_cycle_index: int = -1,
    goal_dfa: bool = False
):

    if cycle_colors == []:
        cycle_colors = get_colors(len(net.bn_collapsed_cycles.cycle_records), True)

    cycle_colors = [
        ([cc[0], cc[1], cc[2], cc[3]])
        for cc, i in zip(cycle_colors, range(len(cycle_colors)))
        if i < len(net.bn_collapsed_cycles.cycle_records)
    ]
    cycle_colors.append([1, 1, 1, 1])

    cycle_labels = ["cycle " + str(i + 1) for i in range(len(cycle_colors))]
    cycle_labels[-1] = "None"
    if goal_cycle_index != -1:
        cycle_labels[goal_cycle_index] = f"({cycle_labels[goal_cycle_index]})"

    patches = []
    for i in range(len(cycle_labels)):
        patches.append(mpatches.Patch(color=cycle_colors[i], label=cycle_labels[i]))

    if show_legend:
        fig_leg, ax = plt.subplots()
        fig_leg.set_size_inches(10, 1)
        ax.set_axis_off()
        ax.legend(handles=patches, mode="expand", ncols=6)
        plt.show()

    # functions setup
    # initial conditions: 1 of each cycle state
    def get_ic_cycles(net: BooleanNetwork) -> list:
        conditions = []
        for i in range(len(net.bn_collapsed_cycles.cycle_records)):
            for c in net.bn_collapsed_cycles.cycle_records[i].cycle_states_list:
                conditions.append([copy.deepcopy(c)])
        return conditions

    # advance state
    def next_state(state, functions, inputs, perturb: int | None = None):
        if perturb is None:
            # print(f"args:\nstate: {state}\nfunctions: {functions}\ninputs: {inputs}, perturb: {perturb}")
            return bytearray(
                functions[i](bool(state[inputs[i][0]]), bool(state[inputs[i][1]]))
                for i in range(len(functions))
            )
        else:
            return bytearray(
                (
                    (
                        not functions[i](  # then perturbs...
                            bool(state[inputs[i][0]]), bool(state[inputs[i][1]])
                        )
                    )
                    if i == perturb  # ...if to perturb
                    else functions[i](
                        bool(state[inputs[i][0]]), bool(state[inputs[i][1]])
                    )
                )
                for i in range(len(functions))
            )

    # setup parallel network states
    functions = [net.nodes[i].function.get_boolean for i in range(len(net.nodes))]
    inputs = net.node_inputs_assignments

    conditions = get_ic_cycles(net)  # 1 per each cycle state

    progress_rows = []
    if frequency_stacked:
        stacked_rows = []

    for step in range(total_steps):
        for i in range(len(conditions)):
            #
            # step conditions[i][-1]
            conditions[i].append(next_state(conditions[i][-1], functions, inputs))
            #
            if not (net.bn_collapsed_cycles.get_index(conditions[i][-1]) == goal_cycle_index and goal_dfa):
                if (
                    with_noise is not None
                    and not isinstance(with_noise, str)
                    and rand().random() <= with_noise
                ):
                    perturb_node = rand().randrange(0, len(net))
                    conditions[i][-1][perturb_node] = not bool(
                        conditions[i][-1][perturb_node]
                    )
                elif (
                    with_noise is not None
                    and isinstance(with_noise, str)
                    and with_noise == "bv"
                ):
                    prev_step_bv = sum(
                        int(conditions[i][-2][k] != conditions[i][-1][k])
                        for k in range(len(conditions[i][-1]))
                    ) / len(net)
                    if rand().random() * (1 - prev_step_bv) < 0.05:
                        perturb_node = rand().randrange(0, len(net))
                        conditions[i][-1][perturb_node] = not bool(
                            conditions[i][-1][perturb_node]
                        )
                elif (
                    with_noise is not None
                    and isinstance(with_noise, str)
                    and with_noise == "inverse bv"
                ):
                    prev_step_bv = sum(
                        int(conditions[i][-2][k] != conditions[i][-1][k])
                        for k in range(len(conditions[i][-1]))
                    ) / len(net)
                    if rand().random() < 0.05 * (1 - prev_step_bv):
                        perturb_node = rand().randrange(0, len(net))
                        conditions[i][-1][perturb_node] = not bool(
                            conditions[i][-1][perturb_node]
                        )

        if step % progress_div == 0:
            progress_rows.append(
                [
                    cycle_colors[  # update with most recent (or smooth by longest cycle?)
                        (
                            -1
                            if net.bn_collapsed_cycles.get_index(c[-1]) is None
                            else net.bn_collapsed_cycles.get_index(c[-1])
                        )
                    ]
                    for c in conditions
                ]
            )
            if frequency_stacked:
                stacked_row = []
                for n in range(-1, len(net.bn_collapsed_cycles.cycle_records)):
                    for _ in range(progress_rows[-1].count(cycle_colors[n])):
                        stacked_row.append(cycle_colors[n])
                stacked_rows.append(stacked_row)

    if not frequency_stacked:
        fig = plt.figure()
        plt.imshow(progress_rows, aspect=1 / 8)
        #
        return fig
        #
    else:
        fig1 = plt.figure()
        # plt.imshow(progress_rows, cmap=cmap)
        # from: https://stackoverflow.com/a/10970122/25597680, aspect
        plt.imshow(progress_rows, aspect=1 / 8)
        fig2 = plt.figure()
        plt.imshow(stacked_rows, aspect=1 / 8)
        #
        return fig1, fig2
        #


def seq_pert_report(
    p1,
    p2,
    goal_cycle_index,
    net: BooleanNetwork,
    cycle_colors: list = [],
    total_steps: int = 100,
    progress_div: int = 1,
    goal_bool: bool = False,
    p3: int | None = None,
    with_noise: str | float | None = None,
    frequency_stacked: bool = False,
    show_legend: bool = True,
):

    if cycle_colors == []:
        cycle_colors = get_colors(len(net.bn_collapsed_cycles.cycle_records), True)

    # c_colors = [cycle_colors[i] for i in range(len(net.bn_collapsed_cycles.cycle_records))]
    # cmap = ListedColormap(c_colors)

    goal_states = set(
        int("0b" + "".join(str(int(s[i])) for i in range(len(s))), base=2)
        for s in net.bn_collapsed_cycles.cycle_records[
            goal_cycle_index
        ].cycle_states_list
    )

    # lower alpha for non-goal states
    cycle_colors = [
        (
            # [cc[0], cc[1], cc[2], 0.5]
            # if i != goal_cycle_index
            # else [cc[0], cc[1], cc[2], cc[3]]
            [cc[0], cc[1], cc[2], cc[3]]
        )
        for cc, i in zip(cycle_colors, range(len(cycle_colors)))
        if i < len(net.bn_collapsed_cycles.cycle_records)
    ]

    # white for states | cycle index = None
    cycle_colors.append([1, 1, 1, 1])

    cycle_labels = ["cycle " + str(i + 1) for i in range(len(cycle_colors))]
    cycle_labels[-1] = "None"

    cycle_labels[goal_cycle_index] = f"({cycle_labels[goal_cycle_index]})"

    patches = []
    for i in range(len(cycle_labels)):
        patches.append(mpatches.Patch(color=cycle_colors[i], label=cycle_labels[i]))

    if show_legend:
        print("goal cycle is in parentheses")
        fig_leg, ax = plt.subplots()
        fig_leg.set_size_inches(10, 1)
        ax.set_axis_off()
        ax.legend(handles=patches, mode="expand", ncols=6)
        # plt.rc('text', usetex=True)
        # for text in ax.legend().get_texts():
        #     if text.get_text() == f"cycle {goal_cycle_index + 1}":
        #         text.set_text("(" + text.get_text() + ")")
        #         # text.set_text("$\underline\{" + text.get_text() + "\}$")
        #         break
        plt.show()

    # functions setup
    # initial conditions: 1 of each cycle state
    def get_ic_cycles(net: BooleanNetwork) -> list:
        conditions = []
        for i in range(len(net.bn_collapsed_cycles.cycle_records)):
            for c in net.bn_collapsed_cycles.cycle_records[i].cycle_states_list:
                conditions.append([copy.deepcopy(c)])
        return conditions

    # perturbation sequences, given p1, p2, vary L
    def get_perturb_tuples(
        p1, p2, net: BooleanNetwork, p3: int | None = None
    ) -> list[tuple]:
        p_p_L = []
        if p3 is None:
            for L in range(
                1, net.longest_cycle_length()
            ):  # could put different distributions here
                p_p_L.append((p1, p2, L))
        else:
            for L in range(
                1, net.longest_cycle_length()
            ):  # could put different distributions here
                p_p_L.append((p1, p2, L, p3))
        return p_p_L

    # advance state
    def next_state(state, functions, inputs, perturb: int | None = None):
        if perturb is None:
            # print(f"args:\nstate: {state}\nfunctions: {functions}\ninputs: {inputs}, perturb: {perturb}")
            return bytearray(
                functions[i](bool(state[inputs[i][0]]), bool(state[inputs[i][1]]))
                for i in range(len(functions))
            )
        else:
            return bytearray(
                (
                    (
                        not functions[i](
                            bool(state[inputs[i][0]]), bool(state[inputs[i][1]])
                        )
                    )
                    if i == perturb
                    else functions[i](
                        bool(state[inputs[i][0]]), bool(state[inputs[i][1]])
                    )
                )
                for i in range(len(functions))
            )

    # setup parallel network states
    functions = [net.nodes[i].function.get_boolean for i in range(len(net.nodes))]
    inputs = net.node_inputs_assignments
    conditions = get_ic_cycles(net)  # 1 per each cycle state
    perturb_tups = get_perturb_tuples(p1, p2, net)
    el_counters = [0 for _ in conditions]  # count intervals between p1, p2
    perturb_occurring = [
        False for _ in conditions
    ]  # sieve application of perturbation sequences
    perturb_selectors = [
        rand().randrange(len(perturb_tups)) for _ in conditions
    ]  # index reference to perturb tups

    # for output
    progress_rows = []
    if frequency_stacked:
        stacked_rows = []

    for step in range(total_steps):
        # sum(int(state[-k]) * 2**k for k in range(1, len(state) + 1))
        for i in range(len(conditions)):
            if (
                goal_bool
                and int("0b" + "".join(str(int(s)) for s in conditions[i][-1]), base=2)
                in goal_states
            ):
                conditions[i].append(
                    conditions[i][-1]
                )  # typo: -2, hmm, could go back a ways (but not using a single transition matrix)
            else:
                if perturb_occurring[i]:
                    el_counters[i] += 1
                    if (
                        el_counters[i] == perturb_tups[perturb_selectors[i]][2]
                    ):  # L: interval, ex: 1 step since start (p1), for L=1, apply p2
                        conditions[i].append(
                            next_state(
                                conditions[i][-1],
                                functions,
                                inputs,
                                perturb_tups[perturb_selectors[i]][1],
                            )
                        )
                        if len(perturb_tups[0]) == 3:
                            perturb_occurring[i] = False
                            perturb_selectors[i] = rand().randrange(
                                len(perturb_tups)
                            )  # can add complexity here
                            el_counters[i] = 0
                    elif (
                        len(perturb_tups[0]) == 4
                        and el_counters[i] == 2 * perturb_tups[perturb_selectors[i]][2]
                    ):  # L: interval, ex: 1 step since start (p1), for L=1, apply p2
                        conditions[i].append(
                            next_state(
                                conditions[i][-1],
                                functions,
                                inputs,
                                perturb_tups[perturb_selectors[i]][3],
                            )
                        )
                        perturb_occurring[i] = False
                        perturb_selectors[i] = rand().randrange(len(perturb_tups))
                        el_counters[i] = 0
                    else:
                        conditions[i].append(
                            next_state(conditions[i][-1], functions, inputs)
                        )
                else:
                    if rand().random() > 0.95 and (
                        not (
                            goal_bool
                            and int(
                                "0b" + "".join(str(int(s)) for s in conditions[i][-1]),
                                base=2,
                            )
                            in goal_states
                        )
                    ):  # 5% chance of perturbation starting
                        perturb_occurring[i] = True
                        conditions[i].append(
                            next_state(
                                conditions[i][-1],
                                functions,
                                inputs,
                                perturb=perturb_tups[perturb_selectors[i]][0],
                            )
                        )
                    else:
                        conditions[i].append(
                            next_state(conditions[i][-1], functions, inputs)
                        )
                if (
                    with_noise is not None
                    and not isinstance(with_noise, str)
                    and rand().random() <= with_noise
                ):
                    perturb_node = rand().randrange(0, len(net))
                    conditions[i][-1][perturb_node] = not bool(
                        conditions[i][-1][perturb_node]
                    )
                elif (
                    with_noise is not None
                    and isinstance(with_noise, str)
                    and with_noise == "bv"
                ):
                    prev_step_bv = sum(
                        int(conditions[i][-2][k] != conditions[i][-1][k])
                        for k in range(len(conditions[i][-1]))
                    ) / len(net)
                    if rand().random() * (1 - prev_step_bv) < 0.05:
                        perturb_node = rand().randrange(0, len(net))
                        conditions[i][-1][perturb_node] = not bool(
                            conditions[i][-1][perturb_node]
                        )
                elif (
                    with_noise is not None
                    and isinstance(with_noise, str)
                    and with_noise == "inverse bv"
                ):
                    prev_step_bv = sum(
                        int(conditions[i][-2][k] != conditions[i][-1][k])
                        for k in range(len(conditions[i][-1]))
                    ) / len(net)
                    if rand().random() < 0.05 * (1 - prev_step_bv):
                        perturb_node = rand().randrange(0, len(net))
                        conditions[i][-1][perturb_node] = not bool(
                            conditions[i][-1][perturb_node]
                        )

        if step % progress_div == 0:
            # prog_row = []
            # for c in conditions:
            #     if int('0b' + ''.join(str(int(s)) for s in c[-1]), base = 2) in goal_states:
            #         prog_row.append(1)
            #     else:
            #         prog_row.append(0)
            # progress_rows.append(prog_row)
            progress_rows.append(
                [
                    cycle_colors[
                        (
                            -1
                            if net.bn_collapsed_cycles.get_index(c[-1]) is None
                            else net.bn_collapsed_cycles.get_index(c[-1])
                        )
                    ]
                    for c in conditions
                ]
            )
            if frequency_stacked:
                stacked_row = []
                for n in range(-1, len(net.bn_collapsed_cycles.cycle_records)):
                    for _ in range(progress_rows[-1].count(cycle_colors[n])):
                        stacked_row.append(cycle_colors[n])
                stacked_rows.append(stacked_row)

    if not frequency_stacked:
        fig = plt.figure()
        plt.imshow(progress_rows, aspect=1 / 8)
        #
        return fig
        #
    else:
        fig1 = plt.figure()
        # plt.imshow(progress_rows, cmap=cmap)
        # from: https://stackoverflow.com/a/10970122/25597680, aspect
        plt.imshow(progress_rows, aspect=1 / 8)
        fig2 = plt.figure()
        plt.imshow(stacked_rows, aspect=1 / 8)
        #
        return fig1, fig2
        #

    # return fig
    # return fig, display_cycle_colors

    # a_state = net.current_states_list[-1]
    # next_state = [functions[i](bool(a_state[inputs[i][0]]), bool(a_state[inputs[i][1]])) for i in range(len(functions))]
    # print(next_state)
    # def advance_state(state, functions, inputs, L = 1):
    #     ad_state = [functions[i](bool(state[inputs[i][0]]), bool(state[inputs[i][1]])) for i in range(len(functions))]
    #     for i in range(1, L):
    #         ad_state = [functions[i](bool(ad_state[inputs[i][0]]), bool(ad_state[inputs[i][1]])) for i in range(len(functions))]  # .../ np namespace and aliasing == good
    #     return ad_state
