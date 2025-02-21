import numpy as np
from package.boolean_networks import BooleanNetwork
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from random import SystemRandom as rand
import copy
from package.plotting import get_colors

import matplotlib.pyplot as plt
import numpy as np


# # Sample categorical data
# data = np.array([[0, 1, 2],
#                  [1, 2, 0],
#                  [2, 0, 1]])

# # Define colors for categories
# colors = ['red', 'green', 'blue']
# cmap = ListedColormap(colors)

# # Display the data with categorical colors
# plt.imshow(data, cmap=cmap, interpolation='nearest')

# # Add a colorbar with categorical labels
# bounds = np.arange(len(colors) + 1) - 0.5
# norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
# cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
#                     ticks=np.arange(len(colors)),
#                     boundaries=bounds)
# cbar.ax.set_yticklabels(['Category A', 'Category B', 'Category C'])

# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # Define the color data as a NumPy array
# color_data = np.array([
#     [ [1, 0, 0], [0, 1, 0], [0, 0, 1] ], # Red, Green, Blue
#     [ [1, 1, 0], [1, 0, 1], [0, 1, 1] ], # Yellow, Magenta, Cyan
#     [ [1, 1, 1], [0.5, 0.5, 0.5], [0, 0, 0] ]  # White, Gray, Black
# ])

# # Display the color data as blocks
# plt.imshow(color_data)
# plt.xticks([])
# plt.yticks([])
# plt.show()
#
# # So, progress bars as lists of colors



def seq_pert_report(p1, p2, goal_cycle_index, net: BooleanNetwork, cycle_colors: list = [], total_steps: int = 100, progress_div: int = 1, goal_bool: bool = False, p3: int | None = None, with_noise: str | float | None = None):
    
    if cycle_colors == []:
        cycle_colors = get_colors(len(net.bn_collapsed_cycles.cycle_records), True)
    
    # c_colors = [cycle_colors[i] for i in range(len(net.bn_collapsed_cycles.cycle_records))]
    # cmap = ListedColormap(c_colors)

    goal_states = set(int('0b' + ''.join(str(int(s[i])) for i in range(len(s))), base = 2)
                    for s in net.bn_collapsed_cycles.cycle_records[goal_cycle_index].cycle_states_list)
    
    cycle_colors = [[cc[0], cc[1], cc[2], 0.9] if i != goal_cycle_index else [cc[0], cc[1], cc[2], cc[3]] for cc, i in zip(cycle_colors, range(len(cycle_colors)))]

    display_cycle_colors = [[cycle_colors[i]] for i in range(len(cycle_colors)) if i in list(range(0, len(net.bn_collapsed_cycles.cycle_records))) + [-1 % len(cycle_colors)]]
    fig_leg = plt.figure()
    fig_leg.set_size_inches(10, 1)
    plt.imshow(np.transpose(display_cycle_colors))
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
    def get_perturb_tuples(p1, p2, net: BooleanNetwork, p3: int | None = None) -> list[tuple]:
        p_p_L = []
        if p3 is None:
            for L in range(1, net.longest_cycle_length()):  # could put different distributions here
                p_p_L.append((p1, p2, L))
        else:
            for L in range(1, net.longest_cycle_length()):  # could put different distributions here
                p_p_L.append((p1, p2, L, p3))
        return p_p_L

    # advance state
    def next_state(state, functions, inputs, perturb: int | None = None):
        if perturb is None:
            # print(f"args:\nstate: {state}\nfunctions: {functions}\ninputs: {inputs}, perturb: {perturb}")
            return bytearray(functions[i](bool(state[inputs[i][0]]), bool(state[inputs[i][1]]))
                            for i in range(len(functions)))
        else:
            return bytearray((not functions[i](bool(state[inputs[i][0]]), bool(state[inputs[i][1]])))
                            if i == perturb else functions[i](bool(state[inputs[i][0]]), bool(state[inputs[i][1]]))
                            for i in range(len(functions)))


    # setup parallel network states
    functions = [net.nodes[i].function.get_boolean for i in range(len(net.nodes))]
    inputs = net.node_inputs_assignments

    conditions = get_ic_cycles(net)  # 1 per each cycle state
    perturb_tups = get_perturb_tuples(p1, p2, net)
    el_counters = [0 for _ in conditions]  # count intervals between p1, p2
    perturb_occurring = [False for _ in conditions]  # sieve application of perturbation sequences
    perturb_selectors = [rand().randrange(len(perturb_tups)) for _ in conditions]  # index reference to perturb tups

    progress_rows = []

    for step in range(total_steps):
        # sum(int(state[-k]) * 2**k for k in range(1, len(state) + 1))
        for i in range(len(conditions)):
            if goal_bool and int('0b' + ''.join(str(int(s)) for s in conditions[i][-1]), base = 2) in goal_states:
                conditions[i].append(conditions[i][-1])  # typo: -2, hmm, could go back a ways (but not using a single transition matrix)
            else:
                if perturb_occurring[i]:
                    el_counters[i] += 1
                    if el_counters[i] == perturb_tups[perturb_selectors[i]][2]:  # L: interval, ex: 1 step since start (p1), for L=1, apply p2
                        conditions[i].append(next_state(conditions[i][-1], functions, inputs, perturb_tups[perturb_selectors[i]][1]))
                        if len(perturb_tups[0]) == 3:
                            perturb_occurring[i] = False
                            perturb_selectors[i] = rand().randrange(len(perturb_tups))  # can add complexity here
                            el_counters[i] = 0
                    elif len(perturb_tups[0]) == 4 and el_counters[i] == 2 * perturb_tups[perturb_selectors[i]][2]:  # L: interval, ex: 1 step since start (p1), for L=1, apply p2
                        conditions[i].append(next_state(conditions[i][-1], functions, inputs, perturb_tups[perturb_selectors[i]][3]))
                        perturb_occurring[i] = False
                        perturb_selectors[i] = rand().randrange(len(perturb_tups))
                        el_counters[i] = 0
                    else:
                        conditions[i].append(next_state(conditions[i][-1], functions, inputs))
                else:
                    if rand().random() > 0.95 and (not (goal_bool and int('0b' + ''.join(str(int(s)) for s in conditions[i][-1]), base = 2) in goal_states)):  # 5% chance of perturbation starting
                        perturb_occurring[i] = True
                        conditions[i].append(next_state(conditions[i][-1], functions, inputs, perturb=perturb_tups[perturb_selectors[i]][0]))
                    else:
                        conditions[i].append(next_state(conditions[i][-1], functions, inputs))
                if with_noise is not None and not isinstance(with_noise, str) and rand().random() <= with_noise:
                    perturb_node = rand().randrange(0, len(net))
                    conditions[i][-1][perturb_node] = not bool(conditions[i][-1][perturb_node])
                elif with_noise is not None and isinstance(with_noise, str) and with_noise == 'bv':
                    prev_step_bv = sum(int(conditions[i][-2][k] != conditions[i][-1][k]) for k in range(len(conditions[i][-1]))) / len(net)
                    if rand().random() * (1 - prev_step_bv) < 0.05:
                        perturb_node = rand().randrange(0, len(net))
                        conditions[i][-1][perturb_node] = not bool(conditions[i][-1][perturb_node])

                    
        if step % progress_div == 0:
            # prog_row = []
            # for c in conditions:
            #     if int('0b' + ''.join(str(int(s)) for s in c[-1]), base = 2) in goal_states:
            #         prog_row.append(1)
            #     else:
            #         prog_row.append(0)
            # progress_rows.append(prog_row)
            progress_rows.append([cycle_colors[-1 if net.bn_collapsed_cycles.get_index(c[-1]) is None else net.bn_collapsed_cycles.get_index(c[-1])] for c in conditions])

    fig = plt.figure()
    # plt.imshow(progress_rows, cmap=cmap)
    plt.imshow(progress_rows)
    # plt.show()

    return fig
    # return fig, display_cycle_colors



    # a_state = net.current_states_list[-1]
    # next_state = [functions[i](bool(a_state[inputs[i][0]]), bool(a_state[inputs[i][1]])) for i in range(len(functions))]
    # print(next_state)
    # def advance_state(state, functions, inputs, L = 1):
    #     ad_state = [functions[i](bool(state[inputs[i][0]]), bool(state[inputs[i][1]])) for i in range(len(functions))]
    #     for i in range(1, L):
    #         ad_state = [functions[i](bool(ad_state[inputs[i][0]]), bool(ad_state[inputs[i][1]])) for i in range(len(functions))]  # .../ np namespace and aliasing == good
    #     return ad_state