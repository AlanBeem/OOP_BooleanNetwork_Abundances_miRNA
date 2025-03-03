from package.boolean_networks import BooleanNetwork
import numpy as np


def select_p1_p2(
    net: BooleanNetwork,
    prec_tuples: list,
    base_cycle_states: list,
    seq_perts: list,
    cycle_index: int,
    states_labels: dict,
):
    got_sequence = False
    initial_states = []
    for p1 in set(prec_tuples[i][1] for i in range(len(prec_tuples))):
        for p2 in set(
            prec_tuples[i][2]
            for i in range(len(prec_tuples))
            if prec_tuples[i][1] == p1
        ):
            print(f"\r\t\t\t\tp1: {p1}, p2: {p2}", end="\r")
            for state in base_cycle_states:
                if got_sequence:
                    # break
                    continue
                # apply p1
                p_state = [
                    not (state[k]) if k == p1 else state[k] for k in range(len(state))
                ]
                net.set_conditions(conditions=p_state)  # set net state to state,
                for L in range(min_interval, max_interval + 1):
                    net.advance_state()
                    initial_states.append(copy.deepcopy(net.current_states_list[-1]))
                # apply p2
                # to each of initial (p1) states; run-in from those conditions
                for m in range(len(initial_states)):
                    p1_p2_state = [
                        not (initial_states[m][k]) if k == p2 else initial_states[m][k]
                        for k in range(len(initial_states[m]))
                    ]
                    # net.run_in_from_conditions(p1_p2_state)
                    net.set_conditions(p1_p2_state)
                    while (
                        int(
                            "0b"
                            + "".join(
                                [
                                    str(int(net.current_states_list[-1][i]))
                                    for i in range(len(net.current_states_list[-1]))
                                ]
                            ),
                            base=2,
                        )
                        not in states_labels
                    ):
                        # print('\r... 1')
                        net.advance_state()
                    else:
                        # print('\r..1 1 1 ')
                        cyc_index = states_labels.get(
                            int(
                                "0b"
                                + "".join(
                                    [
                                        str(int(net.current_states_list[-1][i]))
                                        for i in range(len(net.current_states_list[-1]))
                                    ]
                                ),
                                base=2,
                            ),
                            net.bn_collapsed_cycles.get_index(
                                net.current_states_list[-1]
                            ),
                        )
                        for j in range(len(net.current_states_list)):
                            states_labels.update(
                                {
                                    int(
                                        "0b"
                                        + "".join(
                                            [
                                                str(int(net.current_states_list[j][i]))
                                                for i in range(
                                                    len(net.current_states_list[j])
                                                )
                                            ]
                                        ),
                                        base=2,
                                    ): cyc_index
                                }
                            )
                        if (
                            states_labels.get(
                                int(
                                    "0b"
                                    + "".join(
                                        [
                                            str(int(net.current_states_list[-1][i]))
                                            for i in range(
                                                len(net.current_states_list[-1])
                                            )
                                        ]
                                    ),
                                    base=2,
                                )
                            )
                            == goal_cycle_index
                        ):
                            # print(cycle_index)
                            seq_perts[cycle_index].append(
                                (
                                    (cycle_index, goal_cycle_index),
                                    p1,
                                    p2,
                                    (m + 1) % (max_interval - min_interval),
                                )
                            )
                            got_sequence = True
