# Copyright (c) 2020-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of MILP_Agent, The MILP_Agent is a optimization based agent that manage the power flow
# overthermal using topological actions.


from .global_var import ETA_OR, SWITCH_B, ETA_EX, BRANCH, ORIGIN, EXTREMITY, SET_LINE_STATUS, ETA_PROD, GEN, \
    ETA_LOAD, LOAD, CHANGE_BUS, LINES_OR_ID, LINES_EX_ID, GENERATORS_ID, LOADS_ID


class AgentAnswer:
    """
    Class to convert optimisation solution into the corresponding grid2op action
    """
    def __init__(self, 
                 action_space, 
                 solution: dict):
        """
        :param action_space: Grid2op action space object
        :param solution: Solution of optimisation problem
        """
        self.action_space = action_space
        self.solution = solution

    def solution_to_action(self, line_status: list, bus_status: dict):
        """
        Convert the optimization result into grid2op action

        :param list line_status: Line_status[i] = True if line i is connected, False the line is disconnected
        :param dict bus_status: Gives bus connections for each object
        """
        action = self.action_space({})
        set_status = self.action_space.get_set_line_status_vect()
        using_bus_model = False
        lines_or_id = []
        lines_ex_id = []
        gen_id = []
        loads_id = []
        if self.solution[ETA_OR]:
            using_bus_model = True
        if self.solution[SWITCH_B]:  # Line status and bus change
            for i in self.solution[SWITCH_B].keys():
                delta = self.solution[SWITCH_B][i]
                if delta >= 0.5 and not line_status[i]:
                    if using_bus_model:
                        bus_or = (2 * int(self.solution[ETA_OR][i]) + 2) % 3
                        bus_ex = (2 * int(self.solution[ETA_EX][i]) + 2) % 3
                        action += self.action_space.reconnect_powerline(line_id=i, bus_or=bus_or, bus_ex=bus_ex)
                    else:
                        # action += self.action_space.reconnect_powerline(line_id=i, bus_or=1, bus_ex=1)
                        # grid2op will automatically reconnect it to proper bus in this case
                        set_status[i] = +1
                elif delta < 0.5 and line_status[i]:
                    pass
                    set_status[i] = -1
                elif delta >= 0.5 and line_status[i] and using_bus_model:
                    bus_or = (2 * int(self.solution[ETA_OR][i]) + 2) % 3
                    if abs(bus_or - bus_status[BRANCH][ORIGIN][i]) > 0.1:
                        lines_or_id.append(i)
                    bus_ex = (2 * int(self.solution[ETA_EX][i]) + 2) % 3
                    if abs(bus_ex - bus_status[BRANCH][EXTREMITY][i]) > 0.1:
                        lines_ex_id.append(i)
        action += self.action_space({SET_LINE_STATUS: set_status})
        if using_bus_model:  # Injection bus change
            for i in self.solution[ETA_PROD].keys():
                bus = (2 * int(self.solution[ETA_PROD][i]) + 2) % 3
                if abs(bus - bus_status[GEN][i]) > 0.1:
                    gen_id.append(i)
            for i in self.solution[ETA_LOAD].keys():
                bus = (2 * int(self.solution[ETA_LOAD][i]) + 2) % 3
                if abs(bus - bus_status[LOAD][i]) > 0.1:
                    loads_id.append(i)
        action += self.action_space({CHANGE_BUS: {LINES_OR_ID: lines_or_id, LINES_EX_ID: lines_ex_id,
                                                      GENERATORS_ID: gen_id, LOADS_ID: loads_id}})
        return action 
