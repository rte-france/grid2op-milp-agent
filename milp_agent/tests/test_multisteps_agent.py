# Copyright (c) 2020-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of MILP_Agent, The MILP_Agent is a optimization based agent that manage the power
# flow overthermal using topological actions.
import numpy as np

import grid2op 
from grid2op.Agent import DoNothingAgent
from grid2op.PlotGrid import PlotMatplot
from grid2op.Action import DontAct
from grid2op.Chronics import GridStateFromFileWithForecastsWithoutMaintenance

import milp_agent
from milp_agent import GLOBAL_SWITCH, MIP_CBC
from milp_agent.agent import MILPAgent
from milp_agent.usecase import Simulation

env_name = "l2rpn_case14_sandbox"
env_name = "l2rpn_neurips_2020_track1_small"
env_name = "l2rpn_neurips_2020_track2"

path_result_simulation = 'result_simulation'
env = grid2op.make(env_name, 
                   test=True)
episode_dict = {3:0}
obs = env.get_obs()
margins = 0.95*np.ones(obs.n_line)
Agent_type = GLOBAL_SWITCH
Solver_type = MIP_CBC
simulation = Simulation(env, Agent_type, margins, solver_name = Solver_type,  zone_level = 0, display_obs = True, result_return=path_result_simulation)
result = simulation.run_episodes(max_iter=10, episodes=episode_dict)