# Copyright (c) 2020-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of MILP_Agent, The MILP_Agent is a optimization based agent that manage the power
# flow overthermal using topological actions.

import warnings
import numpy as np

import grid2op
from grid2op.Opponent import BaseOpponent
from grid2op.Action import DontAct

from milp_agent.agent import MILPAgent
from milp_agent import GLOBAL_TOPOLOGY
from milp_agent.global_var import MIP_SCIP

import unittest


class TestCanMakeAgent(unittest.TestCase):
    def _aux_test(self, env_name):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            orig_env = grid2op.make(env_name,
                                    opponent_class=BaseOpponent,
                                    opponent_init_budget=0.,
                                    opponent_action_class=DontAct,
                                    test=True)

        agent = MILPAgent(env=orig_env,
                          agent_solver=GLOBAL_TOPOLOGY,
                          max_overflow_percentage=np.ones(orig_env.n_line) * 0.90,
                          solver_name=MIP_SCIP)
        obs = orig_env.reset()
        act = agent.act(obs, 0.0, done=False)
        new_obs, reward, done, info = orig_env.step(act)

    def test_case_14(self):
        env_name = "l2rpn_case14_sandbox"
        self._aux_test(env_name)

    def test_ieee118(self):
        env_name = "l2rpn_neurips_2020_track2"
        self._aux_test(env_name)

    def test_ieee118_R2(self):
        env_name = "l2rpn_neurips_2020_track1"
        self._aux_test(env_name)
