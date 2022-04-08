# Copyright (c) 2020-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of MILP_Agent, The MILP_Agent is a optimization based agent that manage the power flow
# overthermal using topological actions.

import os
import numpy as np
import pandapower as pp

import grid2op
from grid2op.Agent import BaseAgent
from grid2op.Parameters import Parameters
from grid2op.Backend import PandaPowerBackend
from grid2op.Environment import MultiMixEnvironment, Environment

from .logconf import logger_conf
from .build import Builder
from .global_var import MIP_SCIP, NB_BRANCH, LINE, TRAFO, NB_BUS, X_BRANCH, SHIFT, branch, MAX_I_KA, ZONE_BRANCH, \
    ZONE_PROD, GEN, ZONE_LOAD, LOAD, ZONE_BUS, BUS, RHO, P_OR, Q_OR, MULTI_ZONES, DO_NOTHING, \
    GLOBAL_STATE_ESTIMATION, GLOBAL_SWITCH, GLOBAL_TOPO_ZONE, GLOBAL_DOUBLE_LEVEL, GLOBAL_TOPOLOGY, UNKNOWN_SOLVER, \
    SWITCH_B, THETA, FMAX, SLACK_B, SOLVING_TIME, SOLVER_CHECK_WRONG, ETA, STATE_ESTIMATION, U1, U2, SLACK_N_PLUS, \
    MODIF_BRANCHES, MODIF_SUB, GLOBAL_ACTION, ZONE, LINE_OVERFLOW, ACTIONS_TAKEN, FROM_BUS, HV_BUS, \
    BRANCH, ORIGIN, EXTREMITY, DELTA_FLOW, DELTA_CHARGE, FLOW_B, MAX_FLOW, ETA_OR, ETA_EX, ETA_PROD, ETA_LOAD, \
    SUB_CHANGED
from .zone import Zones    
from .agent_answer import AgentAnswer
from time import time


class MILPAgent(BaseAgent):
    """
    Class derived from the BaseAgent class (from grid2op). The act method should return a valid grid2op action.
    Firstly, solve the optimization problem, then use The AgentAnswer class to return the valid action in act method
    """
    def __init__(self,
                 env,
                 agent_solver: str,
                 max_overflow_percentage,
                 solver_name=MIP_SCIP,
                 zone_instance: dict = None,
                 clustering_path: str = None,
                 zone_level: int = 0,
                 logger=None):
        """
        :param env: grid2op environment
        :param str agent_solver: Name of agent, determine which type of agent used.
        :param max_overflow_percentage: Percentages of the flow limit from which lines are considered overloaded
        :param solver_name : the type of solver used for optimization problem
        :param dict zone_instance: Zone for the zones agent
        :param str clustering_path: the path towards the clustering file, it defines the zones that the
          multi_agent control
        :param int zone_level: Radius of supplementary ring around the zone, used for both double level and
          multi agent controlors
        :param logger: logger to scree showin result of simulation

        """
        self.env = env
        BaseAgent.__init__(self, self.env.action_space)

        if isinstance(env, MultiMixEnvironment):
            grid_path = os.path.join(env.current_env.get_path_env(), "grid.json")
        elif isinstance(env, Environment):
            grid_path = os.path.join(env.get_path_env(), "grid.json")
        else:
            raise RuntimeError("The MILPAgent cannot use the environment you provided. Please make sure it's "
                               "a grid2op Environment.")

        if not os.path.exists(grid_path):
            raise RuntimeError("For now the milp_agent is only available if the grid format used is compatible "
                               "with the PandaPower backend. This means that your grid format should be readable "
                               "with the pandapower library which does not appear to be the case.")
        agent_backend = PandaPowerBackend()
        agent_backend.load_grid(grid_path)
        env_backend = agent_backend._grid._ppc
        self.net = pp.from_json(grid_path)

        self.carac = {NB_BRANCH: len(self.net[LINE]) + len(self.net[TRAFO]), NB_BUS: len(self.net[BUS]), X_BRANCH: {}}
        self.carac[SHIFT] = [env_backend[branch][i][9].real * np.pi/180 for i in range(self.carac[NB_BRANCH])]
        for key in branch_keys(self.net):
            self.carac[X_BRANCH][key] = env_backend[branch][key][3].real * env_backend[branch][key][8].real
        self.carac[MAX_I_KA] = self.net[LINE][MAX_I_KA]
        self.agent_solver = agent_solver
        self.solution = {}
        self.max_overflow_percentage = max_overflow_percentage
        self.zone = zone_instance
        self.solver_name = solver_name
        self.fix_zone = False
        if self.zone is not None:
            self.complementary_zone = self.compute_complementary_zone()
        self.zones = None
        self.zone_level = zone_level  
        self.clustering_path =clustering_path
        
        if logger is None:
            self.logger = logger_conf()
        else:
            self.logger = logger
    def compute_complementary_zone(self) -> dict:
        """
        Returns dict of elements that are not in self.zone

        :return: Complement of self.zone
        """
        return {ZONE_BRANCH: list(set([i for i in range(self.carac[NB_BRANCH])]).difference(self.zone[ZONE_BRANCH])),
                ZONE_PROD: list(set([i for i in range(len(self.net[GEN]))]).difference(self.zone[ZONE_PROD])),
                ZONE_LOAD: list(set([i for i in range(len(self.net[LOAD]))]).difference(self.zone[ZONE_LOAD])),
                ZONE_BUS: list(set([i for i in range(self.carac[NB_BUS])]).difference(self.zone[ZONE_BUS]))}

    def display_lines_overloaded(self, observation) -> None:
        """
        Displays overloaded lines from network observation from grid2op

        :param observation: Observation of the network from grid2op
        """
        lines_overflow = {"Lines overflow": " "}
        for line_id in range(len(observation.rho)):
            if observation.rho[line_id] > self.max_overflow_percentage[line_id]:
                lines_overflow[line_id] = {RHO: str(int(100*observation.rho[line_id]))+" % ",
                                           P_OR: observation.p_or[line_id], Q_OR: observation.q_or[line_id]}
        self.logger.info(lines_overflow)

    def act(self, observation, reward, done=False):
        """
        Overrides method in BaseAgent, called by the environment to retrieve the action to perform

        :param observation: Network observation 
        :param reward: Current reward of the agent used for reinforcement learning agents, will have no use for us
        :param done: Is the environment in a game over situation ?
        :return: Agent action to apply on the environment
        """
        # print(" ")
        self.display_lines_overloaded(observation=observation)
        if self.agent_solver == MULTI_ZONES:  # Multi-zones returns an action not a solution dict
            self.zones = Zones(obs=observation,clustering_path=self.clustering_path )
            return self.multi_zone_action(observation)
        solution = self.solve(observation)
        assert solution
        self.solution = solution
        answer = AgentAnswer(self.action_space, solution)
        action = answer.solution_to_action(observation.line_status, bus_status=compute_bus_status(observation))
        if action != self.action_space({}):
            self.logger.info(action)
        return action

    def solve(self, observation) -> dict:
        """
        Called by act method. Gives solution of the optimization problem solved by the agent with respcet to its name.

        :param observation: Network observation
        :return: Solution of agent optimisation problem
        """
        if len(self.carac[MAX_I_KA]) != self.carac[NB_BRANCH]:  # In case MAX_I_KA is not given in the net dict
            # Read it from observation directly
            self.carac[MAX_I_KA] = observation.thermal_limit * 0.001
            # print("reading thermal limit")
        if self.agent_solver == GLOBAL_STATE_ESTIMATION:
            return self.global_state_estimation(observation)
        elif self.agent_solver == GLOBAL_SWITCH:
            return self.global_switch(observation)
        elif self.agent_solver == GLOBAL_TOPO_ZONE:
            fixed = fixed_items_from_observation(observation, self.complementary_zone)
            return self.global_topology_with_fixed(observation, fixed, list(fixed[SWITCH_B].keys()))
        elif self.agent_solver == GLOBAL_DOUBLE_LEVEL:
            return self.global_double_level_agent(observation)
        elif self.agent_solver == GLOBAL_TOPOLOGY:
            return self.global_topology(observation)
        else:
            self.logger.error(UNKNOWN_SOLVER)
            solution = {}
            return solution

    def global_switch(self, observation) -> dict:
        """
        Solves the optimization problem with only line connection/disconnection actions available

        :param observation: Network observation
        :return: Solution of optimisation problem
        """
        solution = {}
        const_list = [THETA, FMAX, SLACK_B, SWITCH_B]
        problem = Builder(observation, self.carac, self.max_overflow_percentage,
                          solver_name=self.solver_name
                          )
        problem.activate_const_list(const_list)
        problem.common_add_var()
        problem.balance_nodes_for_state_estimation()
        problem.add_max_flow_constraints()
        problem.common_add_constraints_bus()
        problem.common_add_constraints_bus_contrib_fixed()
        problem.common_add_flow_theta_const_mip()
        problem.production_var_equal_observed_prods()
        problem.limit_switch_actions(Parameters().MAX_LINE_STATUS_CHANGED)
        problem.common_add_objective()
        t1 = time()
        if problem.solve_status():
            solution = problem.get_solution(display=False)
            self.logger.info((SOLVING_TIME, time() - t1))
            problem.check_branch_const()
            problem.check_theta(solution, problem.X_branch)
            if not problem.solver.solver.VerifySolution(tolerance=1e-1, log_errors=True):
                self.logger.warning(SOLVER_CHECK_WRONG)
        return solution

    def global_topology(self, observation) -> dict:
        """
        Solves the optimisation problem with all possible actions (line connection and substation topology)

        :param observation: Network observation
        :return: Solution of optimisation problem
        """
        solution = {}
        const_list = [THETA, FMAX, SLACK_B, SWITCH_B, ETA]
        problem = Builder(observation, self.carac, self.max_overflow_percentage,
                          solver_name=self.solver_name)
        problem.activate_const_list(const_list)
        problem.common_add_var()
        problem.balance_nodes_for_state_estimation()
        problem.add_max_flow_constraints()
        problem.common_add_constraints_bus()
        problem.common_add_constraints_bus_contrib()
        problem.common_add_flow_theta_const_mip()
        problem.break_symmetry()
        problem.production_var_equal_observed_prods()
        problem.limit_number_of_substations_changed(Parameters().MAX_SUB_CHANGED)
        problem.limit_switch_actions(Parameters().MAX_LINE_STATUS_CHANGED)
        problem.common_add_objective()
        t2 = time()
        if problem.solve_status():
            t3 = time()
            solution = problem.get_solution(display=False)
            self.logger.info((SOLVING_TIME, t3 - t2))
            problem.check_branch_const()
            problem.check_theta(solution, problem.X_branch)
            if not problem.solver.solver.VerifySolution(tolerance=1e-1, log_errors=True):
                self.logger.warning(SOLVER_CHECK_WRONG)
        return solution

    def global_state_estimation(self, observation) -> dict:
        """
        Runs a state estimation optimisation to stick to AC flow observations. Will not be used afterwards because
        observations from grid2op simulations are exact.

        :param observation: Network observation
        :return: Solution of linearized optimisation problem
        """
        solution = {}
        problem = Builder(observation, self.carac, self.max_overflow_percentage, solver_name=STATE_ESTIMATION,
                          zone_instance=self.zone)
        problem.activate_const_list([THETA])
        problem.common_add_var()
        problem.add_quad_var()
        problem.balance_nodes_for_state_estimation()
        problem.common_add_constraints_bus()
        problem.common_add_flow_theta_const_lp()
        problem.common_add_constraints_bus_contrib_fixed()
        problem.add_linearised_constraints()
        problem.add_overflow_constraints_for_state_estimation()
        problem.production_var_equal_observed_prods()
        problem.set_quad_objective()
        if problem.solve_status():
            solution = problem.get_solution(display=False)
            problem.check_branch_const()
            problem.check_false_negative_state_estimation()
            if not problem.solver.solver.VerifySolution(tolerance=1e-1, log_errors=True):
                self.logger.warning(SOLVER_CHECK_WRONG)
        extensions = [U1, U2]
        for i in problem.subs:
            for bus in range(2):
                ext = extensions[bus]
                solution[SLACK_N_PLUS+ext][i] += problem.nodes_balance_by_bus2[bus+1][i]
        return solution

    def global_topology_with_fixed(self, observation, fixed: dict, no_overflow_branches: list) -> dict:
        """
        Runs a global topology optimization with some fixed variables and branches whose power limits are
        considered as infinite

        :param Any observation: Network observation
        :param dict fixed: Binary variables to fix
        :param list no_overflow_branches: branches that will be viewed with very high capacity
        :return: Solution of the optimisation problem for actions on elements not in fixed
        """
        solution = {}
        const_list = [THETA, FMAX, SLACK_B, SWITCH_B, ETA]
        problem = Builder(observation, self.carac, self.max_overflow_percentage, solver_name=self.solver_name)
        problem.activate_const_list(const_list)
        problem.common_add_var()
        problem.fix_variables_in_fixed(fixed=fixed)
        problem.remove_limit_constraint_on_branch_area(branches=no_overflow_branches)
        problem.balance_nodes_for_state_estimation()
        problem.add_max_flow_constraints()
        problem.common_add_constraints_bus()
        problem.common_add_constraints_bus_contrib()
        problem.common_add_flow_theta_const_mip()
        problem.break_symmetry()
        problem.production_var_equal_observed_prods()
        problem.limit_switch_actions(1 + fixed[MODIF_BRANCHES])
        problem.limit_number_of_substations_changed(1 + fixed[MODIF_SUB])
        problem.common_add_objective()
        t1 = time()
        if problem.solve_status():
            solution = problem.get_solution(display=False)
            self.logger.info((SOLVING_TIME, time() - t1))
        problem.check_branch_const()
        problem.check_theta(solution, problem.X_branch)
        if not problem.solver.solver.VerifySolution(tolerance=1e-1, log_errors=True):
            self.logger.warning(SOLVER_CHECK_WRONG)
        return solution

    def global_double_level_agent(self, observation) -> dict:
        """
        Double level agent. Does an iteration outside the zone and then an iteration on the zone.

        :param observation: Network observation
        :return: Solution as if it was a combined action
        """
        fixed = fixed_items_from_observation(observation, self.zone)
        global_solution = self.global_topology_with_fixed(observation, fixed, list(fixed[SWITCH_B].keys()))
        answer = AgentAnswer(self.action_space, global_solution)
        action = answer.solution_to_action(observation.line_status, bus_status=compute_bus_status(observation))
        if action != self.action_space({}):
            self.logger.info((GLOBAL_ACTION, action))
        fixed = fixed_items_from_solution(global_solution, self.complementary_zone)
        fixed[MODIF_BRANCHES] = distance_solution_observation(solution=global_solution, observation=observation)
        local_solution = self.global_topology_with_fixed(observation, fixed, list(fixed[SWITCH_B].keys()))
        return local_solution

    def multi_zone_action(self, observation):
        """
        Multi-zones agent iterating on each zone of the clustering. Contrary to other agents, this methods returns
        an action instead of a solution dict because it is more convenient to add actions rather than solutions

        :param observation: Network observation
        :return: Total action as concatenation of every zone's action
        """
        if len(self.carac[MAX_I_KA]) != self.carac[NB_BRANCH]:  # In case MAX_I_KA is not given in the net dict
            self.carac[MAX_I_KA] = observation.a_or/(1000*observation.rho)  # Read it from observation directly
        total_action = self.action_space({})
        for zone_id in self.zones.clustering.keys():
            level = self.zone_level  # orbit of zone, default to zero
            self.zone = self.zones.zone_ring(zone=self.zones.clustering[zone_id], level=0)
            zone_overflow = {}
            for line_id in self.zone[ZONE_BRANCH]:
                if observation.rho[line_id] > self.max_overflow_percentage[line_id]:
                    zone_overflow[line_id] = str(int(100*observation.rho[line_id]))+"%"
            self.logger.info((ZONE, zone_id, LINE_OVERFLOW, zone_overflow))
            self.complementary_zone = self.compute_complementary_zone()
            fixed = fixed_items_from_observation(observation=observation, zone_to_fix=self.complementary_zone)
            self.zone = self.zones.zone_ring(zone=self.zones.clustering[zone_id], level=level)
            no_line_limit = self.compute_complementary_zone()[ZONE_BRANCH]
            for line_id in self.zone[ZONE_BRANCH]:  # add overloaded lines in extended zone to the don't-see-limit-list
                if line_id not in self.zones.clustering[zone_id][ZONE_BRANCH]\
                        and observation.rho[line_id] > self.max_overflow_percentage[line_id]:
                    no_line_limit.append(line_id)
            if zone_overflow != {}:  # Act on zone if any constraints
                solution = self.global_topology_with_fixed(observation, fixed=fixed, no_overflow_branches=no_line_limit)
                answer = AgentAnswer(self.action_space, solution)
                action = answer.solution_to_action(observation.line_status, compute_bus_status(observation))
            else:  # Otherwise, do nothing on that zone
                action = self.action_space({})
            if action != self.action_space({}):
                self.logger.info((ZONE, zone_id, ACTIONS_TAKEN, action))
            total_action += action
        if total_action != self.action_space({}):
            self.logger.info((GLOBAL_ACTION, total_action))
        else:
            self.logger.info((GLOBAL_ACTION, DO_NOTHING))
        return total_action


########################################################################################################################
    # Auxiliary functions
########################################################################################################################


def generate_base(filename: str):
    """
    Generates entry data for simulations. Is used for tests

    :param filename: name of environment
    :return: grid network, network observation and generators characteristics
    """
    env = grid2op.make(filename, test=True)
    grid_path = env._init_grid_path
    env_backend = env.backend._grid._ppc
    net = pp.from_json(grid_path)
    carac = {NB_BRANCH: len(net[LINE]) + len(net[TRAFO]), NB_BUS: len(net[BUS]), X_BRANCH: {}}
    carac[SHIFT] = [env_backend[branch][i][9].real * np.pi / 180 for i in range(carac[NB_BRANCH])]
    for key in branch_keys(net):
        carac[X_BRANCH][key] = env_backend[branch][key][3].real * env_backend[branch][key][8].real
    carac[MAX_I_KA] = net[LINE][MAX_I_KA]
    obs = env.get_obs()
    carac[MAX_I_KA] = obs.a_or/(1000*obs.rho)
    return net, obs, carac


def branch_keys(net: dict) -> list:
    """
    :param net: net grid file
    :return: global keys with transformers
    """
    line_keys = net[LINE][FROM_BUS].keys()
    trafo_keys = net[TRAFO][HV_BUS].keys()
    union_keys = list(line_keys)
    max_key_line = max(line_keys) + 1
    for k in trafo_keys:
        union_keys.append(k+max_key_line)
    return union_keys


def compute_bus_status(observation) -> dict:
    """
    Compute bus status from observation to only tell what to change in AgentAnswer methods

    :param observation: Network observation
    :return: Bus position of every element
    """
    bus_status = {BRANCH: {ORIGIN: {}, EXTREMITY: {}}, GEN: {}, LOAD: {}}
    for i in range(len(observation.line_status)):
        bus_status[BRANCH][ORIGIN][i] = observation.state_of(line_id=i)[ORIGIN][BUS]
        bus_status[BRANCH][EXTREMITY][i] = observation.state_of(line_id=i)[EXTREMITY][BUS]
    for i in range(len(observation.prod_p)):
        bus_status[GEN][i] = observation.state_of(gen_id=i)[BUS]
    for i in range(len(observation.load_p)):
        bus_status[LOAD][i] = observation.state_of(load_id=i)[BUS]
    return bus_status


def compute_delta_flow_and_charge(solution: dict, obs_p_or: list) -> None:
    """
    Computes delta between agent solution and observation in flow and in relative charge. Create new field in solution

    :param dict solution: Solution of optimisation problem
    :param list obs_p_or: Observed flows
    """
    delta = np.zeros(len(obs_p_or))
    self.logger.info('The solution variables contains')
    self.logger.info(solution)
    solution[DELTA_FLOW] = {i: 0 for i in solution[FLOW_B].keys()}
    solution[DELTA_CHARGE] = {i: 0 for i in solution[FLOW_B].keys()}
    for i in solution[FLOW_B].keys():
        delta[i] = abs(obs_p_or[i] - solution[FLOW_B][i])
        solution[DELTA_FLOW][i] = delta[i]
        solution[DELTA_CHARGE][i] = delta[i] / solution[MAX_FLOW][i]


def fixed_items_from_observation(observation, zone_to_fix: dict) -> dict:
    """
    Returns dict of variables values to fix from an observation. This function is used to restrict actions the agent can
    take. No actions will be performed on zone_to_fix elements.

    :param observation: Network observation
    :param dict zone_to_fix: Elements to fix. Of the from {ZONE_BRANCH: [...], ZONE_PROD: [...], ZONE_LOAD: [...]}
    :return: Variables values of fixed elements (to be set as hard constraints in future optimisation problem)
    """
    fixed = {SWITCH_B: {}, ETA_OR: {}, ETA_EX: {}, ETA_PROD: {}, ETA_LOAD: {}, MODIF_SUB: 0, MODIF_BRANCHES: 0}
    for line_id in zone_to_fix[ZONE_BRANCH]:
        fixed[SWITCH_B][line_id] = int(observation.line_status[line_id])
        fixed[ETA_OR][line_id] = observation.state_of(line_id=line_id)[ORIGIN][BUS] % 2
        fixed[ETA_EX][line_id] = observation.state_of(line_id=line_id)[EXTREMITY][BUS] % 2
    for prod_id in zone_to_fix[ZONE_PROD]:
        fixed[ETA_PROD][prod_id] = observation.state_of(gen_id=prod_id)[BUS] % 2
    for load_id in zone_to_fix[ZONE_LOAD]:
        fixed[ETA_LOAD][load_id] = observation.state_of(load_id=load_id)[BUS] % 2
    return fixed


def fixed_items_from_solution(solution: dict, zone_to_fix: dict) -> dict:
    """
    Same as for fixed_items_from_observation method but for solution instead of observation. Used only in double_level
    agent to keep track of global action from its solution

    :param dict solution: Solution of optimisation problem
    :param dict zone_to_fix: Elements to fix. Of the from {ZONE_BRANCH: [...], ZONE_PROD: [...], ZONE_LOAD: [...]}
    :return: Variables values of fixed elements (to be set as hard constraints in future optimisation problem)
    """
    fixed = {SWITCH_B: {}, ETA_OR: {}, ETA_EX: {}, ETA_PROD: {}, ETA_LOAD: {}, MODIF_SUB: 0}
    for line_id in zone_to_fix[ZONE_BRANCH]:
        fixed[SWITCH_B][line_id] = solution[SWITCH_B][line_id]
        fixed[ETA_OR][line_id] = solution[ETA_OR][line_id]
        fixed[ETA_EX][line_id] = solution[ETA_EX][line_id]
    for prod_id in zone_to_fix[ZONE_PROD]:
        fixed[ETA_PROD][prod_id] = solution[ETA_PROD][prod_id]
    for load_id in zone_to_fix[ZONE_LOAD]:
        fixed[ETA_LOAD][load_id] = solution[ETA_LOAD][load_id]
    fixed[MODIF_SUB] = sum([solution[SUB_CHANGED][key] for key in solution[SUB_CHANGED].keys()])
    return fixed


def distance_solution_observation(solution: dict, observation) -> int:
    """
    Returns the distance in number of switch actions between solution and observation

    :param dict solution: Solution of optimisation problem
    :param observation: Network observation
    :return: Number of switch actions performed
    """
    distance = 0
    for line_id in solution[SWITCH_B].keys():
        distance += int(observation.line_status[line_id] != solution[SWITCH_B][line_id])
    return distance
