# Copyright (c) 2020-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of MILP_Agent, The MILP_Agent is a optimization based agent that manage the power
# flow overthermal using topological actions.

import os 

import matplotlib.pyplot as plt
import pandas as pd 
import pandapower as pp
import numpy as np
from .logconf import logger_conf

from grid2op.Agent import DoNothingAgent
from grid2op.PlotGrid import PlotMatplot

from .global_var import GLOBAL_DOUBLE_LEVEL, MULTI_ZONES, MIP_SCIP, LINE, OBS, \
    SOL, ACT, DO_NOTHING, MAX_NUMBER_TIME_STEPS_TO_COMPLETE, GAMEOVER, TIME_STEPS_COMPLETED, \
    LINE_OVERFLOW, ACTIONS_TAKEN, PROD, SET_LINE_STATUS, CHANGE_LINE_STATUS, SET_BUS_VECT, CHANGE_BUS_VECT, \
    REDISPATCH, DATA, OUTPUT, ACTIONS, TIME, ID_OBJECT, ATTRIBUTE, VALUE, OBJ, SOLUTIONS, \
    V_OR, A_OR, P_OR, Q_OR, V_EX, A_EX, P_EX, Q_EX, RHO, LINE_STATUS, MAX_I_KA, X_BRANCH, TRAFO, SUB_OR_S, BUS_OR, \
    SUB_EX_S, BUS_EX, ORIGIN, BUS, EXTREMITY, MAX_FLOW, INPUT, CONSERVATIVE_LIMITS, OBSERVATIONS, EPISODE, \
    GLOBAL, DISCONNECTED, RECONNECTED, LINE_BUS_CHANGED, LOCAL, ZONE_BRANCH, CONNECTED_ID, DISCONNECTED_ID, \
    MODIF_SUB_ID, AGENT_LOG, EP_, OBSERVATIONS_NPZ, ACTIONS_NPZ, SOLUTIONS_NPZ, AGENT, GLOBAL_ACTION, LOCAL_ACTION, \
    RTE_CASE118_EXAMPLE, RES_CSV, INFO_NPZ

from .zone import Zones 
from .agent import MILPAgent


class Simulation:
    """
    Class used to simulate agent action on environment chronics of grid2op
    """
    def __init__(self, 
                 env, 
                 agent_solver: str, 
                 max_overflow_percentage, 
                 solver_name : str = MIP_SCIP,
                 clustering_path:str=None, 
                 zone_id: int = None,
                 zone_instance: dict = None, 
                 zone_level: int = 0, 
                 display_obs: bool = False, 
                 result_return:str=None, 
                 logger = None):
        """
        :param env : grip2op environment
        :param str agent_solver: Name of agent, this module support the following agents: GLOBAL_SWITCH
          (only line disconnection),
          GLOBAL_TOPOLOGY (all topological actions),GLOBAL_TOPO_ZONE (global topology with some fixed variables)
          GLOBAL_DOUBLE_LEVEL (two complementary agents),
          MULTI_ZONES (agent in each zone)
        :param max_overflow_percentage: Percentages of flow limit above which lines are considered in overflow
        :param solver_name : type of solver used to run the optimization problem. This module accept as solver:
          MIP_CBC, MIP_XPRESS, LINEAR_GLOP,LINEAR_XPRESS
        :param str clustering_path: the path towards the clustering file, it defines the zones that the
          multi_agent control
        :param int zone_id: Zone id from clustering file to build, it defines the id of zone that the double
          level agent control
        :param dict zone_instance: Zone instance, it defines the zone the double level agent control
        :param int zone_level: Radius of supplementary ring around the zone, used for both double level and multi
          agent controlors
        :param display_obs : Display network when self.agent takes an action
        :param str result_return : Path where the result of of simulation will be saved
        :param logger: logger to scree showin result of simulation
        
        """
        self.result_return = result_return

        input_path  = os.path.join(self.result_return, INPUT)
        output_path = os.path.join(self.result_return, OUTPUT)
        log_path    = os.path.join(self.result_return, AGENT_LOG)


        if not os.path.isdir(input_path):
            os.mkdir(input_path)

        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        if not os.path.isdir(log_path):
            os.mkdir(log_path)

        self.env = env
        self.env.parameters.MAX_SUB_CHANGED =\
            1 + int(agent_solver == GLOBAL_DOUBLE_LEVEL) + 10*int(agent_solver == MULTI_ZONES)
        self.env.parameters.MAX_LINE_STATUS_CHANGED =\
            1 + int(agent_solver == GLOBAL_DOUBLE_LEVEL) + 10*int(agent_solver == MULTI_ZONES)
        self.action_space = self.env.action_space
        self.path = self.env._init_grid_path
        self.net = pp.from_json(self.path)
        self.max_i_ka = self.net[LINE][MAX_I_KA]
        self.agent_solver = agent_solver
        self.solver_name = solver_name
        self.action_path = None
        self.observation_path = None
        self.solution_path = None
        self.parallel_path = {OBS: [], SOL: [], ACT: []}
        self.zone = None
        self.display_obs = display_obs
        self.max_overflow_percentage = max_overflow_percentage
        self.clustering_path = clustering_path
        if not os.path.exists(os.path.join(self.result_return, INPUT, CONSERVATIVE_LIMITS+".npz")):
            self.concervative_thremal_set()
        if zone_instance is not None:  # One can directly pass a dict as a zone instance
            self.zone = zone_instance
        elif zone_id is not None:  # One can use the existing clustering to define a zone with its ID
            self.zones = Zones(self.env.get_obs(), self.clustering_path)
            self.zone = self.zones.zone_ring(zone=self.zones.clustering[zone_id], level=zone_level)
        if agent_solver == DO_NOTHING:
            self.agent = DoNothingAgent(self.action_space)
        else:
            self.agent = MILPAgent(self.env, agent_solver,
                                   max_overflow_percentage,
                                   solver_name=self.solver_name,
                                   zone_instance=self.zone,
                                   clustering_path=self.clustering_path,
                                   zone_level=zone_level)

        if logger is None:
            self.logger = logger_conf()
        else:
            self.logger = logger

    def running(self, max_iter: int, episode_index: int = 0, fast_forward: int = 0,
                save_results: bool = False) -> dict:
        """
        Run simulation of episode

        :param int max_iter: Maximal number of iterations
        :param str save_folder: Folder to save results to
        :param int episode_index: Index of episode in the chronics folder
        :param int fast_forward: Time step to start on
        :param bool save_results: True to save results
        :return dict results: Results of the simulation
        """
        # Preparing environment
        self.env.seed(0)
        self.env.chronics_handler.tell_id(id_num=episode_index)
        self.env.reset()
        if fast_forward != 0:
            self.env.chronics_handler.fast_forward(fast_forward)
        obs = self.env.get_obs()
        game_over = False
        time_step = 0
        reward = 0
        all_obs = []
        all_action = []
        all_solution = []
        line_overflow = {}
        action_type_taken = {}
        all_obs.append(obs.to_vect())
        self.logger.info((MAX_NUMBER_TIME_STEPS_TO_COMPLETE, self.env.chronics_handler.max_timestep() - fast_forward))
        while not game_over and time_step < min(self.env.chronics_handler.max_timestep(), max_iter):  # Run starts
            time_step += 1
            # if time_step == 605:
            #     self.display_observation(observation=obs)
            old_p_or = obs.p_or
            line_overflow[time_step] = {}
            action_type_taken[time_step] = {}
            for r in range(len(obs.rho)):  # To keep track of lines overloaded
                if obs.rho[r] > self.max_overflow_percentage[r]:
                    line_overflow[time_step][r] = 100*obs.rho[r]
            action = self.agent.act(obs, reward, game_over)  # Agent action
            assert self.action_space._is_legal(action, self.env)
            action_type_taken[time_step] = self.readable_action_from_dict(action.as_dict())
            if self.agent_solver != DO_NOTHING and save_results:
                all_solution.append(self.agent.solution)
            obs, reward, game_over, info = self.env.step(action)  # Apply action to environment
            if action != self.action_space({}) and self.display_obs:
                self.display_observation(observation=obs)
            if save_results:
                all_obs.append(obs.to_vect())
                all_action.append(action.to_vect())
        if save_results:  
            self.save_to_npz_format(episode_index=episode_index, all_obs=all_obs,
                                    all_action=all_action, all_solution=all_solution)
            self.save_info_npz_format(info,episode_index=episode_index)
        return {GAMEOVER: game_over, TIME_STEPS_COMPLETED: time_step - int(game_over), LINE_OVERFLOW: line_overflow,
                ACTIONS_TAKEN: action_type_taken}

    def write_actions(self, output: bool = False, write: bool = True):
        """
        Write actions taken by the optimization agent

        :param bool output: returns the panda data frame if set to true
        :param bool write: write results in actions.csv output file
        """
        base_action = self.env._helper_action_player({})
        actions = np.load(self.action_path)
        col = [PROD+"_p", SET_LINE_STATUS, CHANGE_LINE_STATUS, SET_BUS_VECT, CHANGE_BUS_VECT, REDISPATCH]
        lines = []
        for action_as_vector in actions[DATA]:
            line = []
            if np.isnan(action_as_vector).any():
                break
            else:
                base_action.from_vect(action_as_vector)
                action_dict = base_action.as_dict()
            keys = list(action_dict.keys())
            for field in col:
                if field in keys:
                    line.append(action_dict[field])
                else:
                    line.append([])
            lines.append(line)
        data = pd.DataFrame(lines, columns=col)
        if write:
            path = os.path.join(self.result_return, OUTPUT, ACTIONS)
            data.to_csv(path, sep=";", index=False)
        if output:
            return data

    def write_solutions(self, output: bool = False, write: bool = True):
        """
        Writes solution returned by the optimization agent

        :param bool output: returns the panda data frame if set to true
        :param bool write: write solutions in solutions.csv output file
        """
        if self.solution_path is not None:
            data = np.load(self.solution_path, allow_pickle=True)
            all_solution = data[DATA]
            col = [TIME, ID_OBJECT, ATTRIBUTE, VALUE]
            single_valued_fields = [OBJ]
            time_step = 0
            lines = []
            for solution in all_solution:
                time_step += 1
                for field in solution.keys():
                    if field in single_valued_fields:
                        lines.append([time_step, 0, field, solution[field]])
                    else:
                        for i in solution[field].keys():
                            lines.append([time_step, i, field, solution[field][i]])
            data = pd.DataFrame(lines, columns=col)
            if write:
                path = os.path.join(self.result_return, OUTPUT, SOLUTIONS)
                data.to_csv(path, sep=";", index=False)
            if output:
                return data

    def write_observations(self, output: bool = False, write: bool = True):
        """
        Writes observations generated by grid2op of the environment

        :param bool output: returns the panda data frame if set to true
        :param bool write: write observations in observations.csv output file
        """
        basic_obs = self.env.reset()
        observations = np.load(self.observation_path)
        lines = []
        col = [TIME, ID_OBJECT, SUB_OR_S, BUS_OR, SUB_EX_S, BUS_EX, ATTRIBUTE, VALUE]
        fields_name = [V_OR, A_OR, P_OR, Q_OR, V_EX, A_EX, P_EX, Q_EX, RHO, LINE_STATUS, MAX_I_KA, X_BRANCH]
        if self.agent_solver == DO_NOTHING:
            fields_name.remove(X_BRANCH)
            fields_name.remove(MAX_I_KA)
        elif len(self.max_i_ka) < len(self.net[LINE]) + len(self.net[TRAFO]):
            fields_name.remove(MAX_I_KA)
        timestep = 0
        if self.agent_solver != DO_NOTHING:
            self.max_i_ka = self.agent.carac[MAX_I_KA]
        for obs in observations[DATA]:
            if np.isnan(obs).any():
                break
            basic_obs.from_vect(obs)
            obs_dict = basic_obs.to_dict()
            obs_dict[V_OR] = basic_obs.v_or
            obs_dict[V_EX] = basic_obs.v_ex
            obs_dict[A_OR] = basic_obs.a_or
            obs_dict[A_EX] = basic_obs.a_ex
            obs_dict[P_OR] = basic_obs.p_or
            obs_dict[P_EX] = basic_obs.p_ex
            obs_dict[Q_OR] = basic_obs.q_or
            obs_dict[Q_EX] = basic_obs.q_ex
            obs_dict[MAX_I_KA] = self.max_i_ka
            if self.agent_solver != DO_NOTHING:
                obs_dict[X_BRANCH] = self.agent.carac[X_BRANCH]

            for branch_id in range(len(basic_obs.v_or)):
                sub_or = basic_obs.line_or_to_subid[branch_id]
                sub_ex = basic_obs.line_ex_to_subid[branch_id]
                bus_or = basic_obs.state_of(line_id=branch_id)[ORIGIN][BUS]
                bus_ex = basic_obs.state_of(line_id=branch_id)[EXTREMITY][BUS]
                for attribute in fields_name:
                    lines.append([timestep, branch_id, sub_or, bus_or, sub_ex, bus_ex, attribute,
                                  obs_dict[attribute][branch_id]])
            timestep += 1
        data = pd.DataFrame(lines, columns=col)
        path = os.path.join(self.result_return, OUTPUT, OBSERVATIONS)
        if write:
            data.to_csv(path, sep=";", index=False)
        if output:
            return data

    def write_results(self) -> None:
        """
        Writes observations, actions and solution in csv files
        """
        self.write_observations()
        self.write_actions()
        self.write_solutions()

    def concervative_thremal_set(self):
        '''
        Save the lines thremal limit of the grid as .npz file in input folder
        '''
        thremal_limit = self.env.get_thermal_limit()
        np.savez(os.path.join(self.result_return, INPUT, CONSERVATIVE_LIMITS+".npz"), **{DATA: thremal_limit})

    def find_conservative_limit(self):
        """
        Takes the minimal encountered limit for flows in observations
        Needed for the fprecedent simulation to generate observations (with agents DoNothing or StateEstimation)
        Only run once to have the limits for each network
        """
        data = pd.read_csv(os.path.join(self.result_return, OUTPUT , DO_NOTHING, OBSERVATIONS), sep=";")
        max_flow_matrix = {}
        for i in range(len(data[ATTRIBUTE])):
            if data[ATTRIBUTE][i] == MAX_FLOW:
                id_object = data[ID_OBJECT][i]
                if id_object in max_flow_matrix.keys():
                    max_flow_matrix[id_object] = min(max_flow_matrix[id_object], data[VALUE][i])
                else:
                    max_flow_matrix[id_object] = data[VALUE][i]
        nump = np.array([max_flow_matrix[i] for i in max_flow_matrix.keys()])
        np.savez(os.path.join(self.result_return, INPUT, CONSERVATIVE_LIMITS+".npz"), **{DATA: nump})

    def run_episodes(self, max_iter: int, episodes: dict,
                     save_results: bool = True) -> dict:
        """
        Main method of the Simulation class. Runs episodes whose indexes are keys of episodes.
        If save_results is set to true, the results will be saved in the sub-folder of the output folder with the name
        of agent used for simulations

        :param int max_iter: Maximal number of iterations to perform
        :param dict episodes: Episodes to run. Specify the time of start and index of episode. Takes the form
          {ep_id: start_time}
        :param str save_folder: Folder to save npz results to
        :param bool save_results: Save results
        :return: Track-back of simulations with time steps completed and lines overflow observed for each episode
        """
        gameover = {}
        if save_results:
            obs_concat_list = []
            sol_concat_list = []
            act_concat_list = []
            for episode in list(episodes.keys()):
                res = self.running(max_iter=max_iter, episode_index=episode,
                                   fast_forward=episodes[episode], save_results=True)
                gameover[episode] = res
                obs_data = self.write_observations(output=True, write=False)
                obs_data[EPISODE] = [episode for _ in range(len(obs_data))]
                obs_concat_list.append(obs_data)
                if self.agent_solver != DO_NOTHING:
                    sol_data = self.write_solutions(output=True, write=False)
                    sol_data[EPISODE] = [episode for _ in range(len(sol_data))]
                    sol_concat_list.append(sol_data)
                    act_data = self.write_actions(output=True, write=False)
                    act_data[EPISODE] = [episode for _ in range(len(act_data))]
                    act_concat_list.append(act_data)
            obs_final_data = pd.concat(obs_concat_list)
            folder_path = os.path.join(self.result_return, OUTPUT, self.agent_solver)
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
            obs_path = os.path.join(self.result_return, OUTPUT, self.agent_solver, OBSERVATIONS)
            obs_final_data.to_csv(obs_path, sep=";", index=False)
            if self.agent_solver != DO_NOTHING:
                sol_final_data = pd.concat(sol_concat_list)
                sol_path = os.path.join(self.result_return, OUTPUT, self.agent_solver, SOLUTIONS)
                sol_final_data.to_csv(sol_path, sep=";", index=False)
                act_final_data = pd.concat(act_concat_list)
                act_path = os.path.join(self.result_return, OUTPUT, self.agent_solver, ACTIONS)
                act_final_data.to_csv(act_path, sep=";", index=False)
        else:
            for episode in list(episodes.keys()):
                res = self.running(max_iter, episode_index=episode,
                                   fast_forward=episodes[episode], save_results=False)
                gameover[episode] = res
        return gameover

    def readable_action_from_dict(self, action_dict: dict) -> dict:
        """
        More readable format for grid2op actions used in the track-back output of run method

        :param dict action_dict: grid2op actions converted as dict
        :return: Human readable action dict
        """
        readable_action = {GLOBAL: {DISCONNECTED: [], RECONNECTED: [], LINE_BUS_CHANGED: []},
                           LOCAL: {DISCONNECTED: [], RECONNECTED: [], LINE_BUS_CHANGED: []}}
        zone_branch = [i for i in range(len(self.max_overflow_percentage))]
        if self.zone is not None:
            zone_branch = self.zone[ZONE_BRANCH]
        if SET_LINE_STATUS in list(action_dict.keys()):
            for line_id in action_dict[SET_LINE_STATUS][CONNECTED_ID]:
                if int(line_id) in zone_branch:
                    readable_action[LOCAL][RECONNECTED].append(line_id)
                else:
                    readable_action[GLOBAL][RECONNECTED].append(line_id)
            for line_id in action_dict[SET_LINE_STATUS][DISCONNECTED_ID]:
                if int(line_id) in zone_branch:
                    readable_action[LOCAL][DISCONNECTED].append(line_id)
                else:
                    readable_action[GLOBAL][DISCONNECTED].append(line_id)
        if CHANGE_BUS_VECT in list(action_dict.keys()):
            changed_lines = []
            for sub_id in action_dict[CHANGE_BUS_VECT][MODIF_SUB_ID]:
                for line_id in action_dict[CHANGE_BUS_VECT][sub_id].keys():
                    changed_lines.append([line_id, sub_id])
            for item in changed_lines:
                if int(item[0]) in zone_branch:
                    readable_action[LOCAL][LINE_BUS_CHANGED].append(item)
                else:
                    readable_action[GLOBAL][LINE_BUS_CHANGED].append(item)
        return readable_action

    def display_observation(self, observation):
        """
        Displays observation of the environment. Very useful to see the current state of the network

        :param observation: Grid2op observation to display
        """
        plot_helper = PlotMatplot(self.env.observation_space)
        plot_helper.plot_obs(observation, line_info=RHO, load_info=None, gen_info=None)
        plt.plot()
        plt.show()

    def save_to_npz_format(self, episode_index: int,
                           all_obs: list, all_action: list, all_solution: list):
        """
        Saves observations, actions and solution to npz format

        :param str save_folder: Folder to save results to
        :param int episode_index: Index of episode
        :param list all_obs: Observations for this episode
        :param list all_action: Actions for this episode
        :param list all_solution: Solutions for this episode
        """
        self.observation_path = os.path.join(self.result_return, AGENT_LOG, EP_+str(episode_index)+"_"+OBSERVATIONS_NPZ)
        np.savez(self.observation_path, **{DATA: all_obs})
        self.action_path = os.path.join(self.result_return, AGENT_LOG, EP_+str(episode_index)+"_"+ACTIONS_NPZ)
        np.savez(self.action_path, **{DATA: all_action})
        if self.agent_solver != DO_NOTHING:
            self.solution_path = os.path.join(self.result_return, AGENT_LOG, EP_+str(episode_index)+"_"+SOLUTIONS_NPZ)
            np.savez(self.solution_path, **{DATA: all_solution})
            
    def save_info_npz_format(self,info:dict,episode_index:int):
        '''
        In order to analyse the occurent blackout, this function save the info which correspending to 
        dict containing information about the events.
        :param dict info: infomation of the current blackout
        '''
        self.info_path = os.path.join(self.result_return,AGENT_LOG, EP_+str(episode_index)+"_" + INFO_NPZ)
        np.savez(self.info_path, info)

########################################################################################################################
    # Auxiliary functions
########################################################################################################################


def compare_agents_on_episode(env, agents: dict, solver_name, clustering_path, zone_instance, zone_level, episodes: dict,
                                 save_results: bool = False, display_obs: bool = False)-> None:
    """
    Compare performances of agents on episodes. Will save results in res.csv file in output folder

    :param dict agents: Agents to compare {agent_name: zone_id}
    :param str agent_solver: Name of agent, this module support the following agents: GLOBAL_SWITCH
          (only line disconnection),
          GLOBAL_TOPOLOGY (all topological actions),GLOBAL_TOPO_ZONE (global topology with some fixed variables)
          GLOBAL_DOUBLE_LEVEL (two complementary agents),
          MULTI_ZONES (agent in each zone)
        :param max_overflow_percentage: Percentages of flow limit above which lines are considered in overflow
        :param solver_name : type of solver used to run the optimization problem. This module accept as solver:
          MIP_CBC, MIP_XPRESS, LINEAR_GLOP,LINEAR_XPRESS
        :param str clustering_path: the path towards the clustering file, it defines the zones that the
          multi_agent control
        :param int zone_id: Zone id from clustering file to build, it defines the id of zone that the double
          level agent control
        :param dict zone_instance: Zone instance, it defines the zone the double level agent control
        :param int zone_level: Radius of supplementary ring around the zone, used for both double level and multi
          agent controlors
    :param dict episodes: Episodes to run of the form {episode_id: start_time}
    :param save_results: Save results
    :param display_obs: Display observation at each action of agents
    """
    col = [EPISODE, TIME, AGENT, LINE_OVERFLOW, GLOBAL_ACTION, LOCAL_ACTION]
    lines = []
    for agent in agents.keys():
        simulation = Simulation(env, agent, max_overflow_percentage=.95 * np.ones(186),solver_name = solver_name,
                                clustering_path= clustering_path, zone_id=agents[agent], zone_instance=zone_instance, 
                                display_obs=display_obs, zone_level=zone_level)
        result = simulation.run_episodes(max_iter=30, episodes=episodes, save_results=save_results)
        self.logger.info(result)
        for episode_id in result.keys():
            res = result[episode_id]
            for timestep in res[LINE_OVERFLOW].keys():
                lines.append([episode_id, timestep, agent, res[LINE_OVERFLOW][timestep],
                              res[ACTIONS_TAKEN][timestep][GLOBAL], res[ACTIONS_TAKEN][timestep][LOCAL]])
    data = pd.DataFrame(lines, columns=col)
    path = os.path.join(self.result_return, OUTPUT, RES_CSV)
    data.to_csv(path, sep=";", index=False)
