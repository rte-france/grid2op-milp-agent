# Copyright (c) 2020-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of MILP_Agent, The MILP_Agent is a optimization based agent that manage the power flow
# overthermal using topological actions.

import os
import  numpy as np
from .logconf import logger_conf

from .isolver import ISolver
from .global_var import MIP_SCIP, NB_BRANCH, NB_BUS, INPUT, DATA, CONSERVATIVE_LIMITS, \
    ZONE_BRANCH, ZONE_BUS, ZONE_PROD, ZONE_LOAD, X_BRANCH, SHIFT, MAX_I_KA, BUILDER_CONSTRAINT_LIST, \
    ID_OBJECT, V_OR, A_OR, RHO, Q_OR, P_OR, MAX_FLOW, SOLUTION, QUAD_NODE, U1, \
    U2, FLOW_B_MINUS_P_OR, FLOW_B, QUAD_FLOW, ORIGIN, BUS, EXTREMITY, SLACK_N_PLUS, SLACK_N_MINUS, \
    SLACK_B, BRANCH, LINES_OR_ID, LINES_EX_ID, GENERATORS_ID, LOADS_ID, PROD, SOLVER_CHECK_WRONG, SWITCH_B, \
    SUB_CHANGED, SUB_ACTIONS, ETA_OR, ETA_EX, ETA_PROD, ETA_LOAD, THETA, DELTA_THETA_B, \
    FLOW_B_PLUS, FLOW_B_MINUS, OR, EX, ETA, LOAD, FMAX


class Builder:
    """
    Build the opimization submodules (constrains, cost function, solution verification..).
    """
    def __init__(self,
                 obs, 
                 carac, 
                 max_overflow_percentage, 
                 solver_name=MIP_SCIP, 
                 zone_instance=None, 
                 logger=None):
        """
        :param obs: Current observation of the environnement.
        :param dict carac: Network characteristics.
        :param max_overflow_percentage: Percentage of flow limit from which the line is considered overloaded.
        :param str solver_name: Name of solver used.
        :param dict zone_instance: Zone instance (if any).
        :param logger: logger to scree showin result of simulation
        """
        self.obs = obs
        self.carac = carac
        self.branches = [i for i in range(carac[NB_BRANCH])]
        self.subs = [i for i in range(carac[NB_BUS])]
        self.prods = [i for i in range(len(obs.prod_p))]
        self.loads = [i for i in range(len(obs.load_p))]
        conservative_limits_path = os.path.join(INPUT, CONSERVATIVE_LIMITS + ".npz") 
        try:
            self.conservative_limits = np.load(conservative_limits_path)[DATA]
        except FileNotFoundError:
            self.conservative_limits =  self.obs.rho * self.obs.a_or
        if zone_instance is not None:
            self.zone = zone_instance
            self.branches = self.zone[ZONE_BRANCH]
            self.subs = self.zone[ZONE_BUS]
            self.prods = self.zone[ZONE_PROD]
            self.loads = self.zone[ZONE_LOAD]
        self.nb_branch = len(self.branches)
        self.nb_bus = len(self.subs)
        self.X_branch = carac[X_BRANCH]
        self.shift = carac[SHIFT]
        self.i_max_ka = carac[MAX_I_KA]
        self.F_max_b = np.ones(len(max_overflow_percentage))
        self.new_flow_limit()  # new flow limits
        self.M = 1e3
        self.solver = ISolver(solver_name)
        self.constraints_dict = {var: False for var in BUILDER_CONSTRAINT_LIST}
        self.solved = False
        self.solution = None
        self.solver_name = solver_name
        self.line_action_limitation = self.nb_branch
        self.max_overflow_percentage = max_overflow_percentage
        self.F_max_b = self.F_max_b * self.max_overflow_percentage
        self.nodes_balance = {i: 0 for i in self.subs}
        self.nodes_balance_by_bus2 = {1: {i: 0 for i in self.subs}, 2: {i: 0 for i in self.subs}}
        self.nb_prod = len(self.prods)
        self.nb_load = len(self.loads)
        self.forbidden_branches_id = np.nonzero(self.obs.time_before_cooldown_line)[0]
        self.forbidden_sub_id = np.nonzero(self.obs.time_before_cooldown_sub)[0]
        
        if logger is None:
            self.logger = logger_conf()
        else:
            self.logger = logger

    def old_flow_limit(self) -> None:
        """
        Updates self.F_max_b with old power limit calculation
        """
        for i in self.branches:
            if self.obs.v_or[i] * self.obs.a_or[i] != 0:
                p_or = self.obs.p_or[i]
                p_ex = self.obs.p_ex[i]
                if min(abs(p_or), abs(p_ex)) < 1e-5:
                    power = max(abs(p_or), abs(p_ex))
                    if abs(p_or) < abs(p_ex):
                        self.F_max_b[i] = self.i_max_ka[i] * self.obs.v_ex[i] * 1000 * power / (
                            self.obs.v_ex[i] * self.obs.a_ex[i])
                    else:
                        self.F_max_b[i] = self.i_max_ka[i] * self.obs.v_or[i] * 1000 * power / (
                            self.obs.v_or[i] * self.obs.a_or[i])
                else:
                    self.F_max_b[i] = self.i_max_ka[i] * self.obs.v_or[i] * 1000 * abs(self.obs.p_or[i]) / (
                            self.obs.v_or[i] * self.obs.a_or[i])
            else:
                self.F_max_b[i] = float(self.conservative_limits[i])

    def new_flow_limit(self) -> None:
        """
        Updates self.F_max_b with new power limit calculation. Uses the predefined power limit if the line is
        disconnected.
        """
        for i in self.branches:
            if self.obs.line_status[i]:
                max_flow_square = 3*(self.obs.v_or[i]*self.i_max_ka[i])**2 - self.obs.q_or[i]**2
                if max_flow_square >= 0:
                    self.F_max_b[i] = np.sqrt(max_flow_square)
                else:
                    self.logger.warning("Negative term in square root for limit of line")
                    self.logger.info(([ID_OBJECT, V_OR, A_OR, RHO, MAX_I_KA, Q_OR, P_OR],
                          [i, self.obs.v_or[i], self.obs.a_or[i], self.obs.rho[i], self.i_max_ka[i],
                           self.obs.q_or[i], self.obs.p_or[i]]))
                    self.F_max_b[i] = abs(self.obs.p_or[i]) / self.obs.rho[i]
            else:
                self.F_max_b[i] = float(self.conservative_limits[i])

    ################################################################################################
    # General methods for Builder object to communicate with OrTools solver
    ################################################################################################

    def activate_const_list(self, const_name_list: list) -> None:
        """
        Activation of the constraints for the optimization problem

        :param list const_name_list: Types of constraints to activate
        """
        for const_name in const_name_list:
            self.constraints_dict[const_name] = True

    def var(self, category: str, var_id: int):
        """
        Make the solver variable by category and index. Example of use self.var(category=SWITCH_B, var_id=id_branch)

        :param str category: Category of the variable
        :param int var_id: Index of variable
        :return: Solver var_id variable of category category
        """
        return self.solver.variables[category][var_id]

    def display_problem(self) -> None:
        """
        Display the optimization problem

        :return: None
        """
        self.solver.display()

    def solve_status(self) -> bool:
        """
        Solve self.solver optimisation problem and update the status

        :return: New status of optimisation problem
        """
        self.solved = self.solver.solve()
        return self.solved

    def get_solution(self, display: bool = True) -> dict:
        """
        Returns solution of the optimisation problem as a dict of values with categories keys

        :param bool display: Print solutions on return
        :return: Solution dict
        """
        assert self.solved
        self.solution = self.solver.solution_as_dict()
        self.solution[MAX_FLOW] = {i: self.F_max_b[i] for i in self.branches}
        if display:
            self.logger.info((SOLUTION, self.solution))
        return self.solution

    ################################################################################################################
    # State estimation problem : Introduction of quadratic variables and tangents constraints for Linear programming
    ################################################################################################################

    def add_quad_var(self) -> None:
        """
        Creates quadratic variables for the state estimation problem

        :return: None
        """
        # Quadratic slack nodes
        for i in self.subs:
            self.solver.create_num_var(0, self.solver.infinity(), QUAD_NODE, i)
            self.solver.create_num_var(0, self.solver.infinity(), QUAD_NODE+U1, i)
            self.solver.create_num_var(0, self.solver.infinity(), QUAD_NODE+U2, i)
        # Quadratic flow
        for i in self.branches:
            self.solver.create_num_var(-self.solver.infinity(), self.solver.infinity(), FLOW_B_MINUS_P_OR, i)
            self.solver.add_constraint(self.var(FLOW_B_MINUS_P_OR, i)
                                       + (self.obs.p_or[i] - self.var(FLOW_B, i)) / self.F_max_b[i] == 0)
            self.solver.create_num_var(0, self.solver.infinity(), QUAD_FLOW, i)

    def balance_nodes_for_state_estimation(self) -> None:
        """
        Pre-distribution of branch losses. The loss of each branch is devided equitably into two buses of the origin and extremity.

        :return: None
        """
        balance = {1: {i: 0 for i in self.subs}, 2: {i: 0 for i in self.subs}}
        for line_id in self.branches:  # Loss repartition between origin and extremity
            if self.obs.line_status[line_id]:
                sub_or = self.obs.line_or_to_subid[line_id]
                sub_ex = self.obs.line_ex_to_subid[line_id]
                bus_or = self.obs.state_of(line_id=line_id)[ORIGIN][BUS]
                bus_ex = self.obs.state_of(line_id=line_id)[EXTREMITY][BUS]
                balance[bus_ex][sub_ex] -= (self.obs.p_or[line_id] + self.obs.p_ex[line_id]) / 2
                balance[bus_or][sub_or] -= (self.obs.p_or[line_id] + self.obs.p_ex[line_id]) / 2
        self.nodes_balance_by_bus2 = balance

    def add_tangent_constraints(self, variable, quad_var, points) -> None:
        """
        Add tangent constraints to linearize quadratics

        :param variable: Original variable to be squared
        :param quad_var: Linearized quadratic variable
        :param points: Tangent points
        :return: None
        """
        for point in points:
            self.solver.add_constraint(quad_var >= 2 * point * variable - point ** 2)

    def add_linearised_constraints(self) -> None:
        """
        Add linearized quadratic constraints.

        :return: None
        """
        # Quadratic nodes
        points_nodes = np.linspace(start=-50, stop=50, num=20)
        for i in self.subs:
            for ext in [U1, U2]:
                variable = self.var(SLACK_N_PLUS+ext, i) - self.var(SLACK_N_MINUS+ext, i)
                quad_var = self.var(QUAD_NODE+ext, i)
                self.add_tangent_constraints(variable=variable, quad_var=quad_var, points=points_nodes)
        # Quadratic flows
        points_flow = np.linspace(start=-1, stop=1, num=20)
        for i in self.branches:
            variable = self.var(FLOW_B_MINUS_P_OR, i)
            quad_var = self.var(QUAD_FLOW, i)
            self.add_tangent_constraints(variable=variable, quad_var=quad_var, points=points_flow)

    def add_overflow_constraints_for_state_estimation(self) -> None:
        """
        Overloaded lines in observations should appear overloaded at the end of state estimation

        :return: None
        """
        for line_id in self.branches:
            if self.obs.rho[line_id] >= self.max_overflow_percentage[line_id]:
                if self.obs.p_or[line_id] > 0:
                    self.solver.add_constraint(self.var(FLOW_B, line_id) >= self.F_max_b[line_id])
                else:
                    self.solver.add_constraint(-self.var(FLOW_B, line_id) >= self.F_max_b[line_id])

    def check_false_negative_state_estimation(self):
        """
        Check for false negative constraints

        :return: List of false negative branches that are overloaded after finding the optimal solution
        """
        assert self.solved
        false_negative = []
        for branch_id in self.branches:
            if self.obs.rho[branch_id] > self.max_overflow_percentage[branch_id]:
                check = bool(abs(self.solution[FLOW_B][branch_id]) - self.F_max_b[branch_id] >= 0)
                if not check:
                    self.logger.warning("FALSE NEGATIVE : BRANCH ", branch_id, " SHOULD APPEAR OVERLOADED ")
                    false_negative.append(branch_id)
        return false_negative

    def set_quad_objective(self) -> None:
        """
        Define the objective function for quadratic state estimation problem

        :return: None
        """
        self.solver.create_objective()
        # Quadratic slack nodes
        for i in self.subs:
            self.solver.set_objective_coef(self.var(QUAD_NODE + U1, i), 100)
            self.solver.set_objective_coef(self.var(QUAD_NODE + U2, i), 100)
        # Quadratic slack flow
        for i in self.branches:
            self.solver.set_objective_coef(self.var(QUAD_FLOW, i), 100)
        self.solver.set_minimization()

    ################################################################################################
    # Limitation and solution check methods
    ################################################################################################

    def check_branch_const(self):
        """
        Displays overloaded branches after finding the solution

        :return: None
        """
        assert self.solved
        for branch_id in self.branches:
            if self.constraints_dict[SLACK_B] and self.solution[SLACK_B][branch_id] > 0:
                self.logger.info(("Slack on branch " + str(branch_id) + " : ", self.solution[SLACK_B][branch_id]))
            flow = self.solution[FLOW_B][branch_id]
            if abs(flow) > self.F_max_b[branch_id]:
                self.logger.info((BRANCH + str(branch_id) + " overloaded : ", abs(flow), "/", self.F_max_b[branch_id]))
                self.logger.info((f"line {branch_id} should be still overloaded after solution {abs(flow):.2f} / {self.F_max_b[branch_id]:.2f}"))

    def check_bus_const(self, eps: float):
        """
        Displays nodes for which slack is greater than eps

        :param float eps: Precision
        :return: None
        """
        assert self.solved
        for sub_id in self.subs:
            connected = self.obs.get_obj_connect_to(substation_id=sub_id)
            lines_or = list(set(connected[LINES_OR_ID]) & set(self.branches))
            lines_ex = list(set(connected[LINES_EX_ID]) & set(self.branches))
            prods = list(set(connected[GENERATORS_ID]) & set(self.prods))
            loads = list(set(connected[LOADS_ID]) & set(self.loads))
            slack_node = (sum([self.solution[FLOW_B][j] for j in lines_or])
                          + sum([self.obs.load_p[j] for j in loads])
                          - sum([self.solution[PROD][j] for j in prods])
                          - sum([self.solution[FLOW_B][j] for j in lines_ex])
                          - self.nodes_balance[sub_id])
            if abs(slack_node) > eps:
                self.logger.info(("Slack node " + str(sub_id) + " : ", slack_node))

    def check_solution(self):
        """
        Verification of the solution, used to show if the solver has made an errors.

        :return: None
        """
        eps = 1e-3
        self.check_branch_const()
        self.check_bus_const(eps)
        if not self.solver.solver.VerifySolution(tolerance=1e-1, log_errors=True):
            self.logger.error(SOLVER_CHECK_WRONG)

    def limit_switch_actions(self, nb_line_actions: int) -> None:
        """
        Limit the number of line status changes

        :param int nb_line_actions: Maximal number of line status changes.
        """
        if self.constraints_dict[SWITCH_B]:
            status = self.obs.line_status
            self.solver.add_constraint((sum([(1 - int(status[branch_id])) * self.var(SWITCH_B, branch_id)
                                            for branch_id in self.branches])
                                        + sum([int(status[branch_id]) * (1 - self.var(SWITCH_B, branch_id))
                                               for branch_id in self.branches])
                                        <= nb_line_actions))

    def limit_number_of_substations_changed(self, nb_sub_actions: int) -> None:
        """
        Limit the number of substations on which can act

        :param int nb_sub_actions: Maximal number of substations whose topology can be modified
        """
        for sub_id in self.subs:
            self.solver.create_int_var(0, 1, SUB_CHANGED, sub_id)
            self.solver.create_int_var(0, 100, SUB_ACTIONS, sub_id)
            lines_or = self.obs.get_obj_connect_to(substation_id=sub_id)[LINES_OR_ID]
            lines_ex = self.obs.get_obj_connect_to(substation_id=sub_id)[LINES_EX_ID]
            prods = self.obs.get_obj_connect_to(substation_id=sub_id)[GENERATORS_ID]
            loads = self.obs.get_obj_connect_to(substation_id=sub_id)[LOADS_ID]
            lines_or_bus_2 = []
            lines_or_bus_1 = []
            lines_ex_bus_2 = []
            lines_ex_bus_1 = []
            prod_bus_1 = []
            prod_bus_2 = []
            load_bus_1 = []
            load_bus_2 = []
            for line_id in lines_or:
                if self.obs.line_status[line_id]:
                    if self.obs.state_of(line_id=line_id)[ORIGIN][BUS] > 1.5:
                        lines_or_bus_2.append(line_id)
                    else:
                        lines_or_bus_1.append(line_id)
            for line_id in lines_ex:
                if self.obs.line_status[line_id]:
                    if self.obs.state_of(line_id=line_id)[EXTREMITY][BUS] > 1.5:
                        lines_ex_bus_2.append(line_id)
                    else:
                        lines_ex_bus_1.append(line_id)
            for prod_id in prods:
                if self.obs.state_of(gen_id=prod_id)[BUS] > 1.5:
                    prod_bus_2.append(prod_id)
                else:
                    prod_bus_1.append(prod_id)
            for load_id in loads:
                if self.obs.state_of(load_id=load_id)[BUS] > 1.5:
                    load_bus_2.append(load_id)
                else:
                    load_bus_1.append(load_id)
            self.solver.add_constraint(self.var(SUB_ACTIONS, sub_id) ==
                                       sum([self.var(ETA_OR, j) for j in lines_or_bus_2])
                                       + sum([1-self.var(ETA_OR, j) for j in lines_or_bus_1])
                                       + sum([self.var(ETA_EX, j) for j in lines_ex_bus_2])
                                       + sum([1-self.var(ETA_EX, j) for j in lines_ex_bus_1])
                                       + sum([self.var(ETA_PROD, j) for j in prod_bus_2])
                                       + sum([1-self.var(ETA_PROD, j) for j in prod_bus_1])
                                       + sum([self.var(ETA_LOAD, j) for j in load_bus_2])
                                       + sum([1-self.var(ETA_LOAD, j) for j in load_bus_1]))
            if sub_id in self.forbidden_sub_id:
                for j in lines_or:
                    self.solver.add_constraint(self.var(ETA_OR, j) ==
                                               1 - int(self.obs.state_of(line_id=j)[ORIGIN][BUS] > 1.5))
                for j in lines_ex:
                    self.solver.add_constraint(self.var(ETA_EX, j) ==
                                               1 - int(self.obs.state_of(line_id=j)[EXTREMITY][BUS] > 1.5))
                for j in prods:
                    self.solver.add_constraint(self.var(ETA_PROD, j) ==
                                               1 - int(self.obs.state_of(gen_id=j)[BUS] > 1.5))
                for j in loads:
                    self.solver.add_constraint(self.var(ETA_LOAD, j) ==
                                               1 - int(self.obs.state_of(load_id=j)[BUS] > 1.5))

            self.solver.add_constraint(self.var(SUB_CHANGED, sub_id) >= self.var(SUB_ACTIONS, sub_id)/100)
            self.solver.add_constraint(self.var(SUB_CHANGED, sub_id) <= self.var(SUB_ACTIONS, sub_id))
        self.solver.add_constraint(sum([self.var(SUB_CHANGED, i) for i in range(self.nb_bus)]) <= nb_sub_actions)

    def break_symmetry(self):
        """
        Remove symmetry of the substation configurations

        :return: None
        """
        for i in range(self.nb_bus):
            if len(self.obs.get_obj_connect_to(substation_id=i)[LOADS_ID]) > 0:
                id_load = self.obs.get_obj_connect_to(substation_id=i)[LOADS_ID][0]
                self.solver.add_constraint(self.var(ETA_LOAD, id_load) >= 0.6)
            elif len(self.obs.get_obj_connect_to(substation_id=i)[GENERATORS_ID]) > 0:
                id_prod = self.obs.get_obj_connect_to(substation_id=i)[GENERATORS_ID][0]
                self.solver.add_constraint(self.var(ETA_PROD, id_prod) >= 0.6)
            elif len(self.obs.get_obj_connect_to(substation_id=i)[LINES_OR_ID]) > 0:
                id_branch = self.obs.get_obj_connect_to(substation_id=i)[LINES_OR_ID][0]
                self.solver.add_constraint(self.var(ETA_OR, id_branch) >= 0.6)
            elif len(self.obs.get_obj_connect_to(substation_id=i)[LINES_EX_ID]) > 0:
                id_branch = self.obs.get_obj_connect_to(substation_id=i)[LINES_EX_ID][0]
                self.solver.add_constraint(self.var(ETA_EX, id_branch) >= 0.6)

    def check_theta(self, solution: dict, x_branch) -> None:
        """
        Checks the relation between theta and flow on branches in DC approximation

        :param dict solution: solution of the optimization problem
        :param Any x_branch: Reactance of the branches
        """
        branch_id = list(solution[ETA_OR].keys())
        for i in branch_id:
            if solution[SWITCH_B][i] > 0.5:
                sub_or = self.obs.line_or_to_subid[i]
                sub_ex = self.obs.line_ex_to_subid[i]
                if solution[ETA_OR][i] < 0.5:
                    theta_or = solution[THETA+U2][sub_or]
                else:
                    theta_or = solution[THETA+U1][sub_or]
                if solution[ETA_EX][i] < 0.5:
                    theta_ex = solution[THETA+U2][sub_ex]
                else:
                    theta_ex = solution[THETA+U1][sub_ex]
                assert abs(theta_or - theta_ex - x_branch[i]*solution[DELTA_THETA_B][i]) < 1e-4
                assert abs(solution[FLOW_B][i] - solution[DELTA_THETA_B][i]) < 1e-4

    ################################################################################################
    # Main section of the code. Sets the optimisation problem
    ################################################################################################

    def common_add_var(self) -> None:
        """
        Based on the DC current approximation, set the optimization variables according to the activated constraints

        :return: None
        """
        # Branch variables
        for branch_id in self.branches:
            self.solver.create_num_var(-self.solver.infinity(), self.solver.infinity(), FLOW_B, branch_id)
            self.solver.create_num_var(0, self.solver.infinity(), FLOW_B_PLUS, branch_id)
            self.solver.create_num_var(0, self.solver.infinity(), FLOW_B_MINUS, branch_id)
            if self.constraints_dict[SLACK_B]:
                self.solver.create_num_var(0, self.M, SLACK_B, branch_id)
            if self.constraints_dict[SWITCH_B]:
                self.solver.create_int_var(0, 1, SWITCH_B, branch_id)
            for ext_1 in [U1, U2]:
                for ext_2 in [OR, EX]:
                    self.solver.create_num_var(-self.solver.infinity(), self.solver.infinity(), FLOW_B + ext_1 + ext_2,
                                               branch_id)
            self.solver.create_num_var(-self.solver.infinity(), self.solver.infinity(), DELTA_THETA_B, branch_id)
            if self.constraints_dict[ETA]:
                self.solver.create_int_var(0, 1, ETA_OR, branch_id)
                self.solver.create_int_var(0, 1, ETA_EX, branch_id)
        # Nodes variables
        for sub_id in self.subs:
            self.solver.create_num_var(0, self.solver.infinity(), SLACK_N_PLUS + U1, sub_id)
            self.solver.create_num_var(0, self.solver.infinity(), SLACK_N_MINUS + U1, sub_id)
            self.solver.create_num_var(0, self.solver.infinity(), SLACK_N_PLUS + U2, sub_id)
            self.solver.create_num_var(0, self.solver.infinity(), SLACK_N_MINUS + U2, sub_id)
            if self.constraints_dict[THETA]:
                self.solver.create_num_var(-self.solver.infinity(), self.solver.infinity(), THETA + U1, sub_id)
                self.solver.create_num_var(-self.solver.infinity(), self.solver.infinity(), THETA + U2, sub_id)
        # Production variables
        for prod_id in self.prods:
            self.solver.create_num_var(-self.solver.infinity(), self.solver.infinity(), PROD, prod_id)
            self.solver.create_num_var(-self.solver.infinity(), self.solver.infinity(), PROD + U1, prod_id)
            self.solver.create_num_var(-self.solver.infinity(), self.solver.infinity(), PROD + U2, prod_id)
            if self.constraints_dict[ETA]:
                self.solver.create_int_var(0, 1, ETA_PROD, prod_id)
        # Load variables
        for load_id in self.loads:
            self.solver.create_num_var(-self.solver.infinity(), self.solver.infinity(), LOAD + U1, load_id)
            self.solver.create_num_var(-self.solver.infinity(), self.solver.infinity(), LOAD + U2, load_id)
            if self.constraints_dict[ETA]:
                self.solver.create_int_var(0, 1, ETA_LOAD, load_id)

    def add_max_flow_constraints(self) -> None:
        """
        Adds flow limitation constraints. Overload measured b slack_b variables

        :return: None
        """
        for branch_id in self.branches:
            flow_i = self.var(FLOW_B, branch_id)
            max_flow_i = self.F_max_b[branch_id]
            if self.constraints_dict[FMAX] and not self.constraints_dict[SLACK_B]:
                self.solver.add_constraint(flow_i - max_flow_i <= 0)
                self.solver.add_constraint(-max_flow_i - flow_i <= 0)
            if self.constraints_dict[SLACK_B]:
                slack_i = self.var(SLACK_B, branch_id)
                self.solver.add_constraint(flow_i - slack_i - max_flow_i <= 0)
                self.solver.add_constraint(-flow_i - slack_i - max_flow_i <= 0)

    def production_var_equal_observed_prods(self) -> None:
        """
        In this appication, no dispatching is allowed, the production variables must be linked with the observed production

        :return: None
        """
        for i in self.prods:
            self.solver.add_constraint(self.obs.prod_p[i] - self.var(PROD, i) == 0)

    def common_add_constraints_bus(self) -> None:
        """
        Add bus constraints with slack on each bus

        :return: None
        """
        extensions = [U1, U2]
        for sub_id in self.subs:
            connected = self.obs.get_obj_connect_to(substation_id=sub_id)
            lines_or = list(set(connected[LINES_OR_ID]) & set(self.branches))
            lines_ex = list(set(connected[LINES_EX_ID]) & set(self.branches))
            prods = list(set(connected[GENERATORS_ID]) & set(self.prods))
            loads = list(set(connected[LOADS_ID]) & set(self.loads))
            for id_bus in range(2):
                ext = extensions[id_bus]
                self.solver.add_constraint((sum([self.var(FLOW_B + ext + OR, j) for j in lines_or])
                                            + sum([self.var(LOAD + ext, j) for j in loads])
                                            - sum([self.var(PROD + ext, j) for j in prods])
                                            - sum([self.var(FLOW_B + ext + EX, j) for j in lines_ex])
                                            - self.nodes_balance_by_bus2[id_bus+1][sub_id]
                                            == self.var(SLACK_N_PLUS + ext, sub_id) -
                                            self.var(SLACK_N_MINUS + ext, sub_id)))
        if self.constraints_dict[THETA]:
            self.solver.add_constraint(self.var(THETA + U1, self.subs[0]) == 0)

        for branch_id in self.branches:
            self.solver.add_constraint(self.var(FLOW_B, branch_id)
                                       == self.var(FLOW_B + U1 + OR, branch_id) + self.var(FLOW_B + U2 + OR, branch_id))
            self.solver.add_constraint(self.var(FLOW_B, branch_id)
                                       == self.var(FLOW_B + U1 + EX, branch_id) + self.var(FLOW_B + U2 + EX, branch_id))
        for prod_id in self.prods:
            self.solver.add_constraint(self.var(PROD, prod_id)
                                       == self.var(PROD + U1, prod_id) + self.var(PROD + U2, prod_id))
        for load_id in self.loads:
            self.solver.add_constraint(self.obs.load_p[load_id]
                                       - self.var(LOAD + U1, load_id) - self.var(LOAD + U2, load_id) == 0)

    def common_add_constraints_bus_contrib(self) -> None:
        """
        Add constraints linking eta variables and power contributions

        :return: None
        """
        for i in self.branches:  # intersect self.action_branches
            self.solver.add_constraint(self.var(FLOW_B + U1 + OR, i) + self.M * self.var(ETA_OR, i) >= 0)
            self.solver.add_constraint(-self.var(FLOW_B + U1 + OR, i) + self.M * self.var(ETA_OR, i) >= 0)
            self.solver.add_constraint(self.var(FLOW_B + U2 + OR, i) + self.M * (1 - self.var(ETA_OR, i)) >= 0)
            self.solver.add_constraint(-self.var(FLOW_B + U2 + OR, i) + self.M * (1 - self.var(ETA_OR, i)) >= 0)
            self.solver.add_constraint(self.var(FLOW_B + U1 + EX, i) + self.M * self.var(ETA_EX, i) >= 0)
            self.solver.add_constraint(-self.var(FLOW_B + U1 + EX, i) + self.M * self.var(ETA_EX, i) >= 0)
            self.solver.add_constraint(self.var(FLOW_B + U2 + EX, i) + self.M * (1 - self.var(ETA_EX, i)) >= 0)
            self.solver.add_constraint(-self.var(FLOW_B + U2 + EX, i) + self.M * (1 - self.var(ETA_EX, i)) >= 0)
        for i in self.prods:  # intersect self.action_prods
            self.solver.add_constraint(self.var(PROD + U1, i) + self.M * self.var(ETA_PROD, i) >= 0)
            self.solver.add_constraint(-self.var(PROD + U1, i) + self.M * self.var(ETA_PROD, i) >= 0)
            self.solver.add_constraint(self.var(PROD + U2, i) + self.M * (1 - self.var(ETA_PROD, i)) >= 0)
            self.solver.add_constraint(-self.var(PROD + U2, i) + self.M * (1 - self.var(ETA_PROD, i)) >= 0)
        for i in self.loads:  # intersect self.action_loads
            self.solver.add_constraint(self.var(LOAD + U1, i) + self.M * self.var(ETA_LOAD, i) >= 0)
            self.solver.add_constraint(-self.var(LOAD + U1, i) + self.M * self.var(ETA_LOAD, i) >= 0)
            self.solver.add_constraint(self.var(LOAD + U2, i) + self.M * (1 - self.var(ETA_LOAD, i)) >= 0)
            self.solver.add_constraint(-self.var(LOAD + U2, i) + self.M * (1 - self.var(ETA_LOAD, i)) >= 0)

    def common_add_constraints_bus_contrib_fixed(self) -> None:
        """
        Add contribution constraints for fixed bus topology (but switches can be activated)

        :return: None
        """
        ext = [U1, U2]
        for branch_id in self.branches:
            bus_or = self.obs.state_of(line_id=branch_id)[ORIGIN][BUS] % 2
            bus_ex = self.obs.state_of(line_id=branch_id)[EXTREMITY][BUS] % 2
            self.solver.add_constraint(self.var(FLOW_B + ext[bus_or] + OR, branch_id) == 0)
            self.solver.add_constraint(self.var(FLOW_B + ext[bus_ex] + EX, branch_id) == 0)
        for prod_id in self.prods:
            bus = self.obs.state_of(gen_id=prod_id)[BUS] % 2
            self.solver.add_constraint(self.var(PROD + ext[bus], prod_id) == 0)
        for load_id in self.loads:
            bus = self.obs.state_of(load_id=load_id)[BUS] % 2
            self.solver.add_constraint(self.var(LOAD + ext[bus], load_id) == 0)

    def common_add_flow_theta_const_mip(self) -> None:
        """
        Add the relation between flow and theta in DC approximation with switch and eta variables

        :return: None
        """
        for branch_id in self.branches:
            flow_i = self.var(FLOW_B, branch_id)
            max_flow_i = self.F_max_b[branch_id]
            slack_i = self.var(SLACK_B, branch_id)
            switch_i = self.var(SWITCH_B, branch_id)
            flow_i_plus = self.var(FLOW_B_PLUS, branch_id)
            flow_i_minus = self.var(FLOW_B_MINUS, branch_id)
            delta_theta = self.var(DELTA_THETA_B, branch_id)
            sub_id_or = self.obs.line_or_to_subid[branch_id]
            sub_id_ex = self.obs.line_ex_to_subid[branch_id]
            theta_or_1 = self.var(THETA + U1, sub_id_or)
            theta_or_2 = self.var(THETA + U2, sub_id_or)
            theta_ex_1 = self.var(THETA + U1, sub_id_ex)
            theta_ex_2 = self.var(THETA + U2, sub_id_ex)
            self.solver.add_constraint(flow_i - flow_i_plus + flow_i_minus == 0)
            if branch_id in self.forbidden_branches_id:
                self.solver.add_constraint(switch_i == int(self.obs.line_status[branch_id]))
            self.solver.add_constraint(flow_i - max_flow_i * switch_i - slack_i <= 0)
            self.solver.add_constraint(-max_flow_i * switch_i - slack_i - flow_i <= 0)
            self.solver.add_constraint(slack_i - switch_i * self.M <= 0)
            if self.constraints_dict[ETA]:
                eta_or = self.var(ETA_OR, branch_id)
                eta_ex = self.var(ETA_EX, branch_id)
                self.solver.add_constraint(-(1 - switch_i) * self.M - self.M * (eta_or + eta_ex) <=
                                           delta_theta - (theta_or_2 - theta_ex_2)/self.X_branch[branch_id])
                self.solver.add_constraint((1 - switch_i) * self.M + self.M * (eta_or + eta_ex) >=
                                           delta_theta - (theta_or_2 - theta_ex_2)/self.X_branch[branch_id])
                self.solver.add_constraint(-(1 - switch_i) * self.M - self.M * (1 - eta_or + eta_ex) <=
                                           delta_theta - (theta_or_1 - theta_ex_2)/self.X_branch[branch_id])
                self.solver.add_constraint((1 - switch_i) * self.M + self.M * (1 - eta_or + eta_ex) >=
                                           delta_theta - (theta_or_1 - theta_ex_2)/self.X_branch[branch_id])
                self.solver.add_constraint(-(1 - switch_i) * self.M - self.M * (1 + eta_or - eta_ex) <=
                                           delta_theta - (theta_or_2 - theta_ex_1)/self.X_branch[branch_id])
                self.solver.add_constraint((1 - switch_i) * self.M + self.M * (1 + eta_or - eta_ex) >=
                                           delta_theta - (theta_or_2 - theta_ex_1)/self.X_branch[branch_id])
                self.solver.add_constraint(-(1 - switch_i) * self.M - self.M * (2 - eta_or - eta_ex) <=
                                           delta_theta - (theta_or_1 - theta_ex_1)/self.X_branch[branch_id])
                self.solver.add_constraint((1 - switch_i) * self.M + self.M * (2 - eta_or - eta_ex) >=
                                           delta_theta - (theta_or_1 - theta_ex_1)/self.X_branch[branch_id])
            else:
                bus_or = self.obs.state_of(line_id=branch_id)[ORIGIN][BUS] % 2
                bus_ex = self.obs.state_of(line_id=branch_id)[EXTREMITY][BUS] % 2
                ext = [U2, U1]
                theta_or = self.var(THETA + ext[bus_or], sub_id_or)
                theta_ex = self.var(THETA + ext[bus_ex], sub_id_ex)
                self.solver.add_constraint(-2 * (1 - switch_i) * self.M - delta_theta + (theta_or - theta_ex)
                                           / self.X_branch[branch_id] <= 0)
                self.solver.add_constraint(-2 * (1 - switch_i) * self.M + delta_theta - (theta_or - theta_ex)
                                           / self.X_branch[branch_id] <= 0)
            self.solver.add_constraint(flow_i == delta_theta - self.shift[branch_id]/self.X_branch[branch_id])

    def common_add_flow_theta_const_lp(self) -> None:
        """
        Add flow theta constraints when switches and eta variables are fixed (for state estimation)

        :return: None
        """
        for branch_id in self.branches:
            flow_i = self.var(FLOW_B, branch_id)
            flow_i_plus = self.var(FLOW_B_PLUS, branch_id)
            flow_i_minus = self.var(FLOW_B_MINUS, branch_id)
            self.solver.add_constraint(flow_i - flow_i_plus + flow_i_minus == 0)

            sub_id_or = self.obs.line_or_to_subid[branch_id]
            sub_id_ex = self.obs.line_ex_to_subid[branch_id]
            if self.obs.state_of(line_id=branch_id)[ORIGIN][BUS] % 2 < 0.5:
                theta_or = self.var(THETA + U2, sub_id_or)
            else:
                theta_or = self.var(THETA + U1, sub_id_or)
            if self.obs.state_of(line_id=branch_id)[EXTREMITY][BUS] % 2 < 0.5:
                theta_ex = self.var(THETA + U2, sub_id_ex)
            else:
                theta_ex = self.var(THETA + U1, sub_id_ex)
            if self.obs.line_status[branch_id]:
                self.solver.add_constraint(self.X_branch[branch_id] * self.var(FLOW_B, branch_id)
                                           - (theta_or - theta_ex) == 0)
            else:
                self.solver.add_constraint(self.var(FLOW_B, branch_id) == 0)

    def common_add_objective(self) -> None:
        """
        Add an objective function for MIP problems to find the optimal actions.

        :return: None
        """
        np.random.seed(0)
        switch_cost = {}
        topo_branch_cost = {}
        topo_prod_cost = {}
        topo_load_cost = {}
        random_switch = np.random.standard_normal(self.nb_branch)
        random_topo_branch = np.random.standard_normal(self.nb_branch)
        random_topo_prod = np.random.standard_normal(self.nb_prod)
        random_topo_load = np.random.standard_normal(self.nb_load)
        for line_id in self.branches:
            switch_cost[line_id] = 1 / (100 + 10 * random_switch[line_id])
            topo_branch_cost[line_id] = 1 / (100 + 10 * random_topo_branch[line_id])
        for prod_id in self.prods:
            topo_prod_cost[prod_id] = 1 / (100 + 10 * random_topo_prod[prod_id])
        for load_id in self.loads:
            topo_load_cost[load_id] = 1 / (100 + 10 * random_topo_load[load_id])
        status = self.obs.line_status
        self.solver.create_objective()
        for sub_id in self.subs:
            for bus in [U1, U2]:
                self.solver.set_objective_coef(self.var(SLACK_N_PLUS + bus, sub_id), 10)
                self.solver.set_objective_coef(self.var(SLACK_N_MINUS + bus, sub_id), 10)
        if self.constraints_dict[SLACK_B]:
            for line_id in self.branches:
                self.solver.set_objective_coef(self.var(SLACK_B, line_id), 10)
        if self.constraints_dict[SWITCH_B]:
            for line_id in self.branches:
                self.solver.set_objective_coef(self.var(SWITCH_B, line_id),
                                               (1 - 2 * int(status[line_id])) * switch_cost[line_id])
        if self.constraints_dict[ETA]:
            origin_bus_status = {i: self.obs.state_of(line_id=i)[ORIGIN][BUS] % 2 for i in self.branches}
            extremity_bus_status = {i: self.obs.state_of(line_id=i)[EXTREMITY][BUS] % 2 for i in self.branches}
            prod_bus_status = {i: self.obs.state_of(gen_id=i)[BUS] % 2 for i in self.prods}
            load_bus_status = {i: self.obs.state_of(load_id=i)[BUS] % 2 for i in self.loads}
            for line_id in self.branches:
                self.solver.set_objective_coef(self.var(ETA_OR, line_id), (1 - 2 * int(origin_bus_status[line_id]))
                                               * topo_branch_cost[line_id])
                self.solver.set_objective_coef(self.var(ETA_EX, line_id), (1 - 2 * int(extremity_bus_status[line_id]))
                                               * topo_branch_cost[line_id])
            for prod_id in self.prods:
                self.solver.set_objective_coef(self.var(ETA_PROD, prod_id), (1 - 2 * int(prod_bus_status[prod_id]))
                                               * topo_prod_cost[prod_id])
            for load_id in self.loads:
                self.solver.set_objective_coef(self.var(ETA_LOAD, load_id), (1 - 2 * int(load_bus_status[load_id]))
                                               * topo_load_cost[load_id])
        self.solver.set_minimization()

    def remove_limit_constraint_on_branch_area(self, branches: list) -> None:
        """
        Remove the limit of the branches

        :param list branches: branches for which flow limit will be ignored (set to self.M)
        :return : None
        """
        for branch_id in branches:
            self.F_max_b[branch_id] = self.M

    def fix_variables_in_fixed(self, fixed: dict) -> None:
        """
        Fixe variables for branches, prods and loads according to values in fixed

        :param dict fixed: Variables values to fix with keys [SWITCH_B, ETA_OR, ETA_EX, ETA_PROD, ETA_LOAD]
        :return : None
        """
        for i in fixed[SWITCH_B].keys():
            self.solver.add_constraint(self.var(SWITCH_B, i) == fixed[SWITCH_B][i])
        for i in fixed[ETA_OR].keys():
            self.solver.add_constraint(self.var(ETA_OR, i) == fixed[ETA_OR][i])
            self.solver.add_constraint(self.var(ETA_EX, i) == fixed[ETA_EX][i])
        for i in fixed[ETA_PROD].keys():
            self.solver.add_constraint(self.var(ETA_PROD, i) == fixed[ETA_PROD][i])
        for i in fixed[ETA_LOAD].keys():
            self.solver.add_constraint(self.var(ETA_LOAD, i) == fixed[ETA_LOAD][i])
