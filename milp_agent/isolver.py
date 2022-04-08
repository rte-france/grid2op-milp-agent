# Copyright (c) 2020-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of MIP_Agent, The MIP_Agent is a optimization based agent that manage the power flow
# overthermal using topological actions.

from ortools.linear_solver import pywraplp
import numpy as np

from .global_var import MIP_CBC, MIP_XPRESS, LINEAR_GLOP, LINEAR_XPRESS, STATE_ESTIMATION, ISOLVER_VARIABLE_LIST, OBJ, \
    MIP_SCIP, MIP_GUROBI, MIP_GLPK


class ISolver:
    """
    Solver implemented on OrTools model
    """

    def __init__(self, solver_name):
        """
        Initialize ORtools solver

        :param solver_name: Name of solver type
        """
        if solver_name == MIP_CBC:
            self.solver = pywraplp.Solver('Pb', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        elif solver_name == MIP_XPRESS:
            self.solver = pywraplp.Solver('Pb', pywraplp.Solver.XPRESS_MIXED_INTEGER_PROGRAMMING)
        elif solver_name == LINEAR_GLOP:
            self.solver = pywraplp.Solver('Pb', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
        elif solver_name == LINEAR_XPRESS:
            self.solver = pywraplp.Solver('Pb', pywraplp.Solver.XPRESS_LINEAR_PROGRAMMING)
        elif solver_name == STATE_ESTIMATION:
            self.solver = pywraplp.Solver('Pb', pywraplp.Solver.XPRESS_LINEAR_PROGRAMMING)
        elif solver_name == MIP_SCIP:
            self.solver = pywraplp.Solver('Pb', pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)
        elif solver_name == MIP_GUROBI:
            self.solver = pywraplp.Solver('Pb', pywraplp.Solver.GUROBI_MIXED_INTEGER_PROGRAMMING)
        elif solver_name == MIP_GLPK:
            self.solver = pywraplp.Solver('Pb', pywraplp.Solver.GLPK_MIXED_INTEGER_PROGRAMMING)
        self.constraints = {}
        self.variables = {var: {} for var in ISOLVER_VARIABLE_LIST}

    def infinity():
        return np.infty
    infinity = staticmethod(infinity)

    def create_num_var(self, lb, ub, category, index):
        """
        Creates numerical varible for the optimization problem. example: create_num_var(0,np.infinity,'branch',0)

        :param float lb: lower bound of variable
        :param float ub: upper bound of variable
        :param str category: type of vairable
        :param int index: the id of vairable in category sets.
        :return : None
        """
        self.variables[category][index] = self.solver.NumVar(lb, ub, category + "_" + str(index))

    def create_int_var(self, lb, ub, category, index):
        """
        Creates integer varible for the optimization problem. example: create_num_var(0,1,'eta',0)

        :param int lb: lower bound of variable
        :param int ub: upper bound of variable
        :param str category: type of vairable
        :param int index: the id of vairable in category sets.
        :return : None
        """
        self.variables[category][index] = self.solver.IntVar(lb, ub, category + "_" + str(index))

    def add_constraint(self, constraint):
        """
        Adds optimization constraint

        :param constaint: optimization constraint 
        :return : None
        """
        self.solver.Add(constraint)

    def create_objective(self):
        """
        Create the objective function
        """
        return self.solver.Objective()

    def set_objective_coef(self, variable, coef):
        """
        Sets the coefficient of the vairables in the cost function
        
        :param str variable : solver variable to be defined
        :param float coef : the coefficient corresponding to this variable
        """
        self.solver.Objective().SetCoefficient(variable, coef)

    def set_minimization(self):
        self.solver.Objective().SetMinimization()

    def display(self):
        print(self.solver.ExportModelAsLpFormat(False).replace('\\', '').replace(',_', ','), sep='\n')

    def solve(self):
        return self.solver.Solve() == pywraplp.Solver.OPTIMAL

    def solution_as_dict(self):
        """
        Returns the solution of the problem. Can only be called after solving.

        :return: Solution
        """
        solution = {var: {} for var in ISOLVER_VARIABLE_LIST}
        solution[OBJ] = self.solver.Objective().Value()
        for category in list(self.variables.keys()):
            for i in self.variables[category].keys():
                solution[category][i] = self.variables[category][i].solution_value()
        return solution
