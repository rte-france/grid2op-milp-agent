# Mixed Integer Linear Programming (MILP) agent

The MILP agent controller is a power grid monitoring developed to manage over-thermal lines using topological actions.
The MLIP agent takes a snapshot of the grid state (environment), then under linear constraints based on the DC 
approximation, this agent seeks to minimize the overthermal lines by using topological actions 
(line switching, bus reconfiguration). Currently, the MILP agent is adapted with pandapower backend powerflow solver.

## Installation
### Requirements
This agent uses the following python libraries, which will be installed when you install milp_agent:
- Python >= 3.6

The MILP agent will used by the way the following libraries:
- grid2op
- pandapower
- ortools
- numpy
- pandas

They will be installed if you install the MILP agent from source.

You can install them using pip:
`pip install -U grid2op pandapower ortools numpy pandas`

### Install the package
You can directly install the milp agent from pip too:

```commandline
# first you need to clone the repository
git clone THEGITHUBLINK
cd milp_agent
pip install -e .
```
And you can now use it as any python library.

## How to use MILP_agent ?

After install the MIP_Agent, the following example shows how use the MLIP agent.
```python3
import grid2op 
from grid2op.Agent import DoNothingAgent
from grid2op.PlotGrid import PlotMatplot
from grid2op.Action import DontAct
from grid2op.Chronics import GridStateFromFileWithForecastsWithoutMaintenance
import numpy as np
import milp_agent
from milp_agent.agent import MILPAgent


env_name = "rte_case5_example"
env = grid2op.make(env_name, test=True)
obs = env.get_obs()
margins = 0.95*np.ones(obs.n_line)
obs = env.get_obs()
Agent_type = milp_agent.GLOBAL_SWITCH 
Solver_type = milp_agent.MIP_CBC
agent = MILPAgent(env, Agent_type, margins, solver_name=Solver_type)
# and now you can use it as any grid2op compatible agent, for example:
action = agent.act(obs, reward=0.0, done=False)
new_state, reward, done, info = env.step(action)
```
Note that this library takes the environment from grid2op library. The MILP agent is tested on 
[L2RPN](https://competitions.codalab.org/competitions/20767) data.

## Additional information

### The process of execution
In order to use the MILP agent for topological agent controller, you follow  these steps:
- Set the grid environment by using grid2op.
- Specify the type of the agent and the solver name
- For a fixed snapshot, the MILP agent minimizes the over-thermal limits of line under constraint to this snapshot.
- The remaining action from the MILP agent is adapted with gri2op agent, where it will be performed.

You could use the usecase method to run the MILP agent automatically to solve the over thermal lines in complete 
episode until the end of the episode or gameover due to the violation of one of grid2op rules. Using this method, 
the result will be saved in output and log_agent folders.

### MILP agent types

The MILP agent is implemented in different ways:
- Global agent - line switches only: agent uses only the line switching to manage the over-thremal lines for the 
  entire network.
- Global agent - full actions: manages the overflow constraints by using all the topological actions available 
  in the power grid (line switching, bus reconfiguration) for the whole grid.
- Complimentary agent - double level agent: two independent full topological agents, the first one solves the 
  overflow constraint for a fixed number of electrical bus group, while the second agent controls the elementary 
  structures. The two agents are independent at each iteration.
- Multi-zone agent: several mini-agent controllers, each agent controls a zone independently from the 
  remaining grid structures. This type of agent requires a bus clustering. We use the clustering based on the 
  [segmentation](https://arxiv.org/pdf/1711.09715.pdf). The segmentation process clusters the power lines basing of 
  transfer power, causing by line disconnection. The translation of power lines clustering to electrical bus 
  creates a boundary bus. The boundary buses are assigned to each cluster connected to it.

### MILP Solver
The MILP agent uses [Or-tools](https://developers.google.com/optimization/introduction/python) to set the constraint 
equations, define the cost function and solve the optimization problem. Or-tools is an open source software shaped 
for handling integer, linear and constraint programming. Or-tools allow us to implement the modelization of any 
problems in the programming language of your choice. Or-tools is adapted to many open-source solvers 
like [Scip](https://www.scipopt.org/),[GLPK](https://www.gnu.org/software/glpk/), 
[GloP]('https://developers.google.com/optimization/lp/glop') or 
[CBC]('https://projects.coin-or.org/Cbc'), 
or commercial solvers like [Xpress]('https://www.fico.com/en/products/fico-xpress-solver') or 
[CPLEX](https://www.ibm.com/fr-fr/analytics/cplex-optimizer). In out work, we adapt our MILP agent to run under CBC, 
SCIP, GLOP or Xpress solver.

### Grid2op environment
The environment that the MLIP agent relies on to observe the state of the grid, set actions and check the correctness 
of the action are encapsulated under [Grid2op](https://github.com/rte-france/Grid2Op). This platform constructed 
to simulate the power grid control, allows the MILP agent to get a snapshot of the grid and perform the action 
that minimize the over-thremal lines on the grid. 

### Agent characteristics
Each type of the MILP agents needs to specify the following parameters: 
- `env`: the grid2op environment.
- `agent_solver`: the type of agent.
- `max_overflow_percentage`: the limit thermal threshold, beyond which the agent considers that the line is under 
  constraint. 
- `solver_name`: the solver that will be used by the agent in order to optimize and solve the constraint in the grid.

The uses of the complimentary double agent need to specify:
- `Instance zone`: the bus group that will be monitored by one iteration of topological agent
- `zone_id`: the id zones from the clustering file that the complimentary or double level agent needs to 
  build a bus group controlled by one iteration of topological agent., 

The other iteration manages the complimentary grid structures.

The utility of complimentary double and multi-zone agent need additionally to specify:
- `clustering_path`: the path towards the clustering file, it defines the segmentation of the power grid
- `zone_level`: Radius of supplementary ring around the zone that allows the agent to have outside zone vision.

The agents could be customized by:

- `constant list`: a list of structures that the agent could not modify
- `limit number actions`: the number of actions that the agent can perform in one iteration.


## License information
copyright 2020-2021 RTE France
```
RTE: http://www.rte-france.com
```
This Source Code is subject to the terms of the Mozilla Public Licence (MPL) v2 also avaible [here](https://www.mozilla.org/en-US/MPL/2.0/)

## Contributing
We welcome contribution from everyone. They can take the form of pull requests. In case of a major change, 
please open an issue first to discuss what you would like to change.

Code in the contribution should pass all the tests, have some dedicated tests for the new feature and documentation.
