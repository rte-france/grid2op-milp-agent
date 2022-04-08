
import os
import time
import numpy as np
import re

import grid2op
from grid2op.Backend import PandaPowerBackend
from lightsim2grid import LightSimBackend
from grid2op.Opponent import BaseOpponent
from grid2op.Action import DontAct

import milp_agent
from milp_agent.agent import AgentMPC
from milp_agent import GLOBAL_TOPOLOGY, GLOBAL_SWITCH

from tqdm import tqdm

# from ortools import InitGoogleLogging
# InitGoogleLogging()

env_name = "l2rpn_case14_sandbox"
env_name = "l2rpn_neurips_2020_track1_small"
env_name = "l2rpn_neurips_2020_track2_small"

if env_name == "l2rpn_case14_sandbox":
    lines_attackable = ["1_3_3", "1_4_4", "3_6_15", "9_10_12", "11_12_13", "12_13_14"]
    lines_attackable = [3, 4, 15, 12, 13, 14]
    lines_attackable = [4, 15, 12, 13, 14, 3]
elif re.match("^l2rpn_neurips_2020_track1_.*", env_name) is not None:
    lines_attackable = ["62_58_180", "62_63_160", "48_50_136", "48_53_141", "41_48_131", "39_41_121",
                      "43_44_125", "44_45_126", "34_35_110", "54_58_154"]
    lines_attackable = [56, 45, 23, 27, 18, 9, 13, 14, 0, 39]
elif re.match("^l2rpn_neurips_2020_track2_.*", env_name) is not None:
    lines_attackable = ["TODO"]
    lines_attackable = [10]
else:
    raise RuntimeError("Unknown env")

orig_env = grid2op.make(env_name,
                        backend=LightSimBackend(),
                        opponent_class=BaseOpponent,
                        opponent_init_budget=0.,
                        opponent_action_class=DontAct)

if re.match("^l2rpn_neurips_2020_track2_.*", env_name) is not None:
    # use only the hardest mix
    env = orig_env["l2rpn_neurips_2020_track2_x3"]
else:
    env = orig_env
from grid2op.Runner import Runner
runner = Runner(**env.get_params_for_runner())

grid_path = os.path.join(env.get_path_env(), "grid.json")
agent_backend = PandaPowerBackend()
agent_backend.load_grid(grid_path)

agent = AgentMPC(env.action_space,
                 grid_path,
                 GLOBAL_SWITCH,
                 max_overflow_percentage=np.ones(env.n_line) * 0.90,
                 solver_name=milp_agent.MIP_SCIP,  # MIP_CBC LINEAR_GLOP MIP_SCIP MIP_GUROBI
                 env_backend=agent_backend._grid._ppc,
                 thermal_limits=env.get_thermal_limit())
action_every_xxx_steps = 50

nb_step = 288 * 7
nb_real_act = 0
total_time = 0.
env.seed(1)
# for case14_sandbox
# 0 is super easy (dn stops at ts=1091)
# 1 (dn=>807)
# 2 (>= 2016)
# 3 is hard one (dn stop at ts => 3)
# 4 => 804
# 5 => 513
# for neurips_track1
# 0 => >= 2016
# 1 => >= 2016
# 2 => >= 2016
# 3 => >= 2016 ...
# for neurips track2 (x3 mix)
# 0 => 992
# 1 => 424
# 2 => 1274
# 3 => 121
# 4 => 78
# 5 => 113
# 6 => 2148
# 7 => 132
# 8 => 438
# 9 => 1845
env.set_id(4)
dn = env.action_space()
obs = env.reset()
prev_obs = obs
init_obs = obs
if False:
    # first run to look at the do nothing
    nb_epi = 10
    res = runner.run(nb_epi, env_seeds=[1 for _ in range(nb_epi)], pbar=True)
    for i, el in enumerate(res):
        print(f"scen {i}: {el[3]} / {el[4]}")

li_action = [env.action_space({"set_line_status": [[lines_attackable[i % len(lines_attackable)], -1]]})
             for i in range(int(nb_step // action_every_xxx_steps) + 1)]
for i in tqdm(range(nb_step), disable=False):
    # print(f"{obs.line_status = }")
    # print(f"###########################")
    # print(f"           step {i}        ")
    # print(f"###########################")
    act_this = False
    if i % action_every_xxx_steps == 0 and False:
        # pass
        act = li_action[i // action_every_xxx_steps]
        act = dn
        # print(f"Random action: \n{act}")
    else:
        start_ = time.time()
        act = agent.act(obs, reward=0.0)
        end_ = time.time()
        lines_aff, sub_aff = act.get_topological_impact()
        if np.sum(lines_aff) + np.sum(sub_aff):
            nb_real_act += 1
            print(f"at step {i}: {act}")
            act_this = True
        else:
            pass
            # print(f"Doing nothing at step {i} (max rho: {np.max(obs.rho):.2f})")
        total_time += end_ - start_
    obs, reward, done, info = env.step(act)
    if info["exception"] and not done:
        import pdb
        pdb.set_trace()
    # if act_this:
    #     import pdb
    #     pdb.set_trace()
    if done:
        break
    prev_obs = obs
    # print()

print(f"Time to perform {i + 1}/{nb_step} steps ({nb_real_act} real actions): {total_time:.2f}s")
