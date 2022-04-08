# Copyright (c) 2020-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of MILP_Agent, The MILP_Agent is a optimization based agent that manage the power flow
# overthermal using topological actions.

import logging
def logger_conf():
    logger= logging.getLogger()
    logger.setLevel(logging.INFO) 
    handler = logging.FileHandler('example.log', 'w', 'utf-8') 
    handler.setFormatter(logging.Formatter('%(asctime)s,%(levelname)s: %(message)s')) 
    logger.addHandler(handler) 
    #logger.info('Logger file for topological agent')
    return logger
