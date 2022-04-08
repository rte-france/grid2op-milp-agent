# Copyright (c) 2020-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of MILP_Agent, The MILP_Agent is a optimization based agent that manage the power flow
# overthermal using topological actions.


import pandas as pd
from .global_var import (CLUSTER, ZONE_BUS, ZONE_BRANCH, ZONE_PROD, ZONE_LOAD, SUB_ID, ORIGIN, EXTREMITY,
                         GENERATORS_ID, LINES_OR_ID, LINES_EX_ID, LOADS_ID)


class Zones:
    """
    Reads the clustering.csv file from the Input folder to build zones.
    The clustering.csv file contains the bus clustering, which containts two columnsthe bus identity
    and corresponding cluster.
    The cluster 0 would not controled by any micro-milp. the clusters must be taked as ordred interge
    values like [1,2,3...N]
    Zones of id i is accessible by calling self.clustering[i]. It consists in dict with keys in fields list.
    """

    def __init__(self, obs, clustering_path):
        """
        param obj obs: grid2op grid observation.
        param str clustering_path: the path toward the clustering file.
        """

        path = clustering_path
        data = pd.read_csv(path, sep=",")
        cluster = list(data[CLUSTER])
        nb_sub = len(cluster)
        temporary = []
        for x in cluster:
            try:
                temporary.append(int(x))
            except ValueError or TypeError:
                raise RuntimeError("The cluster tag must be integer where the cluster 0 in not controled by any agent."
                                    "Think to map the cluster tag before you run the agent.")
        nb_cluster = max(temporary)
        fields = [ZONE_BUS, ZONE_BRANCH, ZONE_PROD, ZONE_LOAD]
        clustering_info = {i: {field: [] for field in fields} for i in range(nb_cluster)}
        cluster_matching = []
        for sub_id in range(nb_sub):
            string_clusters = cluster[sub_id]
            sub_clusters = []
            if x != 0:
                sub_clusters.append(int(x) - 1)
            cluster_matching.append(sub_clusters)
        # Zone bus
        for id_sub in range(nb_sub):
            for c in cluster_matching[id_sub]:
                clustering_info[c][ZONE_BUS].append(id_sub)
        # Zone branch
        for branch_id in range(len(obs.line_or_to_subid)):
            intersection = list(set(cluster_matching[obs.line_or_to_subid[branch_id]])
                                & set(cluster_matching[obs.line_ex_to_subid[branch_id]]))
            if len(intersection) > 0:  # Set branch to first cluster if more than two (branch connecting two boundaries)
                clustering_info[intersection[0]][ZONE_BRANCH].append(branch_id)
        # Zone Prod
        for id_prod in range(len(obs.prod_p)):
            sub_prod_clusters = cluster_matching[obs.state_of(gen_id=id_prod)[SUB_ID]]
            if len(sub_prod_clusters) > 0:
                clustering_info[sub_prod_clusters[0]][ZONE_PROD].append(id_prod)
        # Zone Load
        for id_load in range(len(obs.load_p)):
            sub_load_clusters = cluster_matching[obs.state_of(load_id=id_load)[SUB_ID]]
            if len(sub_load_clusters) > 0:
                clustering_info[sub_load_clusters[0]][ZONE_LOAD].append(id_load)

        self.nb_cluster = nb_cluster
        self.clustering = clustering_info
        self.matching = cluster_matching
        self.obs = obs

    def zone_ring(self, zone: dict, level: int) -> dict:
        """
        Recursive function that increases the zone up to depth equal to level
        This new zone includes all substations at distance equal to or less than the level of the original zone
        in terms of connections
        As well as lines between newly added boundary substations

        :param dict zone: Original zone
        :param int level: Orbit of the ring
        :return: New zone
        """
        if level == 0:
            return zone
        else:
            new_elements = {ZONE_BRANCH: [], ZONE_BUS: [], ZONE_PROD: [], ZONE_LOAD: []}
            for sub_id in zone[ZONE_BUS]:
                connected_or_lines_id = list(self.obs.get_obj_connect_to(substation_id=sub_id)[LINES_OR_ID])
                connected_ex_lines_id = list(self.obs.get_obj_connect_to(substation_id=sub_id)[LINES_EX_ID])
                connected_lines_id = connected_or_lines_id + connected_ex_lines_id
                for line_id in list(set(connected_lines_id).difference(set(zone[ZONE_BRANCH]))):
                    new_elements[ZONE_BRANCH].append(line_id)
                    sub_or = self.obs.state_of(line_id=line_id)[ORIGIN][SUB_ID]
                    sub_ex = self.obs.state_of(line_id=line_id)[EXTREMITY][SUB_ID]
                    if sub_or not in zone[ZONE_BUS] and sub_or not in new_elements[ZONE_BUS]:
                        new_elements[ZONE_BUS].append(sub_or)
                        new_elements[ZONE_PROD] += list(
                            self.obs.get_obj_connect_to(substation_id=sub_or)[GENERATORS_ID])
                        new_elements[ZONE_LOAD] += list(self.obs.get_obj_connect_to(substation_id=sub_or)[LOADS_ID])
                    elif sub_ex not in zone[ZONE_BUS] and sub_ex not in new_elements[ZONE_BUS]:
                        new_elements[ZONE_BUS].append(sub_ex)
                        new_elements[ZONE_PROD] += list(
                            self.obs.get_obj_connect_to(substation_id=sub_ex)[GENERATORS_ID])
                        new_elements[ZONE_LOAD] += list(self.obs.get_obj_connect_to(substation_id=sub_ex)[LOADS_ID])
            # Not forgetting lines between two newly added substations
            for sub_id in new_elements[ZONE_BUS]:
                for line_id in list(self.obs.get_obj_connect_to(substation_id=sub_id)[LINES_OR_ID]):
                    if self.obs.state_of(line_id=line_id)[EXTREMITY][SUB_ID] in new_elements[ZONE_BUS] \
                            and line_id not in zone[ZONE_BRANCH]:
                        zone[ZONE_BRANCH].append(line_id)
            for key in new_elements.keys():
                zone[key] += new_elements[key]
            return self.zone_ring(zone, level - 1)