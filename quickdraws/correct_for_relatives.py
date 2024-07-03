# This file is part of the Quickdraws GWAS software suite.
#
# Copyright (C) 2024 Quickdraws Developers
#
# Quickdraws is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Quickdraws is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Quickdraws. If not, see <http://www.gnu.org/licenses/>.


import pandas as pd
import numpy as np
from scipy.linalg import eigh

## Functions adapted using chatgpt
def find_root(mapping, i):
    while mapping[i] != i:
        i = mapping[i]
    return i

def merge_clusters(cluster_mapping, root1, root2):
    if root1 != root2:
        cluster_mapping[root1] = root2

def king_value_to_relatedness(king):
    if king > 2**(-1.5):
        return 1
    elif king > 2**(-2.5):
        return 0.5
    elif king > 2**(-3.5):
        return 0.25
    elif king > 2**(-4.5):
        return 0.125
    else:
        return 0

def get_correction_for_relatives(sample_list, kinship, h2):
    kinship = kinship[kinship["ID1"].isin(sample_list) & kinship["ID2"].isin(sample_list)]
    kinship = kinship[kinship['Kinship'] > 2**(-4.5)]
    data = kinship[['ID1', 'ID2', 'Kinship']].values
    data = data[data[:, 2].argsort()[::-1]] 

    # Initialize clusters and cluster mapping
    clusters = {i: [i] for i in sample_list}
    cluster_mapping = {i: i for i in clusters}

    # Merge clusters based on the King value
    for ID1, ID2, King_value in data:
        root1 = find_root(cluster_mapping, ID1)
        root2 = find_root(cluster_mapping, ID2)
        if root1 != root2:
            merge_clusters(cluster_mapping, root1, root2)

    # Update cluster roots and compile final clusters
    final_clusters = {}
    for i in cluster_mapping:
        root = find_root(cluster_mapping, i)
        if root not in final_clusters:
            final_clusters[root] = []
        final_clusters[root].append(i)

    # Compute average King value for each cluster
    cluster_king_values = {root: [] for root in final_clusters}
    for ID1, ID2, King_value in data:
        root1 = find_root(cluster_mapping, ID1)
        root2 = find_root(cluster_mapping, ID2)
        if root1 == root2:
            cluster_king_values[root1].append(King_value)

    # List of lists of IDs and their corresponding average King values
    clusters_list = list(final_clusters.values())
    cluster_size = [len(vals) if vals else 0 for vals in final_clusters.values()]
    average_kings = [sum(vals)/len(vals) if vals else 0 for vals in cluster_king_values.values()]

    # Get eigen values for the GRM
    grm_eigen_sum = np.zeros_like(h2)
    for root, members in final_clusters.items():
        n = len(members)
        if n == 1:
            grm_eigen_sum += 1.0
        elif n == 2:
            grm = np.eye(n)
            grm[0,1] = king_value_to_relatedness(cluster_king_values[root][0])
            grm[1,0] = king_value_to_relatedness(cluster_king_values[root][0])
            eigenvalues = eigh(grm, eigvals_only=True)
            for phen_no in range(len(h2)):
                grm_eigen_sum[phen_no] += np.sum(eigenvalues/((eigenvalues-1)*h2[phen_no] + 1))
        else:
            grm = np.eye(n)
            index_map = {id: idx for idx, id in enumerate(members)}
            for ID1, ID2, King_value in data:
                if ID1 in members and ID2 in members:
                    i, j = index_map[ID1], index_map[ID2]
                    grm[i, j] = grm[j, i] = king_value_to_relatedness(King_value)
            eigenvalues = eigh(grm, eigvals_only=True)
            for phen_no in range(len(h2)):
                grm_eigen_sum[phen_no] += np.sum(eigenvalues/((eigenvalues-1)*h2[phen_no] + 1))

    sample_size_correction = grm_eigen_sum/len(sample_list)
    return sample_size_correction

def get_unrelated_list(sample_list, kinship):
    kinship = kinship[kinship["ID1"].isin(sample_list) & kinship["ID2"].isin(sample_list)]
    kinship = kinship[kinship['Kinship'] > 2**(-4.5)]
    unrel_list = set(sample_list) - set(kinship.ID1)
    unrel_list = unrel_list - set(kinship.ID2)
    unrel_list = list(unrel_list)

    graph = {}
    for i in range(len(kinship)):
        node1, node2 = kinship.iloc[i].values
        if node1 not in graph:
            graph[node1] = []
        if node2 not in graph:
            graph[node2] = []
        graph[node1].append(node2)
        graph[node2].append(node1)

    label = {}
    for node in graph:
        label[node] = False

    for node in graph:
        all_false = True
        for neighbor in graph[node]:
            if label[neighbor]:
                all_false = False
                break
        if all_false:
            label[node] = True
    mis = [node for node in graph if label[node]]

    unrel_list = unrel_list + mis
    unrel_list = np.unique(unrel_list)
    return unrel_list


if __name__ == '__main__':
    kinship = pd.read_csv('/well/palamara/projects/UKBB_APPLICATION_43206/new_copy/sample_lists/relatedness/ukb_rel_a43206_s488248.dat', '\s+')
    fam = pd.read_csv('/well/palamara/users/vnk166/workspace/meta_learning_prs/ukbb_gbp/genotype.fam', '\s+', names=["FID", "IID", 0, 1, 2, 3])
    # fam = pd.read_csv('/well/palamara/projects/UKBB_APPLICATION_43206/new_copy/plink_missingness_regenielike_filters/ukb_app43206_500k.maf00001.fam', '\s+', names=["FID", "IID", 0, 1, 2, 3])
    # fam = pd.read_csv('ukbb50k_rel3_gbp.fam', '\s+', names=["FID", "IID", 0, 1, 2, 3])

    ## merge with phenotype
    # traits = pd.read_csv('')
    # fam = pd.merge(fam, traits[])

    ## get the estimated h2
    h2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    correction = get_correction_for_relatives(fam.IID, kinship, h2)
    print(correction)

