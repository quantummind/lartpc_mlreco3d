from mlreco.utils.gnn.features.utils import *
import numpy as np
from mlreco.utils.gnn.features.cone import cone_features
from mlreco.utils.gnn.features.spectral import spectral_features
from mlreco.utils.gnn.features.dbscan import dbscan_features
from sklearn.cluster import DBSCAN
from sklearn.neighbors import RadiusNeighborsRegressor

# node features: x, y, z
# edge features: distance
def basic_features(positions, edges):
    nf = positions
    ef = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    ef = np.reshape(ef, (-1, 1))
    return nf, ef

def get_em_positions(data, return_groups=False, filter_compton=False, compton_cut=30):
    input_true = data['input_true']
    input_reco = data['input_reco']
    segment_label = data['segment_label']
    group_label = data['group_label']
    
    input_true = input_true[np.where(segment_label[:, -1] == 2)]
    
    if filter_compton:
        clusters = DBSCAN(eps=1.01, min_samples=3).fit(input_true[:, :3]).labels_
        u, c = np.unique(clusters, return_counts=True)
        for l in u[np.where(c < compton_cut)]:
            clusters[np.where(clusters == l)] = -1
        compton_filter = np.where(clusters != -1)[0]
        input_true = input_true[compton_filter]
    
    chosen_indices = []
    chosen_reco_indices = []
    
    current_batch = 0
    current_batch_selection = np.where(input_true[:, -2] == current_batch)[0]
    current_input_true = input_true[current_batch_selection]
    for r in range(len(input_reco)):
        row = input_reco[r]
        b = row[-2]
        if b != current_batch:
            current_batch = b
            current_batch_selection = np.where(input_true[:, -2] == current_batch)[0]
        pos = row[:3]
        region_selection = np.where((current_input_true[:, 0] == pos[0]) & (current_input_true[:, 1] == pos[1]))[0]
        input_true_region = current_input_true[region_selection]
        for i in range(len(input_true_region)):
            row2 = input_true_region[i]
            pos2 = row2[:3]
            if np.array_equal(pos, pos2):
                chosen_indices.append(current_batch_selection[region_selection[i]])
                chosen_reco_indices.append(r)
                break
    
    if len(chosen_indices) == 0:
        return None
    
    chosen_indices = np.array(chosen_indices)
    
    if not return_groups:
        return input_true[chosen_indices][:, :3]
    
    
    found_data = input_true[chosen_indices]
    
    # find where the chosen indices are in the group data
    found_group_data = -np.ones((len(found_data), len(found_data[0])))
    for i in range(len(found_data)):
        row = found_data[i]
        filter0 = group_label[np.where(group_label[:, -2] == row[-2])]
        filter1 = filter0[np.where(filter0[:, 0] == row[0])]
        filter2 = filter1[np.where(filter1[:, 1] == row[1])]
        filter3 = filter2[np.where(filter2[:, 2] == row[2])]
        g = filter3[0]
        found_group_data[i] = g
    return found_data[:, :3], found_group_data[:, -1]

# returns positions, edges, node features, edge features (assumes batch size == 1)
def generate_graph(data, feature_types=['basic', 'cone', 'spectral', 'dbscan'], filter_compton=False, compton_cut=30):
    positions, groups = get_em_positions(data, return_groups=True, filter_compton=filter_compton, compton_cut=compton_cut)
    edges = create_edge_indices(positions)
    
    em_positions = data['em_primaries'][:, :3]
    
    all_nf = []
    all_ef = []
    for ft in feature_types:
        if ft == 'basic':
            nf, ef = basic_features(positions, edges)
        elif ft == 'cone':
            nf, ef = cone_features(positions, em_positions, edges)
        elif ft == 'spectral':
            nf, ef = spectral_features(positions, edges)
        elif ft == 'dbscan':
            nf, ef = dbscan_features(positions, em_positions, edges)
        all_nf.append(nf)
        all_ef.append(ef)
    
    all_nf = np.concatenate(tuple(all_nf), axis=1)
    all_ef = np.concatenate(tuple(all_ef), axis=1)
    
    return positions, edges, all_nf, all_ef, groups

# returns positions, edges, node features, edge features (assumes batch size == 1)
def generate_truth(data, positions=None, groups=None, edges=None, filter_compton=False, compton_cut=30):
    if positions is None or groups is None:
        positions, groups = get_em_positions(data, return_groups=True, filter_compton=filter_compton, compton_cut=compton_cut)
    if edges is None:
        edges = create_edge_indices(positions)
    edge_labels = node_labels_to_edge_labels(edges, groups)
    return positions, edges, edge_labels