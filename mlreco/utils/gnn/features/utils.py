import itertools
import numpy as np
from scipy.spatial import Delaunay

from topologylayer.util import get_clusts
import torch

def node_labels_to_edge_labels(edges, node_labels):
    label_starts = node_labels[edges[:, 0]]
    label_ends = node_labels[edges[:, 1]]
    edge_labels = np.ones(len(edges))
    edge_labels[np.where(label_starts != label_ends)] = 0
    return edge_labels

def find_parent(parent, i):
    if i != parent[i]:
        parent[i] = find_parent(parent, parent[i])
    return parent[i]

# union find
# def edge_labels_to_node_labels(edges, edge_labels_raw, threshold=0.5):
#     if isinstance(edge_labels_raw, torch.Tensor):
#         edge_labels = edge_labels_raw.numpy()
#     else:
#         edge_labels = edge_labels_raw
#     n = np.amax(edges) + 1
#     labels = get_clusts(edges, edge_labels, n, 0.5)
#     return np.array(labels)

def edge_labels_to_node_labels(edges, edge_labels, threshold=0.5):
#     edge_labels = edge_labels_raw.numpy()
    on_edges = edges[np.where(edge_labels > threshold)[0]]
    node_labels = np.arange(int(np.amax(edges)) + 1)
    for a, b in on_edges:
        p1 = find_parent(node_labels, a)
        p2 = find_parent(node_labels, b)
        if p1 != p2:
            node_labels[p1] = p2
    return node_labels

def node_labels_to_cluster_sizes(node_labels):
    unique, counts = np.unique(node_labels, return_counts=True)
    sizes = np.zeros(len(node_labels))
    for i in range(len(unique)):
        sizes[np.where(node_labels == unique[i])] = counts[i]
    return sizes
    
def create_edge_indices(positions):
    n = len(positions)
    nodes = np.arange(n)
    
    simplices = Delaunay(positions).simplices
    simplices.sort()
    edges = set()
    for s in simplices:
        edges |= set(itertools.combinations(s, 2))
    edges = np.array(list(edges))
    return edges