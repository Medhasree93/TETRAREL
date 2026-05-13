import numpy as np


def compute_node_features(G):
    features = {}

    for node in G.nodes():
        incoming = []
        outgoing = []
        timestamps = []

        for u, v, data in G.in_edges(node, data=True):
            incoming.append(data['amount'])
            timestamps.append(data['timestamp'])

        for u, v, data in G.out_edges(node, data=True):
            outgoing.append(data['amount'])
            timestamps.append(data['timestamp'])

        features[node] = np.array([
            np.mean(incoming) if len(incoming) > 0 else 0,
            np.mean(outgoing) if len(outgoing) > 0 else 0,
            len(incoming) + len(outgoing),
            len(set(timestamps))
        ])

    return features
