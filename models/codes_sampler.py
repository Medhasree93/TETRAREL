import random


class CODESSampler:
    def __init__(self, sample_ratio=0.8):
        self.sample_ratio = sample_ratio

    def temporal_transaction_score(self, edge_data):
        amount = edge_data['amount']
        timestamp = edge_data['timestamp']

        score = amount * 0.7 + timestamp * 0.3
        return score

    def sample_edges(self, ego_graph):
        edges = list(ego_graph.edges(data=True))

        scored_edges = []

        for edge in edges:
            score = self.temporal_transaction_score(edge[2])
            scored_edges.append((edge, score))

        scored_edges.sort(key=lambda x: x[1], reverse=True)

        keep = int(len(scored_edges) * self.sample_ratio)

        sampled = scored_edges[:keep]

        return [x[0] for x in sampled]
