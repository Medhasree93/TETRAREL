from preprocessing.build_graph import build_transaction_graph
from preprocessing.feature_engineering import compute_node_features
from preprocessing.ego_graph import extract_ego_graphs
from models.codes_sampler import CODESSampler


if __name__ == '__main__':

    graph = build_transaction_graph('data/raw/ethereum.csv')

    features = compute_node_features(graph)

    ego_graphs = extract_ego_graphs(graph)

    sampler = CODESSampler(sample_ratio=0.8)

    for node, ego in ego_graphs.items():
        sampled_edges = sampler.sample_edges(ego)

        print(node, len(sampled_edges))
