import networkx as nx


def extract_ego_graphs(G, hop=1):
    ego_graphs = {}

    for node in G.nodes():
        ego_graphs[node] = nx.ego_graph(
            G,
            node,
            radius=hop,
            undirected=False
        )

    return ego_graphs
