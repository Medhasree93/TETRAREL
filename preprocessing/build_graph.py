
import pandas as pd
import networkx as nx


def build_transaction_graph(csv_path):
    df = pd.read_csv(csv_path)

    G = nx.DiGraph()

    for _, row in df.iterrows():
        sender = row['sender']
        receiver = row['receiver']
        amount = row['amount']
        timestamp = row['timestamp']

        G.add_edge(
            sender,
            receiver,
            amount=amount,
            timestamp=timestamp
        )

    return G
