import os
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, HeteroData,Data
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.transforms import ToUndirected
import torch.nn as nn
from tools.nlp import clean_text

class IssueAssignDataset(InMemoryDataset):
    def __init__(self, root, hetero=True, transform=None, pre_transform=None):
        self.hetero = hetero
        self.data_type = 'hetero' if self.hetero else 'homo'
        super(IssueAssignDataset, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['issue_content.csv', 'user_issue.csv', 'resolved_issues.csv', 'opened_issues.csv']

    @property
    def processed_dir(self):
            # Return different processing directories based on the 'hetero' parameter
        return os.path.join(self.root, f'processed_{self.data_type}')

    @property
    def processed_file_names(self):
        return [f'data.pt']

    def download(self):
        pass

    def process(self):
        # Define file paths
        issue_content_path = os.path.join(self.raw_dir, 'issue_content.csv')
        user_issue_path = os.path.join(self.raw_dir, 'user_issue.csv')
        resolved_issues_path = os.path.join(self.raw_dir, 'resolved_issues.csv')
        opened_issues_path = os.path.join(self.raw_dir, 'opened_issues.csv')
        dim = 64

        # Prepare text feature extractors
        title_vectorizer = TfidfVectorizer(max_features=(dim // 2))
        body_vectorizer = TfidfVectorizer(max_features=(dim // 2))

        # Load nodes and mappings
        issue_x, issue_mapping = self.load_issue_nodes(issue_content_path, opened_issues_path, title_vectorizer, body_vectorizer)
        user_embedding, user_mapping = self.load_user_nodes(user_issue_path, opened_issues_path, dim)

        # Mark open_issues
        is_open_issue = torch.zeros(len(issue_mapping), dtype=torch.bool)
        opened_issues_df = pd.read_csv(opened_issues_path)
        for number in opened_issues_df['number']:
            idx = issue_mapping[number]
            is_open_issue[idx] = True

        # Load edge data
        participate_edge_index, edge_weight = self.load_participate_edges(user_issue_path, user_mapping, issue_mapping)
        resolved_edge_index = self.get_resolved_edges(resolved_issues_path, issue_mapping, user_mapping)
        open_edge_index = self.load_open_edges(opened_issues_path, user_mapping, issue_mapping)

        # Remove resolved edges from participate edges
        participate_edge_index, edge_weight = self.remove_positive_edges_from_participate(
            participate_edge_index, edge_weight, resolved_edge_index, user_mapping, issue_mapping
        )

        if self.hetero:
            data = self.build_hetero_data(
                user_embedding, issue_x,participate_edge_index, edge_weight, resolved_edge_index, open_edge_index, is_open_issue
            )
        else:
            data,issue_mapping= self.build_homo_data(
                user_embedding, issue_x, user_mapping, issue_mapping,
                participate_edge_index, edge_weight, resolved_edge_index, open_edge_index, is_open_issue
            )

        # Apply pre-processing transformations
        if self.pre_transform:
            data = self.pre_transform(data)

        # Save the processed data
        self.save([data], self.processed_paths[0])

        # Save mappings for future use
        torch.save((user_mapping, issue_mapping), os.path.join(self.processed_dir, 'mappings.pt'))

    def load_issue_nodes(self, issue_content_path, opened_issues_path, title_vectorizer, body_vectorizer):
        # Load issue node data
        issue_x, issue_mapping = self.get_node_mapping(
            issue_content_path, 'number',
            encoders={
                'title': lambda x: title_vectorizer.fit_transform(x).toarray(),
                'body': lambda x: body_vectorizer.fit_transform(x).toarray()
            },
            cleaners={
                'title': clean_text,
                'body': clean_text
            }
        )

        open_issue_x, open_issue_mapping = self.get_node_mapping(
            opened_issues_path, 'number',
            encoders={
                'title': lambda x: title_vectorizer.transform(x).toarray(),
                'body': lambda x: body_vectorizer.transform(x).toarray()
            },
            cleaners={
                'title': clean_text,
                'body': clean_text
            }
        )

        # Merge issue_mapping and issue_x
        for number, idx in open_issue_mapping.items():
            if number not in issue_mapping:
                issue_mapping[number] = len(issue_mapping)
                issue_x = torch.cat([issue_x, open_issue_x[idx].unsqueeze(0)], dim=0)
            else:
                pass
        return issue_x, issue_mapping

    def load_user_nodes(self, user_issue_path, opened_issues_path, dim):
        # Load user node data
        _, user_mapping = self.get_node_mapping(user_issue_path, 'UserName')

        # Update user_mapping
        opened_issues_df = pd.read_csv(opened_issues_path)
        for opener in opened_issues_df['user']:
            if opener not in user_mapping:
                user_mapping[opener] = len(user_mapping)

        # Create user embeddings
        user_embedding = nn.Embedding(len(user_mapping), dim)
        return user_embedding, user_mapping

    def load_participate_edges(self, user_issue_path, user_mapping, issue_mapping):
        # Edge weight mapping
        weight_mapping = {
            'PR_OPEN': 1,
            'COMMENTED': 1,
            'REVIEW_COMMENT': 1,
            'ISSUE_OPEN': 1,
            'LABELED': 1,
            'NORMAL_COMMENT': 1
        }

        # Load participate edge data
        edge_index, _, edge_weight = self.get_edge_index(
            user_issue_path,
            'UserName', user_mapping,
            'IssueNumber', issue_mapping,
            weight_mapping=weight_mapping,
            weight_col='EventType'
        )
        return edge_index, edge_weight

    def load_open_edges(self, opened_issues_path, user_mapping, issue_mapping):
        # Load 'open' edges
        opened_issues_df = pd.read_csv(opened_issues_path)
        opener_indices = []
        issue_indices = []
        for _, row in opened_issues_df.iterrows():
            opener = row['user']
            issue_number = row['number']
            opener_idx = user_mapping[opener]
            issue_idx = issue_mapping[issue_number]
            opener_indices.append(opener_idx)
            issue_indices.append(issue_idx)
        open_edge_index = torch.tensor([opener_indices, issue_indices], dtype=torch.long)
        return open_edge_index

    def build_hetero_data(self, user_embedding, issue_x,participate_edge_index, edge_weight, resolved_edge_index, open_edge_index, is_open_issue):
        # Build HeteroData object
        data = HeteroData()
        data['user'].x = user_embedding.weight
        data['issue'].x = issue_x
        data['user', 'participate', 'issue'].edge_index = participate_edge_index
        data['user', 'participate', 'issue'].edge_weight = edge_weight
        data['issue', 'resolved_by', 'user'].edge_index = resolved_edge_index
        data['user', 'open', 'issue'].edge_index = open_edge_index
        data['issue'].is_open_issue = is_open_issue
        data.num_users = data['user'].num_nodes
        data.num_issues = data['issue'].num_nodes
        data = ToUndirected()(data)
        return data

    def build_homo_data(self, user_embedding, issue_x, user_mapping, issue_mapping,
                        participate_edge_index, edge_weight, resolved_edge_index, open_edge_index, is_open_issue):
        # Build homogeneous graph Data object

        # Adjust issue node indices to be globally unique
        num_users = len(user_mapping)
        num_issues = len(issue_mapping)
        adjusted_issue_mapping = {}
        for number, idx in issue_mapping.items():
            adjusted_idx = idx + num_users
            adjusted_issue_mapping[number] = adjusted_idx
        issue_mapping = adjusted_issue_mapping

        # Global node feature matrix x
        x = torch.cat([user_embedding.weight, issue_x], dim=0)  # Shape: [num_users + num_issues, dim]

        # Node type tensor
        node_type = torch.zeros(num_users + num_issues, dtype=torch.long)
        node_type[:num_users] = 0  # users
        node_type[num_users:] = 1  # issues

        # Create is_open_issue tensor
        is_open_issue_full = torch.zeros(num_users + num_issues, dtype=torch.bool)
        is_open_issue_full[num_users:] = is_open_issue

        # Collect edge indices, edge types, edge weights
        edge_indices = []
        edge_types = []
        edge_weights = []

        # Edge type mapping
        edge_type_mapping = {
            'participate': 0,
            'resolved_by': 1,
            'open': 2
        }

        # Adjust participate_edge_index
        participate_edge_index[0] = participate_edge_index[0]
        participate_edge_index[1] += num_users
        edge_indices.append(participate_edge_index)
        edge_types.append(torch.full((participate_edge_index.size(1),), edge_type_mapping['participate'], dtype=torch.long))
        edge_weights.append(edge_weight)

        # Adjust resolved_edge_index
        resolved_edge_index[0] += num_users  # issue nodes adjusted
        resolved_edge_index[1] = resolved_edge_index[1]
        edge_indices.append(resolved_edge_index)
        edge_types.append(torch.full((resolved_edge_index.size(1),), edge_type_mapping['resolved_by'], dtype=torch.long))
        resolved_edge_weight = torch.ones(resolved_edge_index.size(1), dtype=torch.float)
        edge_weights.append(resolved_edge_weight)

        # Adjust open_edge_index
        open_edge_index[0] = open_edge_index[0]
        open_edge_index[1] += num_users
        edge_indices.append(open_edge_index)
        edge_types.append(torch.full((open_edge_index.size(1),), edge_type_mapping['open'], dtype=torch.long))
        open_edge_weight = torch.ones(open_edge_index.size(1), dtype=torch.float)
        edge_weights.append(open_edge_weight)

        # Merge all edges
        edge_index = torch.cat(edge_indices, dim=1)
        edge_type = torch.cat(edge_types, dim=0)
        edge_weight = torch.cat(edge_weights, dim=0)

        # Build Data object
        data = Data(x=x, edge_index=edge_index)
        data.node_type = node_type
        data.edge_type = edge_type
        data.edge_weight = edge_weight
        data.is_open_issue = is_open_issue_full
        data.num_users = num_users
        data.num_issues = num_issues
        return data, issue_mapping

    def get_node_mapping(self, file_path, index_col, encoders=None, cleaners=None):
        try:
            df = pd.read_csv(file_path, index_col=index_col)
        except Exception as e:
            print(f"Error reading the CSV file {file_path}: {e}")
            return None, None

        node_mapping = {index: i for i, index in enumerate(df.index.unique())}

        if cleaners:
            for col, cleaner in cleaners.items():
                df[col] = df[col].fillna('').apply(cleaner)

        node_vec = None
        if encoders:
            node_vec_list = []
            for col, encoder in encoders.items():
                encoded = encoder(df[col])
                node_vec_list.append(torch.tensor(encoded, dtype=torch.float))
            node_vec = torch.cat(node_vec_list, dim=-1) if node_vec_list else None

        return node_vec, node_mapping

    def get_edge_index(self, file_path, src_index_col, src_mapping,
                       dst_index_col, dst_mapping, weight_mapping=None, weight_col=None):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading the CSV file {file_path}: {e}")
            return None, None, None

        # Map source and destination nodes to indices
        src = df[src_index_col].map(src_mapping)
        dst = df[dst_index_col].map(dst_mapping)

        # Remove missing values
        valid = src.notna() & dst.notna()
        src = src[valid].astype(int).tolist()
        dst = dst[valid].astype(int).tolist()

        edge_index = torch.tensor([src, dst], dtype=torch.long)

        edge_attr = None  # Handle edge attributes

        edge_weight = None
        if weight_mapping and weight_col:
            weights = df[weight_col].map(weight_mapping).fillna(0)
            weights = weights[valid].tolist()
            edge_weight = torch.tensor(weights, dtype=torch.float)

        return edge_index, edge_attr, edge_weight

    def get_resolved_edges(self, file_path, issue_mapping, user_mapping):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading the CSV file {file_path}: {e}")
            return None

        # Create a list of edges
        issue_indices = []
        user_indices = []
        df['resolver'] = df['resolver'].apply(self.extract_and_filter_resolvers)
        for _, row in df.iterrows():
            issue_idx = issue_mapping.get(row['number'], None)
            if issue_idx is not None:
                for resolver in row['resolver']:
                    user_idx = user_mapping.get(resolver, None)
                    if user_idx is not None:
                        issue_indices.append(issue_idx)
                        user_indices.append(user_idx)

        resolved_edge_index = torch.tensor([issue_indices, user_indices], dtype=torch.long)
        return resolved_edge_index

    def extract_and_filter_resolvers(self, resolvers_str):
        resolvers = resolvers_str.strip("[]").split(',')
        filtered_resolvers = [resolver.strip().strip("'") for resolver in resolvers]
        return filtered_resolvers

    def remove_positive_edges_from_participate(self, participate_edge_index, participate_edge_weight,
                                               resolved_edge_index, user_mapping, issue_mapping):
        """
        Remove the edges in 'resolved_by' from 'participate' edges to prevent information leakage.
        """
        # Get the total number of nodes to calculate unique edge IDs
        num_users = len(user_mapping)
        num_issues = len(issue_mapping)
        max_node_index = max(num_users, num_issues)

        # Calculate unique IDs for participate edges
        edge_ids_participate = participate_edge_index[0] * max_node_index + participate_edge_index[1]
        # Calculate unique IDs for resolved edges
        edge_ids_resolved = resolved_edge_index[1] * max_node_index + resolved_edge_index[0]

        # Find edges to keep (those in participate edges that are not in resolved edges)
        mask = ~torch.isin(edge_ids_participate, edge_ids_resolved)

        # Filter participate edges and corresponding edge weights
        filtered_participate_edge_index = participate_edge_index[:, mask]
        filtered_participate_edge_weight = participate_edge_weight[mask]

        return filtered_participate_edge_index, filtered_participate_edge_weight

import networkx as nx
import matplotlib.pyplot as plt

def analyze_network(data, user_mapping, issue_mapping):
    """
    Analyzing the classical index and degree distribution of complex networks
    Consider only the relationship ('user','participant','issue')
    And the node label is consistent with the actual data
    """
    relation = ('user', 'participate', 'issue')
    if relation not in data.edge_index_dict:
        print(f"Relationship {relation} not in DB.")
        return

    edge_index = data.edge_index_dict[relation]  # [2, num_edges]

    user_mapping_inv = {idx: user for user, idx in user_mapping.items()}
    issue_mapping_inv = {idx: issue for issue, idx in issue_mapping.items()}

    user_indices = edge_index[0].numpy()
    issue_indices = edge_index[1].numpy()

    user_names = [user_mapping_inv[idx] for idx in user_indices]
    issue_numbers = [issue_mapping_inv[idx] for idx in issue_indices]

    G = nx.Graph()
    G.add_nodes_from(user_mapping.keys(), bipartite='user', type='user')
    G.add_nodes_from(issue_mapping.keys(), bipartite='issue', type='issue')

    edges = list(zip(user_names, issue_numbers))
    G.add_edges_from(edges)

    degrees = dict(G.degree())
    degree_values = list(degrees.values())

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    average_degree = sum(degree_values) / num_nodes if num_nodes > 0 else 0
    density = nx.density(G)
    average_clustering = nx.average_clustering(G) if num_nodes > 0 else 0
    number_of_connected_components = nx.number_connected_components(G)
    if nx.is_connected(G):
        diameter = nx.diameter(G)
    else:
        diameters = []
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            if subgraph.number_of_nodes() > 1:
                diameters.append(nx.diameter(subgraph))
        diameter = max(diameters) if diameters else "∞"

    print("=== Complex Network Metrics ===")
    print(f"Number of Nodes: {num_nodes}")
    print(f"Number of Edges: {num_edges}")
    print(f"Average Degree: {average_degree:.2f}")
    print(f"Density: {density:.4f}")
    print(f"Average Clustering Coefficient: {average_clustering:.4f}")
    print(f"Number of Connected Components: {number_of_connected_components}")
    print(f"Diameter of the Graph: {diameter}")

    # Plot degree distribution
    plt.figure(figsize=(8, 6))
    plt.hist(degree_values, bins=50, edgecolor='black',log=True)
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Number of Nodes")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.1, seed=42)
    user_nodes = list(user_mapping.keys())
    issue_nodes = list(issue_mapping.keys())
    nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_color='blue', node_size=50, label='User')
    nx.draw_networkx_nodes(G, pos, nodelist=issue_nodes, node_color='red', node_size=50, label='Issue')
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.legend(scatterpoints=1)
    plt.title("user-participate-issue")
    plt.axis('off')
    plt.show()


def dataset_to_graph(dataset_name, hetero):
    print("Loading node and edge data...")
    dataset = IssueAssignDataset(os.path.abspath(os.path.join('dataset', dataset_name)), hetero=hetero)
    data = dataset[0] 
    user_mapping, issue_mapping = torch.load(os.path.join(dataset.processed_dir, 'mappings.pt'))
    # analyze_network(data,user_mapping, issue_mapping)
    return data, user_mapping, issue_mapping

