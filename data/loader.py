from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader,NeighborLoader
from torch_geometric.utils import negative_sampling
import torch
from torch_geometric.data import Data
# Split the data into training set, validation set, and testing set
def split_dataset(data, hetero=True):
    print("将数据划分为训练集、验证集------------------------------------------")
    if hetero:
        dataset_split = RandomLinkSplit(
            num_val=0.1,
            num_test=0,  # Test set provided separately
            is_undirected=True,
            add_negative_train_samples=True,
            edge_types=[('issue', 'resolved_by', 'user')],
            rev_edge_types=[('user', 'rev_resolved_by', 'issue')]
        )
        train_data, val_data, _ = dataset_split(data)
    else:
        # 1. Filter edges of type 'resolved_by' based on edge_type as positive samples
        resolved_edge_type = 1  # According to the edge_type_mapping definition in the issueassignmentdataset
        mask = data.edge_type == resolved_edge_type
        pos_edge_index = data.edge_index[:, mask]

        # 2. Remove the 'resolved_by' edge from data.edge_index to prevent information leakage
        remaining_mask = data.edge_type != resolved_edge_type
        data.edge_index = data.edge_index[:, remaining_mask]
        if hasattr(data, 'edge_weight'):
            data.edge_weight = data.edge_weight[remaining_mask]
        if hasattr(data, 'edge_type'):
            data.edge_type = data.edge_type[remaining_mask]

        # 3. Split the positive sample edges into a training set and a validation set
        num_pos_edges = pos_edge_index.size(1)
        perm = torch.randperm(num_pos_edges)
        num_val = int(num_pos_edges * 0.1)
        num_train = num_pos_edges - num_val

        train_pos_edge_index = pos_edge_index[:, perm[:num_train]]
        val_pos_edge_index = pos_edge_index[:, perm[num_train:]]

        # 4. Generate negative samples
        num_nodes = data.num_nodes
        num_neg_train = train_pos_edge_index.size(1)
        num_neg_val = val_pos_edge_index.size(1)

        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=num_nodes,
            num_neg_samples=num_neg_train + num_neg_val,
            method='sparse'
        )
        train_neg_edge_index = neg_edge_index[:, :num_neg_train]
        val_neg_edge_index = neg_edge_index[:, num_neg_train:num_neg_train + num_neg_val]

        # 5. Building edge_label_index and edge_label for training and validation sets
        train_edge_label_index = torch.cat([train_pos_edge_index, train_neg_edge_index], dim=1)
        train_edge_label = torch.cat([torch.ones(train_pos_edge_index.size(1)),
                                      torch.zeros(train_neg_edge_index.size(1))], dim=0)

        val_edge_label_index = torch.cat([val_pos_edge_index, val_neg_edge_index], dim=1)
        val_edge_label = torch.cat([torch.ones(val_pos_edge_index.size(1)),
                                    torch.zeros(val_neg_edge_index.size(1))], dim=0)

        # 6. Create training and validation data objects
        train_data = Data(
            x=data.x,
            edge_index=data.edge_index,
            edge_weight=data.edge_weight if hasattr(data, 'edge_weight') else None,
            edge_label_index=train_edge_label_index,
            edge_label=train_edge_label,
        )

        val_data = Data(
            x=data.x,
            edge_index=data.edge_index,
            edge_weight=data.edge_weight if hasattr(data, 'edge_weight') else None,
            edge_label_index=val_edge_label_index,
            edge_label=val_edge_label,
        )

    return train_data, val_data

def dataset_to_batch(data, train_data, val_data, batch_size, hetero=True):
    num_neighbors = [10, 10]  # Number of neighbors sampled
    if hetero:
        print("train_loader ---------------------------------")
        train_loader = LinkNeighborLoader(
            train_data,
            num_neighbors=num_neighbors,
            edge_label_index=(('issue', 'resolved_by', 'user'),
                              train_data['issue', 'resolved_by', 'user'].edge_label_index),
            edge_label=train_data['issue', 'resolved_by', 'user'].edge_label,
            batch_size=batch_size,
            shuffle=True
        )
        print("val_loader ---------------------------------")
        val_loader = LinkNeighborLoader(
            val_data,
            num_neighbors=num_neighbors,
            edge_label_index=(('issue', 'resolved_by', 'user'),
                              val_data['issue', 'resolved_by', 'user'].edge_label_index),
            edge_label=val_data['issue', 'resolved_by', 'user'].edge_label,
            batch_size=batch_size,
            shuffle=False
        )
    else:
        print("train_loader ---------------------------------")
        train_loader = LinkNeighborLoader(
            data=train_data,
            num_neighbors=num_neighbors,
            edge_label_index=train_data.edge_label_index,
            edge_label=train_data.edge_label,
            batch_size=batch_size,
            shuffle=True
        )
        print("val_loader ---------------------------------")
        val_loader = LinkNeighborLoader(
            data=val_data,
            num_neighbors=num_neighbors,
            edge_label_index=val_data.edge_label_index,
            edge_label=val_data.edge_label,
            batch_size=batch_size,
            shuffle=False
        )

    # test_loader
    if hetero:
        # obtain the index of open_issues
        open_issue_indices = torch.nonzero(data['issue'].is_open_issue).squeeze()
        test_loader = NeighborLoader(
            data,
            num_neighbors=[0],
            input_nodes=('issue', open_issue_indices),
            batch_size=open_issue_indices.size(0),
            shuffle=False
        )
    else:
        # For homogeneous graphs, obtain the node index with node_type is issue and is_open_issue set to True
        issue_node_type = 1
        open_issue_indices = torch.nonzero((data.node_type == issue_node_type) & data.is_open_issue).squeeze()
        test_loader = NeighborLoader(
            data,
            num_neighbors=[0],
            input_nodes=open_issue_indices,
            batch_size=open_issue_indices.size(0),
            shuffle=False
        )

    return train_loader, val_loader, test_loader



