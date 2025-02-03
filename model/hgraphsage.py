from .registry import ModelRegistry
from .graphbasemodel import GraphBaseModel
import os
from torch_geometric.nn import SAGEConv,HeteroConv
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.conv import GraphConv
import torch
from datetime import datetime, timezone
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@ModelRegistry.register('hgraphsage')
class HGraphSage(GraphBaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.in_channels = int(self.config["hyperparameter"]['in_channels'])
        self.hidden_channels = int(self.config["hyperparameter"]['hidden_channels'])
        self.out_channels = int(self.config["hyperparameter"]['out_channels'])
        self.dropout = float(self.config["hyperparameter"]['dropout'])
        self.model = HeteroGraphSAGE(self.in_channels,self.hidden_channels, self.out_channels,self.dropout).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train(self):
        self.model.train()
        for epoch in range(self.epoch):
            preds = []
            labels = []
            total_loss = 0
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                batch = batch.to(device)
                x_dict = batch.x_dict
                edge_index_dict = batch.edge_index_dict
                
                # Build edge_weight_dict
                edge_weight_dict = {}
                for rel in edge_index_dict.keys():
                    if 'edge_weight' in batch[rel]:
                        edge_weight_dict[rel] = batch[rel].edge_weight
                    else:
                        edge_weight_dict[rel] = None
                
                out_dict = self.model(x_dict, edge_index_dict,edge_weight_dict)
                
                issue_emb = out_dict['issue']
                user_emb = out_dict['user']
                src = batch['issue', 'resolved_by', 'user'].edge_label_index[0]
                dst = batch['issue', 'resolved_by', 'user'].edge_label_index[1]
                
                src_emb = issue_emb[src]
                dst_emb = user_emb[dst]
                
                pred = (src_emb * dst_emb).sum(dim=-1)
                loss = self.criterion(pred, batch['issue', 'resolved_by', 'user'].edge_label.float())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                preds.append(pred.cpu())
                labels.append(batch['issue', 'resolved_by', 'user'].edge_label.cpu()) 

            self.log.info(f'Epoch {epoch+1}/{self.epoch}, Loss: {total_loss:.4f}')       
            preds = torch.cat(preds)
            labels = torch.cat(labels)
            pred_labels = (preds > 0.4).float()
            accuracy = accuracy_score(labels, pred_labels)
            f1 = f1_score(labels, pred_labels)
            auc = roc_auc_score(labels.detach().numpy(), preds.detach().numpy())
            self.log.info(f'Train Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')

    def validate(self):
        self.model.eval()
        total_loss = 0
        preds = []
        labels = []
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(device)
                x_dict = batch.x_dict
                edge_index_dict = batch.edge_index_dict
                
                edge_weight_dict = {}
                for rel in edge_index_dict.keys():
                    if 'edge_weight' in batch[rel]:
                        edge_weight_dict[rel] = batch[rel].edge_weight
                    else:
                        edge_weight_dict[rel] = None
                
                out_dict = self.model(x_dict, edge_index_dict,edge_weight_dict)
                
                issue_emb = out_dict['issue']
                user_emb = out_dict['user']
                
                src = batch['issue', 'resolved_by', 'user'].edge_label_index[0]
                dst = batch['issue', 'resolved_by', 'user'].edge_label_index[1]
                
                src_emb = issue_emb[src]
                dst_emb = user_emb[dst]
                
                pred = (src_emb * dst_emb).sum(dim=-1)
                loss = self.criterion(pred, batch['issue', 'resolved_by', 'user'].edge_label.float())
                total_loss += loss.item()               
                preds.append(pred.cpu())
                labels.append(batch['issue', 'resolved_by', 'user'].edge_label.cpu())

        preds = torch.sigmoid(torch.cat(preds))
        labels = torch.cat(labels)
        pred_labels = ( preds > 0.6).float()
        accuracy = accuracy_score(labels, pred_labels)
        f1 = f1_score(labels, pred_labels)
        auc = roc_auc_score(labels.detach().numpy(), preds.detach().numpy())
        self.log.info(f'Validate Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')

    # # For large-scale graphs, save user embeddings in batches
    # def get_allnode_emb(self):
    #     self.user_loader = NeighborLoader(
    #                 self.data,
    #                 num_neighbors=[-1], 
    #                 input_nodes=('user', torch.arange(self.data['user'].num_nodes)),
    #                 batch_size=self.batch_size,
    #                 shuffle=False
    #             )
    #     self.model.eval()
    #     user_emb_list = []
    #     with torch.no_grad():
    #         for batch in self.user_loader:
    #             batch = batch.to(device)
    #             x_dict = batch.x_dict
    #             edge_index_dict = batch.edge_index_dict

    #             edge_weight_dict = {}
    #             for rel in edge_index_dict.keys():
    #                 if 'edge_weight' in batch[rel]:
    #                     edge_weight_dict[rel] = batch[rel].edge_weight
    #                 else:
    #                     edge_weight_dict[rel] = None

    #             out_dict = self.model(x_dict, edge_index_dict, edge_weight_dict)
    #             user_emb = out_dict['user']

    #             # Retrieve the global user node index within the batch
    #             global_user_indices = batch['user'].n_id.cpu()

    #             # Save embeddings and indexes
    #             user_emb_list.append((global_user_indices, user_emb.cpu()))
        
    #     # According to the global user index, concatenate the embeddings into a complete user embedding tensor
    #     num_users = self.data['user'].num_nodes
    #     self.user_emb = torch.zeros((num_users, self.out_channels))
    #     for indices, emb in user_emb_list:
    #         self.user_emb[indices] = emb
    #     print("All user embeddings have been computed and saved, shape:", self.user_emb.shape)
     
    def get_allnode_emb(self):
        '''
        For small and medium-sized graphs, a forward propagation of the entire
        graph yields the embedding of the user and saves it
        '''
        self.model.eval()
        with torch.no_grad():
            self.data = self.data.to(device)
            x_dict = self.data.x_dict
            edge_index_dict = self.data.edge_index_dict
            edge_weight_dict = {}
            for rel in edge_index_dict.keys():
                if 'edge_weight' in self.data[rel]:
                    edge_weight_dict[rel] = self.data[rel].edge_weight
                else:
                    edge_weight_dict[rel] = None
            out_dict = self.model(x_dict, edge_index_dict, edge_weight_dict)
            self.user_emb = out_dict['user']
        print("All user embeddings have been computed and saved, shape:", self.user_emb.shape)

    def test(self):
        '''
        Predicting the future is a completely new issue, 
        without ground truth or any interactive information
        '''
        self.get_allnode_emb()
        self.model.eval()
        with torch.no_grad():
            for subgraph in self.test_loader:
                subgraph = subgraph.to(device)
                x_dict = subgraph.x_dict
                edge_index_dict = subgraph.edge_index_dict

                edge_weight_dict = {}
                for rel in edge_index_dict.keys():
                    if 'edge_weight' in subgraph[rel]:
                        edge_weight_dict[rel] = subgraph[rel].edge_weight
                    else:
                        edge_weight_dict[rel] = None

                out_dict = self.model(x_dict,edge_index_dict,edge_weight_dict)
                # issue_emb = self.model.issue_mlp(x_dict['issue'])
                issue_emb = out_dict['issue']  # [batch_size, out_channels]

                # Retrieve the issue index from the batch
                # issue_batch = subgraph['issue'].batch  # [batch_size]
                # For each issue, calculate the similarity with all users
                scores = torch.matmul(self.user_emb, issue_emb.T).T  # [batch_size, num_users]
                probabilities = torch.sigmoid(scores)

                # Get the top-K users for each issue
                top_k_scores, top_k_indices = torch.topk(probabilities, k=self.topk, dim=1)  # [batch_size, top_k]

                # Map user index to username
                user_indices_np = top_k_indices.cpu().numpy()
                # Create reverse user mapping
                user_mapping_inv = {idx: user for user, idx in self.user_mapping.items()}
                # Traverse each element (per user index) in user_indices_np and 
                # use user_mapping_inv.get to query the username corresponding to each index
                user_names_array = np.vectorize(user_mapping_inv.get)(user_indices_np)

                # Obtain the corresponding issue number
                issue_global_indices = subgraph['issue'].n_id.cpu().numpy()  # Global index
                # Create reverse issue mapping
                issue_mapping_inv = {idx: number for number, idx in self.issue_mapping.items()}
                print(f"self.issue_mapping: {self.issue_mapping}")
                open_issue_numbers = [issue_mapping_inv[idx] for idx in issue_global_indices]
                print(f"Open issue numbers: {open_issue_numbers}")

                # Save prediction results
                for issue_number, user_names, scores in zip(open_issue_numbers, user_names_array, top_k_scores.cpu().numpy()):
                    probabilities_list = scores.tolist()
                    user_names_list = user_names.tolist()
                    self.save_issue_assign(
                        self.owner, self.name, issue_number,
                        probabilities_list, user_names_list, self.issue_assign_collection
                    )


class HeteroGraphSAGE(nn.Module):
    def __init__(self,in_channels,hidden_channels, out_channels, dropout):
        super(HeteroGraphSAGE, self).__init__()
        self.issue_mlp = nn.Linear(in_channels, in_channels)

        self.conv1 = HeteroConv({
            ('user', 'participate', 'issue'): GraphConv((-1, -1), hidden_channels),
            ('issue', 'rev_participate', 'user'): GraphConv((-1, -1), hidden_channels),
            # Add other relationships as needed
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('user', 'participate', 'issue'): GraphConv(hidden_channels, out_channels),
            ('issue', 'rev_participate', 'user'): GraphConv(hidden_channels, out_channels),
            # Add other relationships as needed
        }, aggr='mean')
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        x_dict = x_dict.copy() 
        # Process node features initially
        x_dict['issue'] = self.issue_mlp(x_dict['issue'])
        # Then perform message passing           
        x_dict = self.conv1(x_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)
        x_dict = {key: self.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)
        return x_dict

  