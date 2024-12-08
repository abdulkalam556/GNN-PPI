import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool as gmp

class ProteinInteractionGNN(nn.Module):
    def __init__(self, num_features_pro, output_dim=128, dropout=0.2, model_type="gcn", heads=1):
        """
        Protein Interaction GNN model supporting GCN and GAT backbones.
        
        Args:
            num_features_pro (int): Number of input features for each node.
            output_dim (int): Output dimension for graph embeddings.
            dropout (float): Dropout rate.
            model_type (str): Backbone type - "gcn" or "gat".
            heads (int): Number of attention heads (only used for GAT).
        """
        super(ProteinInteractionGNN, self).__init__()
        
        self.model_type = model_type
        self.output_dim = output_dim

        # Protein 1 encoder
        if model_type == "gcn":
            self.pro1_conv = GCNConv(num_features_pro, output_dim)
        elif model_type == "gat":
            self.pro1_conv = GATConv(num_features_pro, output_dim, heads=heads, dropout=dropout)
        self.pro1_fc = nn.Linear(output_dim, output_dim)

        # Protein 2 encoder
        if model_type == "gcn":
            self.pro2_conv = GCNConv(num_features_pro, output_dim)
        elif model_type == "gat":
            self.pro2_conv = GATConv(num_features_pro, output_dim, heads=heads, dropout=dropout)
        self.pro2_fc = nn.Linear(output_dim, output_dim)

        # Activation and dropout
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        # Combined layers for classification
        self.fc1 = nn.Linear(2 * output_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, pro1_data, pro2_data):
        """
        Forward pass for the model.
        
        Args:
            pro1_data (torch_geometric.data.Data): Graph data for protein 1.
            pro2_data (torch_geometric.data.Data): Graph data for protein 2.
            
        Returns:
            torch.Tensor: Interaction prediction (sigmoid output).
        """
        # Protein 1 graph embedding
        pro1_x, pro1_edge_index, pro1_batch = pro1_data.x, pro1_data.edge_index, pro1_data.batch
        
        print(f"pro1_x shape: {pro1_x.shape}")
        print(f"pro1_edge_index shape: {pro1_edge_index.shape}")
        print(f"pro1_batch shape: {pro1_batch.shape}")
        print(f"pro1_edge_index min: {pro1_edge_index.min()}, pro1_edge_index max: {pro1_edge_index.max()}, num_nodes: {pro1_x.size(0)}")

        
        pro1_x = self.pro1_conv(pro1_x, pro1_edge_index)
        pro1_x = self.relu(pro1_x)
        pro1_x = gmp(pro1_x, pro1_batch)  # Global pooling
        pro1_x = self.relu(self.pro1_fc(pro1_x))
        pro1_x = self.dropout(pro1_x)

        # Protein 2 graph embedding
        pro2_x, pro2_edge_index, pro2_batch = pro2_data.x, pro2_data.edge_index, pro2_data.batch

        print(f"pro2_x shape: {pro2_x.shape}")
        print(f"pro2_edge_index shape: {pro2_edge_index.shape}")
        print(f"pro2_batch shape: {pro2_batch.shape}")
        print(f"pro2_edge_index min: {pro2_edge_index.min()}, pro2_edge_index max: {pro2_edge_index.max()}, num_nodes: {pro2_x.size(0)}")
        
        pro2_x = self.pro2_conv(pro2_x, pro2_edge_index)
        pro2_x = self.relu(pro2_x)
        pro2_x = gmp(pro2_x, pro2_batch)  # Global pooling
        pro2_x = self.relu(self.pro2_fc(pro2_x))
        pro2_x = self.dropout(pro2_x)

        # Concatenate embeddings
        combined = torch.cat((pro1_x, pro2_x), dim=1)

        # Classification layers
        combined = self.relu(self.fc1(combined))
        combined = self.dropout(combined)
        combined = self.relu(self.fc2(combined))
        combined = self.dropout(combined)
        out = self.out(combined)
        return torch.sigmoid(out)
