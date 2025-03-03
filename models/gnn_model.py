"""
Graph Neural Network model for trade link prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GAT


class RAGEnhancedGNN(nn.Module):
    """
    Graph Neural Network enhanced with Retrieval-Augmented Generation capabilities
    for trade link prediction.
    """
    
    def __init__(
        self, 
        node_features: int, 
        edge_features: int,
        hidden_channels: int = 64, 
        out_channels: int = 32
    ):
        """
        Initialize the RAG-enhanced GNN model.
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            hidden_channels: Hidden layer size
            out_channels: Output size
        """
        super(RAGEnhancedGNN, self).__init__()
        
        # Graph convolutional layers
        self.conv1 = GCNConv(node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Edge feature processing
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Final node embeddings
        self.final_layer = nn.Linear(hidden_channels, out_channels)
        
        # Edge prediction from node embeddings
        self.edge_predictor = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        
        # For predicting trade volume
        self.volume_predictor = nn.Sequential(
            nn.Linear(out_channels * 2 + edge_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, data):
        """
        Forward pass through the network.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            node_embeddings: Embeddings for each node
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Node embeddings
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        node_embeddings = self.final_layer(x)
        
        return node_embeddings
    
    def predict_link(self, node_embeddings, source_idx, target_idx):
        """
        Predict likelihood of a trade link between two countries.
        
        Args:
            node_embeddings: Embeddings for all nodes
            source_idx: Source country index
            target_idx: Target country index
            
        Returns:
            score: Probability of a trade relationship
        """
        # Get embeddings for the source and target countries
        source_emb = node_embeddings[source_idx]
        target_emb = node_embeddings[target_idx]
        
        # Concatenate embeddings
        pair_emb = torch.cat([source_emb, target_emb], dim=-1)
        
        # Predict link
        score = self.edge_predictor(pair_emb)
        return score.squeeze()
    
    def predict_trade_volume(self, node_embeddings, source_idx, target_idx, edge_features=None):
        """
        Predict trade volume between two countries.
        
        Args:
            node_embeddings: Embeddings for all nodes
            source_idx: Source country index
            target_idx: Target country index
            edge_features: Features of the edge (optional)
            
        Returns:
            volume: Predicted trade volume
        """
        # Get embeddings for the source and target countries
        source_emb = node_embeddings[source_idx]
        target_emb = node_embeddings[target_idx]
        
        # Concatenate embeddings
        pair_emb = torch.cat([source_emb, target_emb], dim=-1)
        
        if edge_features is not None:
            # If we have edge features, include them
            pair_emb = torch.cat([pair_emb, edge_features], dim=-1)
        else:
            # Default edge features (zeros)
            default_edge_features = torch.zeros(pair_emb.shape[0], 3)
            pair_emb = torch.cat([pair_emb, default_edge_features], dim=-1)
        
        # Predict volume
        volume = self.volume_predictor(pair_emb)
        return volume.squeeze()