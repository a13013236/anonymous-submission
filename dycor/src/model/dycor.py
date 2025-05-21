import torch
import torch.nn as nn

from encoding import StockEncoding
from clustering import DynamicStockClustering
from intra_corr import IntraStockCorrelation
from aggregation import InterClusterAggregator

class DYCOR(nn.Module):
    """
    Dynamic Correlation model for stock trend prediction.
    This is the main model that integrates all components:
    1. Stock Encoding - encodes historical features of stocks
    2. Dynamic Stock Clustering - clusters stocks into market segments and subsegments
    3. Intra-Stock Correlation - models relationships between stocks within subcluster
    4. Inter-Cluster Aggregation - combines multiple views of each stock
    5. Prediction - forecasts stock return ratios
    """
    def __init__(
        self,
        stock_num,           # Number of stocks in the market
        lookback_length,     # Length of historical time window
        fea_num,             # Number of features per stock
        hidden_dim,          # Dimension of hidden representations
        min_var_ratio,       # Minimum variance ratio for PCA clustering
        n_subclusters,       # Number of subclusters per market segment
        temperature,         # Temperature for softmax in clustering
        dropout              # Dropout probability
    ):
        
        super().__init__()
        self.stock_num = stock_num
        self.lookback_length = lookback_length
        self.fea_num = fea_num
        self.hidden_dim = hidden_dim
        self.n_subclusters = n_subclusters

        # Stock encoding module: processes historical features to create embeddings
        self.stock_encoding = StockEncoding(
            time_steps=lookback_length,
            channels=fea_num
        )

        # Clustering module: groups stocks into dynamic market segments
        self.clustering = DynamicStockClustering(
            embedding_dim=hidden_dim,
            min_var_ratio=min_var_ratio,
            temperature=temperature,
            n_subclusters=n_subclusters
        )

        # Intra-correlation module: models relationships within segments
        self.intra_corr = IntraStockCorrelation(
            hidden_dim=hidden_dim,
            n_subclusters=n_subclusters,
            dropout=dropout
        )

        # Aggregation module: combines representations from different segments
        self.aggregation = InterClusterAggregator(
            hidden_dim=hidden_dim,
            n_subclusters=n_subclusters
        )

        # Prediction module: multilayer perceptron to forecast return ratios
        self.prediction = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch_x):
        """
        Forward pass of the DYCOR model.
        
        Args:
            batch_x: Batch of stock features [N, fea_num, lookback_length]
            
        Returns:
            pred: Predicted return ratios [N, 1]
            clustering_info: Dictionary containing clustering information for analysis
        """
        # 1. Encode historical stock features into embeddings
        stock_embs = self.stock_encoding(batch_x)
        
        # 2. Perform dynamic stock clustering
        soft_cluster_weights, latent_subsegment_assignments, current_n_clusters, market_stock_similarities, market_reps = self.clustering(stock_embs)
        
        # 3. Model intra-stock correlations within subclusters
        interval_outputs = self.intra_corr(stock_embs, soft_cluster_weights, latent_subsegment_assignments, current_n_clusters)
        
        # 4. Aggregate stock representations across clusters
        final_stock_reps = self.aggregation(stock_embs, soft_cluster_weights, latent_subsegment_assignments, interval_outputs, current_n_clusters)
        
        # 5. Make predictions for stock return ratios
        pred = self.prediction(final_stock_reps)

        # Return additional information for analysis and visualization
        clustering_info = {
            'market_stock_similarities': market_stock_similarities,
            'market_reps': market_reps,
            'soft_cluster_weights': soft_cluster_weights,
            'latent_subsegment_assignments': latent_subsegment_assignments
        }

        return pred, clustering_info
