"""
Data processor module for loading, cleaning, and preparing trade data
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """Process and prepare trade, GDP, population, and distance data for modeling."""
    
    def __init__(self, data_path: str = "data/"):
        """
        Initialize the data processor.
        
        Args:
            data_path: Directory containing data files
        """
        self.data_path = r"D:\Final Year Project\RAG_GNN implementation\data"
        self.trade_data = None
        self.gdp_data = None
        self.population_data = None
        self.distance_data = None
        self.product_categories = None
        self.country_metadata = None
        self.G = None  # NetworkX graph
        self.pyg_data = None  # PyTorch Geometric data
        
    def load_data(self):
        """Load all required datasets."""
        # Load trade data
        self.trade_data = pd.read_csv(os.path.join(self.data_path, "comtrade_data.csv"))
        
        # Load GDP data
        self.gdp_data = pd.read_csv(os.path.join(self.data_path, "gdp_data.csv"))
        
        # Load population data
        self.population_data = pd.read_csv(os.path.join(self.data_path, "population_data.csv"))
        
        # Load distance data
        self.distance_data = pd.read_csv(os.path.join(self.data_path, "distance_data.csv"))
        
        return self
    
    def preprocess_data(self):
        """Clean and preprocess the datasets."""
        # Handle missing values
        self.trade_data.dropna(subset=['exporter_id', 'importer_id', 'hs_code', 'value'], inplace=True)
        
        # Aggregate trade data by year, exporter, importer, and HS code
        self.trade_data = self.trade_data.groupby(
            ['year', 'exporter_id', 'exporter_name', 'importer_id', 'importer_name', 'hs_code', 'product_name']
        )['value'].sum().reset_index()
        
        # Create product categories at the 2-digit HS level
        self.trade_data['hs2'] = self.trade_data['hs_code'].astype(str).str[:2]
        self.product_categories = self.trade_data.groupby('hs2')['product_name'].first().reset_index()
        
        # Process GDP and population data
        # Ensure we have data for all countries in trade data
        all_countries = set(self.trade_data['exporter_id'].unique()) | set(self.trade_data['importer_id'].unique())
        
        # Create country metadata
        self.country_metadata = pd.DataFrame({'country_id': list(all_countries)})
        
        # Merge GDP and population data
        self.country_metadata = self.country_metadata.merge(
            self.gdp_data, how='left', left_on='country_id', right_on='country_id'
        ).merge(
            self.population_data, how='left', left_on='country_id', right_on='country_id'
        )
        
        # Fill missing values with medians
        self.country_metadata['gdp'] = self.country_metadata['gdp'].fillna(self.country_metadata['gdp'].median())
        self.country_metadata['population'] = self.country_metadata['population'].fillna(
            self.country_metadata['population'].median()
        )
        
        # Add GDP per capita
        self.country_metadata['gdp_per_capita'] = self.country_metadata['gdp'] / self.country_metadata['population']
        
        # Normalize features
        scaler = StandardScaler()
        self.country_metadata[['gdp', 'population', 'gdp_per_capita']] = scaler.fit_transform(
            self.country_metadata[['gdp', 'population', 'gdp_per_capita']]
        )
        
        return self
    
    def build_trade_graph(self, year: int = None):
        """
        Build a trade network graph.
        
        Args:
            year: Specific year to filter data for (default: most recent year)
        """
        if year is None:
            year = self.trade_data['year'].max()
        
        # Filter data for the given year
        year_data = self.trade_data[self.trade_data['year'] == year]
        
        # Create a graph
        self.G = nx.DiGraph()
        
        # Add country nodes
        for _, row in self.country_metadata.iterrows():
            self.G.add_node(
                row['country_id'],
                gdp=row['gdp'],
                population=row['population'],
                gdp_per_capita=row['gdp_per_capita']
            )
        
        # Add trade edges
        for _, row in year_data.iterrows():
            exporter = row['exporter_id']
            importer = row['importer_id']
            hs_code = row['hs_code']
            value = row['value']
            
            if self.G.has_edge(exporter, importer):
                # Add to existing edge attributes
                self.G[exporter][importer]['trade_volume'] += value
                self.G[exporter][importer]['products'].add(hs_code)
            else:
                # Create new edge
                self.G.add_edge(
                    exporter, 
                    importer, 
                    trade_volume=value,
                    products={hs_code},
                    weight=value
                )
        
        # Add distance between countries if available
        if self.distance_data is not None:
            for _, row in self.distance_data.iterrows():
                country1 = row['country1_id']
                country2 = row['country2_id']
                distance = row['distance']
                
                if self.G.has_edge(country1, country2):
                    self.G[country1][country2]['distance'] = distance
                if self.G.has_edge(country2, country1):
                    self.G[country2][country1]['distance'] = distance
        
        return self
    
    def convert_to_pytorch_geometric(self):
        """Convert NetworkX graph to PyTorch Geometric format."""
        # Extract node features
        node_ids = list(self.G.nodes())
        node_mapping = {node_id: i for i, node_id in enumerate(node_ids)}
        
        # Extract node features from graph
        x = []
        for node_id in node_ids:
            features = [
                self.G.nodes[node_id].get('gdp', 0),
                self.G.nodes[node_id].get('population', 0),
                self.G.nodes[node_id].get('gdp_per_capita', 0)
            ]
            x.append(features)
        
        x = torch.tensor(x, dtype=torch.float)
        
        # Extract edges and edge features
        edge_index = []
        edge_attr = []
        trade_volumes = []
        
        for source, target, data in self.G.edges(data=True):
            edge_index.append([node_mapping[source], node_mapping[target]])
            
            # Edge features: trade_volume, number of products, distance (if available)
            edge_features = [
                data.get('trade_volume', 0),
                len(data.get('products', set())),
                data.get('distance', 0)  # Use 0 if distance is not available
            ]
            edge_attr.append(edge_features)
            trade_volumes.append(data.get('trade_volume', 0))
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Create PyG data object
        self.pyg_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_mapping=node_mapping,
            reverse_node_mapping={v: k for k, v in node_mapping.items()},
            trade_volumes=torch.tensor(trade_volumes, dtype=torch.float)
        )
        
        return self