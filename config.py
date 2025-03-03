"""
Configuration settings for the trade prediction system
"""

# Data paths
DATA_DIR = "data/"
MODELS_DIR = "saved_models/"

# Model parameters
GNN_PARAMS = {
    "hidden_channels": 64,
    "out_channels": 32,
    "learning_rate": 0.01,
    "epochs": 100,
    "early_stopping": 10
}

# RAG parameters
RAG_PARAMS = {
    "model_name": "google/flan-t5-base",
    "max_length": 512,
    "temperature": 0.7
}

# Analysis parameters
ANALYSIS_PARAMS = {
    "min_probability": 0.5,
    "top_k": 10,
    "underutilized_threshold": 25  # percentile
}

# Visualization parameters
VIZ_PARAMS = {
    "network_seed": 42,
    "node_size_factor": 10,
    "regular_edge_width": 0.5,
    "opportunity_edge_width": 2,
    "regular_edge_color": "#888",
    "opportunity_edge_color": "green"
}

# Streamlit parameters
STREAMLIT_PARAMS = {
    "title": "Trade Opportunity Prediction System",
    "page_icon": "🌐",
    "layout": "wide"
}

# Logging parameters
LOGGING_PARAMS = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}