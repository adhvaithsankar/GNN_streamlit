"""
Helper functions for the trade prediction system
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional


def preprocess_comtrade_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess comtrade data.
    
    Args:
        df: Raw comtrade DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    # Drop rows with missing values in key columns
    df = df.dropna(subset=['exporter_id', 'importer_id', 'hs_code', 'value'])
    
    # Convert value to numeric if it's not already
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    # Filter out rows with zero or negative values
    df = df[df['value'] > 0]
    
    # Create 2-digit HS code
    df['hs2'] = df['hs_code'].astype(str).str[:2]
    
    # Convert year to numeric
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)
    
    return df


def create_sample_data(output_dir: str):
    """
    Create sample datasets for testing.
    
    Args:
        output_dir: Directory to save sample data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample countries
    countries = {
        1: "United States",
        2: "China", 
        3: "Germany",
        4: "Japan",
        5: "United Kingdom",
        6: "France", 
        7: "India",
        8: "Italy",
        9: "Brazil",
        10: "Canada"
    }
    
    # Sample HS codes and products
    products = {
        "01": "Live animals",
        "02": "Meat products",
        "10": "Cereals",
        "27": "Mineral fuels",
        "30": "Pharmaceutical products",
        "39": "Plastics",
        "72": "Iron and steel",
        "84": "Machinery",
        "85": "Electrical machinery",
        "87": "Vehicles"
    }
    
    # Create comtrade data
    comtrade_data = []
    year = 2022
    
    for exporter_id, exporter_name in countries.items():
        for importer_id, importer_name in countries.items():
            if exporter_id != importer_id:  # No self-trade
                # Each country pair trades in some random products
                for hs_code, product_name in list(products.items()):
                    # Not all pairs trade all products
                    if np.random.random() < 0.7:  
                        value = np.random.lognormal(mean=10, sigma=2)  # Trade values follow log-normal distribution
                        quantity = int(value / np.random.uniform(10, 100))  # Random quantity
                        
                        comtrade_data.append({
                            'year': year,
                            'exporter_id': exporter_id,
                            'exporter_name': exporter_name,
                            'importer_id': importer_id,
                            'importer_name': importer_name,
                            'hs_code': hs_code,
                            'product_name': product_name,
                            'hs_revision': 2017,
                            'value': value,
                            'quantity': quantity,
                            'unit_abbrevation': 'kg',
                            'unit_name': 'kilograms'
                        })
    
    # Create GDP data
    gdp_data = []
    for country_id, country_name in countries.items():
        gdp = np.random.uniform(500e9, 20e12)  # Random GDP between $500B and $20T
        gdp_data.append({
            'country_id': country_id,
            'country_name': country_name,
            'year': year,
            'gdp': gdp
        })
    
    # Create population data
    population_data = []
    for country_id, country_name in countries.items():
        population = np.random.uniform(10e6, 1.4e9)  # Random population between 10M and 1.4B
        population_data.append({
            'country_id': country_id,
            'country_name': country_name,
            'year': year,
            'population': population
        })
    
    # Create distance data
    distance_data = []
    for country1_id, country1_name in countries.items():
        for country2_id, country2_name in countries.items():
            if country1_id < country2_id:  # Avoid duplicates
                distance = np.random.uniform(500, 15000)  # Random distance in km
                distance_data.append({
                    'country1_id': country1_id,
                    'country1_name': country1_name,
                    'country2_id': country2_id,
                    'country2_name': country2_name,
                    'distance': distance
                })
    
    # Write to CSV
    pd.DataFrame(comtrade_data).to_csv(os.path.join(output_dir, 'comtrade_data.csv'), index=False)
    pd.DataFrame(gdp_data).to_csv(os.path.join(output_dir, 'gdp_data.csv'), index=False)
    pd.DataFrame(population_data).to_csv(os.path.join(output_dir, 'population_data.csv'), index=False)
    pd.DataFrame(distance_data).to_csv(os.path.join(output_dir, 'distance_data.csv'), index=False)
    
    print(f"Sample data created in {output_dir}")


def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, task: str = "classification") -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task: Task type ("classification" or "regression")
        
    Returns:
        Dictionary of performance metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    results = {}
    
    if task == "classification":
        results["accuracy"] = accuracy_score(y_true, y_pred > 0.5)
        results["precision"] = precision_score(y_true, y_pred > 0.5)
        results["recall"] = recall_score(y_true, y_pred > 0.5)
        results["f1"] = f1_score(y_true, y_pred > 0.5)
        results["auc"] = roc_auc_score(y_true, y_pred)
    else:  # regression
        results["mse"] = mean_squared_error(y_true, y_pred)
        results["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        results["mae"] = mean_absolute_error(y_true, y_pred)
        results["r2"] = r2_score(y_true, y_pred)
    
    return results