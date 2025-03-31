import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pyvis.network import Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
import plotly.express as px
import plotly.graph_objects as go
import os
import pickle
import tempfile
import base64
from pathlib import Path
import io
import streamlit as st
import requests
import sqlite3
import google.generativeai as genai
from bs4 import BeautifulSoup

GENAI_API_KEY = st.secrets["GENAI_API_KEY"]


# Configure Generative AI model
genai.configure(api_key=GENAI_API_KEY)
model_gen = genai.GenerativeModel("gemini-1.5-flash")

# Database setup
DB_NAME = "news.db"

def create_table():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            url TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

create_table()

def insert_news(title, summary, url):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO news (title, summary, url) VALUES (?, ?, ?)", (title, summary, url))
    conn.commit()
    conn.close()

def load_news():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, summary, url FROM news")
    news_list = cursor.fetchall()
    conn.close()
    return news_list

# Web scraping function to fetch trade summary from WITS
def fetch_trade_summary(country, year):
    url = f"https://wits.worldbank.org/CountryProfile/en/Country/{country}/Year/{year}/SummaryText"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Print the page structure for debugging
        print(soup.prettify())

        # Adjust the selector based on actual HTML structure
        summary_div = soup.find("div", class_="col-md-9 posRelative")
        
        if summary_div:
            return summary_div.text.strip()
        else:
            return "Trade summary not found. The webpage structure might have changed."

    except Exception as e:
        return f"Error fetching trade summary: {e}"
    

# Function to get the title from the article
def get_title(soup):
    title_tag = soup.find('h1')  # Assuming the title is within an <h1> tag
    return title_tag.get_text(strip=True) if title_tag else "No title found"

# Function to get the description from the article
def get_description(soup):
    description_tag = soup.find('meta', {'name': 'description'})  # Assuming the description is in a <meta> tag
    return description_tag['content'] if description_tag else "No description available"

# Scrape news articles from the website
# Scrape news articles from the website
def scrape_news():
    URL = "https://indianexpress.com/about/world-trade-organization/"
    HEADERS = {'User-agent': 'Mozilla/5.0', 'Accept-Language': 'en-US, en;q=0.5'}

    try:
        # Fetch the main page content
        webpage = requests.get(URL, headers=HEADERS)
        webpage.raise_for_status()  # Raise error for bad HTTP status codes
        soup = BeautifulSoup(webpage.content, "html.parser")

        # Find article links
        articles = soup.find_all('div', class_='details')
        article_links = [article.find('a', href=True)['href'] for article in articles if article.find('a', href=True)]

        # Extract title and description from each article
        for article_link in article_links:
            try:
                # Full URL if the article link is relative
                if article_link.startswith('/'):
                    article_link = f"https://indianexpress.com{article_link}"

                # Fetch the article page
                new_webpage = requests.get(article_link, headers=HEADERS)
                new_webpage.raise_for_status()
                new_soup = BeautifulSoup(new_webpage.content, "html.parser")
                title = get_title(new_soup)
                description = get_description(new_soup)

                # Check if the article title already exists in the database
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM news WHERE title = ?", (title,))
                count = cursor.fetchone()[0]
                conn.close()

                # If the title doesn't exist, insert the new article
                if count == 0:
                    conn = sqlite3.connect(DB_NAME)  # Ensure you're using the correct DB
                    cursor = conn.cursor()
                    cursor.execute('INSERT INTO news (title, summary, url) VALUES (?, ?, ?)', (title, description, article_link))
                    conn.commit()
                    conn.close()
                    print(f"Inserted article: {title}")  # Debugging print statement
                else:
                    print(f"Article with title '{title}' already exists. Skipping insertion.")

            except requests.RequestException as e:
                print(f"Error fetching article from {article_link}: {e}")
            except Exception as e:
                print(f"Error processing article {article_link}: {e}")

    except requests.RequestException as e:
        print(f"Error fetching main page: {e}")


# Set page configuration
st.set_page_config(
    page_title="Global Trade Network Analysis",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the model architecture (same as your original code for loading)
class TemporalGAT(nn.Module):
    def __init__(self, in_feats, h_feats=128):
        super().__init__()
        self.h_feats = h_feats
        
        # First GAT layer with multi-head attention
        self.gat1 = GATConv(
            in_feats, 
            h_feats//2,  # h_feats/2 per head * 2 heads = h_feats total
            num_heads=2,
            feat_drop=0.3,
            attn_drop=0.3,
            negative_slope=0.2,
            residual=False,
            allow_zero_in_degree=True
        )
        
        # Second GAT layer with single-head attention
        self.gat2 = GATConv(
            h_feats,  # Input size from concatenated heads
            h_feats,  # Output size
            num_heads=1,
            feat_drop=0.3,
            attn_drop=0.3,
            negative_slope=0.2,
            residual=True,
            allow_zero_in_degree=True
        )
        
        # Projection for residual connection from input to output
        self.proj = nn.Linear(in_feats, h_feats)
        self.norm1 = nn.LayerNorm(h_feats)
        self.norm2 = nn.LayerNorm(h_feats)

    def forward(self, g, x):
        # Residual connection path
        h_residual = self.proj(x)
        
        # First GAT layer - multi-head attention
        h = self.gat1(g, x)  # [nodes, num_heads, h_feats//2]
        h = h.flatten(1)     # [nodes, num_heads * h_feats//2]
        h = F.elu(self.norm1(h))
        
        # Second GAT layer - single-head attention
        h = self.gat2(g, h)  # [nodes, 1, h_feats]
        h = h.squeeze(1)     # [nodes, h_feats]
        h = self.norm2(h + h_residual)  # Add custom residual connection
        
        return h

class LinkPredictor(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.fc1 = nn.Linear(h_dim * 2, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)
        
    def forward(self, h_src, h_dst):
        # Ensure tensors have appropriate dimensions before concatenation
        if h_src.dim() == 1:
            h_src = h_src.unsqueeze(0)  # Add batch dimension if it's a single vector
        if h_dst.dim() == 1:
            h_dst = h_dst.unsqueeze(0)  # Add batch dimension if it's a single vector
            
        # Now concatenate along dimension 1
        h = torch.cat([h_src, h_dst], dim=1)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return h.squeeze()  # Remove extra dimensions for single predictions

# Helper functions for data preprocessing and model loading
def load_data(file_path):
    """Load and preprocess trade data"""
    df = pd.read_csv(file_path)
    if 'exporter_name' not in df.columns and 'importer_name' not in df.columns:
        st.warning("Using basic dataset version. For enhanced functionality, use the extended dataset with country names.")
    return df

def create_country_mapping(df):
    """Create mapping from country ID to index"""
    unique_exporters = df['exporter_id'].unique()
    unique_importers = df['importer_id'].unique()
    unique_countries = np.unique(np.concatenate([unique_exporters, unique_importers]))
    country_to_idx = {country: idx for idx, country in enumerate(unique_countries)}
    idx_to_country = {idx: country for country, idx in country_to_idx.items()}
    
    # If extended dataset has country names, create name mappings too
    country_id_to_name = {}
    if 'exporter_name' in df.columns:
        exporter_map = df[['exporter_id', 'exporter_name']].drop_duplicates().set_index('exporter_id')['exporter_name'].to_dict()
        importer_map = df[['importer_id', 'importer_name']].drop_duplicates().set_index('importer_id')['importer_name'].to_dict()
        country_id_to_name.update(exporter_map)
        country_id_to_name.update(importer_map)
    
    return country_to_idx, idx_to_country, country_id_to_name

def prepare_node_features(df, country_to_idx, year):
    """Prepare node features for the model"""
    # Filter data for selected year
    year_data = df[df['year'] == year]
    
    # Calculate aggregate trade statistics for each country
    country_stats = {}
    for country_id in country_to_idx.keys():
        # Export statistics
        export_data = year_data[year_data['exporter_id'] == country_id]
        total_exports = export_data['value'].sum()
        unique_export_partners = export_data['importer_id'].nunique()
        
        # Import statistics
        import_data = year_data[year_data['importer_id'] == country_id]
        total_imports = import_data['value'].sum()
        unique_import_partners = import_data['exporter_id'].nunique()
        
        # Store statistics
        country_stats[country_id] = {
            'total_exports': total_exports,
            'total_imports': total_imports,
            'export_partners': unique_export_partners,
            'import_partners': unique_import_partners,
            'trade_balance': total_exports - total_imports
        }
    
    # Create features tensor
    num_nodes = len(country_to_idx)
    features = torch.zeros((num_nodes, 5), dtype=torch.float32)
    
    for country_id, stats in country_stats.items():
        idx = country_to_idx[country_id]
        features[idx, 0] = np.log1p(stats['total_exports'])
        features[idx, 1] = np.log1p(stats['total_imports'])
        features[idx, 2] = stats['export_partners']
        features[idx, 3] = stats['import_partners']
        features[idx, 4] = stats['trade_balance']
    
    # Normalize features
    scaler = StandardScaler()
    features_np = features.numpy()
    features_scaled = scaler.fit_transform(features_np)
    
    return torch.FloatTensor(features_scaled)

def create_trade_graph(df, year, country_to_idx):
    """Create a DGL graph from trade data for a specific year"""
    # Filter data for selected year
    year_data = df[df['year'] == year]
    
    # Aggregate trade values for each country pair
    agg_data = year_data.groupby(['exporter_id', 'importer_id'])['value'].sum().reset_index()
    
    # Map country IDs to indices
    edges_src = [country_to_idx[exp] for exp in agg_data['exporter_id']]
    edges_dst = [country_to_idx[imp] for imp in agg_data['importer_id']]
    
    # Create DGL graph
    g = dgl.graph((torch.tensor(edges_src), torch.tensor(edges_dst)), 
                  num_nodes=len(country_to_idx))
    
    # Add edge weights (normalized log values)
    edge_values = np.log1p(agg_data['value'].values)
    scaler = MinMaxScaler()
    edge_values_norm = scaler.fit_transform(edge_values.reshape(-1, 1)).flatten()
    g.edata['weight'] = torch.FloatTensor(edge_values_norm)
    
    return g

def load_gat_model(model_path, predictor_path, in_feats, h_feats=128):
    """Load trained GAT model"""
    model = TemporalGAT(in_feats, h_feats)
    predictor = LinkPredictor(h_feats)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        predictor.load_state_dict(torch.load(predictor_path, map_location=torch.device('cpu')))
        model.eval()
        predictor.eval()
        return model, predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def get_embeddings(model, graph, features):
    """Generate node embeddings using the GAT model"""
    with torch.no_grad():
        embeddings = model(graph, features)
    return embeddings

def predict_link_probability(predictor, src_embedding, dst_embedding):
    """Predict probability of a trade link between two countries"""
    with torch.no_grad():
        logit = predictor(src_embedding, dst_embedding)
        prob = torch.sigmoid(logit).item()
    return prob

# Visualization functions
def plot_network_graph(df, year, country_id_to_name, min_value=0):
    """Create a Pyvis network visualization of the trade network"""
    # Filter data for selected year
    year_data = df[df['year'] == year]
    year_data = year_data[year_data['value'] > min_value]
    
    # Create network
    G = nx.DiGraph()
    
    # Add nodes
    all_countries = set(year_data['exporter_id']).union(set(year_data['importer_id']))
    for country in all_countries:
        name = country_id_to_name.get(country, f"Country {country}")
        G.add_node(country, label=name, title=name)
    
    # Add edges
    for _, row in year_data.iterrows():
        exp_id = row['exporter_id']
        imp_id = row['importer_id']
        value = row['value']
        if exp_id != imp_id:  # Skip self-loops
            G.add_edge(exp_id, imp_id, weight=np.log1p(value), title=f"${value:,.2f}")
    
    # Create Pyvis network
    net = Network(height="600px", width="100%", directed=True, notebook=False)
    
    # Customize appearance
    net.from_nx(G)
    net.toggle_physics(True)
    net.set_edge_smooth('dynamic')
    
    # Generate HTML file
    html_path = "network_graph.html"
    net.save_graph(html_path)
    
    # Read the HTML file and return it
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    return html_content

def plot_feature_distribution(df, year, feature_type='value'):
    """Plot distribution of trade values or other features"""
    # Filter data for selected year
    year_data = df[df['year'] == year]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if feature_type == 'value':
        # Plot log-transformed trade values
        log_values = np.log1p(year_data['value'])
        sns.histplot(log_values, kde=True, ax=ax)
        ax.set_xlabel('Log(1+Trade Value)')
        ax.set_title(f'Distribution of Trade Values (Log Scale) - {year}')
    elif feature_type == 'country_activity':
        # Count trade activities by country
        exporters = year_data['exporter_id'].value_counts()
        importers = year_data['importer_id'].value_counts()
        
        # Combine and take top countries
        combined = pd.concat([
            exporters.rename('exports'), 
            importers.rename('imports')
        ], axis=1).fillna(0)
        
        combined['total'] = combined['exports'] + combined['imports']
        combined = combined.sort_values('total', ascending=False).head(20)
        
        # Create stacked bar chart
        combined[['exports', 'imports']].plot(kind='bar', stacked=True, ax=ax)
        ax.set_title(f'Trade Activity by Top Countries - {year}')
        ax.set_xlabel('Country ID')
        ax.set_ylabel('Number of Trade Connections')
        plt.xticks(rotation=45)
    
    return fig

def visualize_embeddings(embeddings, labels, method='tsne', perplexity=30, n_components=2):
    """Visualize node embeddings using t-SNE or PCA"""
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    else:  # PCA
        reducer = PCA(n_components=n_components)
    
    reduced_embeddings = reducer.fit_transform(embeddings.numpy())
    
    # Create DataFrame for plotting
    df = pd.DataFrame(reduced_embeddings, columns=[f'Component {i+1}' for i in range(n_components)])
    df['Country'] = labels
    
    # Create interactive plot
    if n_components == 2:
        fig = px.scatter(
            df, x='Component 1', y='Component 2',
            hover_data=['Country'],
            color='Country',
            title=f"{method.upper()} Visualization of Country Embeddings"
        )
    else:  # 3D plot for 3 components
        fig = px.scatter_3d(
            df, x='Component 1', y='Component 2', z='Component 3',
            hover_data=['Country'],
            color='Country',
            title=f"{method.upper()} Visualization of Country Embeddings"
        )
    
    fig.update_layout(height=600, width=800)
    return fig

def predict_potential_trading_partners(predictor, embeddings, country_idx, idx_to_country, country_id_to_name, top_n=10):
    """Predict potential trading partners for a given country"""
    country_embedding = embeddings[country_idx]
    
    results = []
    for idx in range(len(embeddings)):
        if idx != country_idx:  # Skip self-predictions
            prob = predict_link_probability(predictor, country_embedding, embeddings[idx])
            country_id = idx_to_country[idx]
            country_name = country_id_to_name.get(country_id, f"Country {country_id}")
            results.append((country_id, country_name, prob))
    
    # Sort by probability in descending order
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]

def find_trade_communities(graph, embeddings, n_clusters=5, method='kmeans'):
    """Find communities of countries with similar trading patterns"""
    # Use embeddings for clustering
    if method == 'kmeans':
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
    else:  # spectral clustering
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    
    clusters = clustering.fit_predict(embeddings.numpy())
    return clusters

def calculate_centrality_metrics(df, year):
    """Calculate various centrality metrics for countries in the trade network"""
    # Filter data for selected year
    year_data = df[df['year'] == year]
    
    # Create NetworkX graph for centrality calculations
    G = nx.DiGraph()
    for _, row in year_data.iterrows():
        G.add_edge(row['exporter_id'], row['importer_id'], weight=np.log1p(row['value']))
    
    # Calculate centrality metrics
    degree_centrality = nx.degree_centrality(G)
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    
    try:
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
    except:
        eigenvector_centrality = {node: 0 for node in G.nodes()}
        st.warning("Eigenvector centrality calculation failed. Using zeros instead.")
    
    # Combine metrics into DataFrame
    centrality_df = pd.DataFrame({
        'Country': list(degree_centrality.keys()),
        'Degree': list(degree_centrality.values()),
        'In-Degree': list(in_degree_centrality.values()),
        'Out-Degree': list(out_degree_centrality.values()),
        'Betweenness': list(betweenness_centrality.values()),
        'Eigenvector': list(eigenvector_centrality.values())
    })
    
    return centrality_df

def analyze_covid_impact(df, pre_covid_year=2019, covid_year=2020):
    """Analyze the impact of COVID-19 on international trade"""
    # Filter data for pre-COVID and COVID years
    pre_covid_data = df[df['year'] == pre_covid_year]
    covid_data = df[df['year'] == covid_year]
    
    # Calculate aggregate trade volumes
    pre_covid_volume = pre_covid_data['value'].sum()
    covid_volume = covid_data['value'].sum()
    
    # Calculate percentage change
    pct_change = ((covid_volume - pre_covid_volume) / pre_covid_volume) * 100
    
    # Analyze country-level impact
    country_impact = {}
    
    # Exporters
    pre_covid_exports = pre_covid_data.groupby('exporter_id')['value'].sum()
    covid_exports = covid_data.groupby('exporter_id')['value'].sum()
    
    # Calculate percentage change for each exporter
    export_change = pd.DataFrame({
        'pre_covid': pre_covid_exports,
        'covid': covid_exports
    }).fillna(0)
    
    export_change['pct_change'] = ((export_change['covid'] - export_change['pre_covid']) / 
                                  export_change['pre_covid'].replace(0, 1)) * 100
    
    # Similar analysis for importers
    pre_covid_imports = pre_covid_data.groupby('importer_id')['value'].sum()
    covid_imports = covid_data.groupby('importer_id')['value'].sum()
    
    import_change = pd.DataFrame({
        'pre_covid': pre_covid_imports,
        'covid': covid_imports
    }).fillna(0)
    
    import_change['pct_change'] = ((import_change['covid'] - import_change['pre_covid']) / 
                                  import_change['pre_covid'].replace(0, 1)) * 100
    
    # Reset index for easier handling
    export_change = export_change.reset_index().rename(columns={'exporter_id': 'country_id'})
    import_change = import_change.reset_index().rename(columns={'importer_id': 'country_id'})
    
    return {
        'overall_change': pct_change,
        'export_change': export_change,
        'import_change': import_change,
        'pre_covid_volume': pre_covid_volume,
        'covid_volume': covid_volume
    }

# Streamlit app layout and functionality
def main():
    st.title("🌐 Global Trade Network Analysis")
    st.markdown("""
    This application allows you to analyze international trade data, visualize trade networks, 
    and explore predictions from a Graph Attention Network (GAT) model.
    """)
    
    # Sidebar for navigation and controls
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["📊 Data Overview", 
         "🔍 Graph Visualizations", 
         "📈 Feature Distributions", 
         "🧠 Embedding Visualizations", 
         "🤝 Trading Partner Predictions", 
         "🌍 Trade Communities", 
         "🦠 COVID-19 Impact", 
         "🌟 Trade Hub Analysis",
         "📰 Trade News",
         "🌎📉 Country Stats"]
    )
    
    # Data upload section
    st.sidebar.title("Data Input")
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV file", "Use sample data"]
    )
    
    if data_option == "Upload CSV file":
        uploaded_file = st.sidebar.file_uploader("Upload trade data CSV", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Data loaded successfully!")
        else:
            st.sidebar.warning("Please upload a CSV file to proceed.")
            # Load sample data as fallback
            df = pd.DataFrame({
                'year': [2018, 2018, 2019, 2019, 2020, 2020],
                'exporter_id': ['USA', 'CHN', 'USA', 'CHN', 'USA', 'CHN'],
                'exporter_name': ['United States', 'China', 'United States', 'China', 'United States', 'China'],
                'importer_id': ['CHN', 'USA', 'CHN', 'USA', 'CHN', 'USA'],
                'importer_name': ['China', 'United States', 'China', 'United States', 'China', 'United States'],
                'hs_code': ['270900', '847130', '270900', '847130', '270900', '847130'],
                'value': [1000000, 2000000, 1200000, 1800000, 800000, 1500000]
            })
    else:
        # Load sample data
        df = pd.DataFrame({
            'year': [2018, 2018, 2019, 2019, 2020, 2020],
            'exporter_id': ['USA', 'CHN', 'USA', 'CHN', 'USA', 'CHN'],
            'exporter_name': ['United States', 'China', 'United States', 'China', 'United States', 'China'],
            'importer_id': ['CHN', 'USA', 'CHN', 'USA', 'CHN', 'USA'],
            'importer_name': ['China', 'United States', 'China', 'United States', 'China', 'United States'],
            'hs_code': ['270900', '847130', '270900', '847130', '270900', '847130'],
            'value': [1000000, 2000000, 1200000, 1800000, 800000, 1500000]
        })
    
    # Model loading section
    st.sidebar.title("GAT Model")
    model_option = st.sidebar.radio(
        "Choose model option:",
        ["Upload trained model", "Use sample model"]
    )
    
    model = None
    predictor = None
    embeddings = None
    
    if model_option == "Upload trained model":
        model_file = st.sidebar.file_uploader("Upload GAT model (.pt)", type="pt")
        predictor_file = st.sidebar.file_uploader("Upload link predictor (.pt)", type="pt")
        
        if model_file and predictor_file:
            # Save uploaded files temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model:
                tmp_model.write(model_file.getvalue())
                model_path = tmp_model.name
                
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_predictor:
                tmp_predictor.write(predictor_file.getvalue())
                predictor_path = tmp_predictor.name
            
            # Create country mappings
            country_to_idx, idx_to_country, country_id_to_name = create_country_mapping(df)
            
            # Create graph and features for first available year
            available_years = sorted(df['year'].unique())
            if available_years:
                selected_year = st.sidebar.selectbox("Select year for analysis:", available_years)
                graph = create_trade_graph(df, selected_year, country_to_idx)
                features = prepare_node_features(df, country_to_idx, selected_year)
                
                # Load model
                in_feats = features.shape[1]
                model, predictor = load_gat_model(model_path, predictor_path, in_feats)
                
                if model and predictor:
                    # Generate embeddings
                    embeddings = get_embeddings(model, graph, features)
                    st.sidebar.success("Model loaded successfully!")
    else:
        # Use sample model (for demonstration)
        st.sidebar.info("Using sample model for demonstration.")
        
        # Create country mappings
        country_to_idx, idx_to_country, country_id_to_name = create_country_mapping(df)
        
        # Create graph and features for demo
        available_years = sorted(df['year'].unique())
        if available_years:
            selected_year = st.sidebar.selectbox("Select year for analysis:", available_years)
            graph = create_trade_graph(df, selected_year, country_to_idx)
            features = prepare_node_features(df, country_to_idx, selected_year)
            
            # Create dummy model and embeddings for demo
            in_feats = features.shape[1]
            model = TemporalGAT(in_feats)
            predictor = LinkPredictor(128)
            
            # Random embeddings for demo
            embeddings = torch.randn((len(country_to_idx), 128))
    
    # Main content area based on selected mode
    if app_mode == "📊 Data Overview":
        st.header("📊 Trade Data Overview")
        
        st.subheader("Data Sample")
        st.dataframe(df.head())
        
        st.subheader("Data Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Years Covered", f"{df['year'].min()} - {df['year'].max()}")
        with col3:
            st.metric("Total Trade Volume", f"${df['value'].sum():,.2f}")
        
        st.subheader("Countries in Dataset")
        exporters_count = df['exporter_id'].nunique()
        importers_count = df['importer_id'].nunique()
        unique_countries = set(df['exporter_id']).union(set(df['importer_id']))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Unique Exporters", exporters_count)
        with col2:
            st.metric("Unique Importers", importers_count)
        with col3:
            st.metric("Total Unique Countries", len(unique_countries))
        
        # Trade volume by year
        st.subheader("Trade Volume by Year")
        yearly_volume = df.groupby('year')['value'].sum().reset_index()
        fig = px.bar(yearly_volume, x='year', y='value',
                    labels={'value': 'Trade Volume', 'year': 'Year'},
                    title='Total Trade Volume by Year')
        st.plotly_chart(fig)
        
        # Top trading relationships
        st.subheader("Top Trading Relationships")
        if 'exporter_name' in df.columns and 'importer_name' in df.columns:
            # Use country names if available
            top_trades = df.groupby(['exporter_name', 'importer_name'])['value'].sum().reset_index()
            top_trades = top_trades.sort_values('value', ascending=False).head(10)
            top_trades['trade_pair'] = top_trades['exporter_name'] + ' → ' + top_trades['importer_name']
        else:
            # Use country IDs otherwise
            top_trades = df.groupby(['exporter_id', 'importer_id'])['value'].sum().reset_index()
            top_trades = top_trades.sort_values('value', ascending=False).head(10)
            top_trades['trade_pair'] = top_trades['exporter_id'] + ' → ' + top_trades['importer_id']
        
        fig = px.bar(top_trades, x='value', y='trade_pair',
                    orientation='h',
                    labels={'value': 'Trade Volume', 'trade_pair': 'Trading Relationship'},
                    title='Top 10 Trading Relationships')
        st.plotly_chart(fig)
    
    elif app_mode == "🔍 Graph Visualizations":
        st.header("🔍 Trade Network Visualization")
        
        available_years = sorted(df['year'].unique())
        if not available_years:
            st.error("No year data available in the dataset.")
            return
        
        year = st.selectbox("Select year to visualize:", available_years)
        min_value = st.slider("Minimum trade value (filters small trades):", 
                             min_value=0, 
                             max_value=int(df['value'].max() / 10), 
                             value=0)
        
        st.subheader(f"Trade Network in {year}")
        
        if country_id_to_name:
            with st.spinner("Generating network visualization..."):
                html_content = plot_network_graph(df, year, country_id_to_name, min_value)
                st.components.v1.html(html_content, height=600)
        else:
            st.warning("Country name mapping not available. Using country IDs for visualization.")
            with st.spinner("Generating network visualization..."):
                # Create minimal name mapping using IDs
                minimal_mapping = {country_id: str(country_id) for country_id in 
                                  set(df['exporter_id']).union(set(df['importer_id']))}
                html_content = plot_network_graph(df, year, minimal_mapping, min_value)
                st.components.v1.html(html_content, height=600)
        
        st.info("Hover over nodes to see country names and over edges to see trade values. You can zoom, pan, and drag nodes to explore the network.")
    elif app_mode == "📈 Feature Distributions":
        st.header("📈 Trade Feature Distributions")
        
        available_years = sorted(df['year'].unique())
        if not available_years:
            st.error("No year data available in the dataset.")
            return
        
        year = st.selectbox("Select year to analyze:", available_years)
        feature_type = st.radio("Select feature to visualize:", 
                              ["Trade Values", "Country Activity"],
                              format_func=lambda x: x)
        
        if feature_type == "Trade Values":
            st.subheader(f"Distribution of Trade Values in {year}")
            fig = plot_feature_distribution(df, year, 'value')
            st.pyplot(fig)
            
            # Add additional statistics
            year_data = df[df['year'] == year]
            st.metric("Total Trade Volume", f"${year_data['value'].sum():,.2f}")
            st.metric("Average Trade Value", f"${year_data['value'].mean():,.2f}")
            st.metric("Median Trade Value", f"${year_data['value'].median():,.2f}")
        else:  # Country Activity
            st.subheader(f"Trade Activity by Top Countries in {year}")
            fig = plot_feature_distribution(df, year, 'country_activity')
            st.pyplot(fig)
    
    elif app_mode == "🧠 Embedding Visualizations":
        st.header("🧠 Country Embedding Visualizations")
        
        if embeddings is None:
            st.error("Embeddings not available. Please upload or use a sample model first.")
            return
        
        method = st.radio("Visualization Method:", ["t-SNE", "PCA"])
        dimensions = st.radio("Dimensions:", [2, 3])
        
        if method == "t-SNE":
            perplexity = st.slider("t-SNE Perplexity:", min_value=5, max_value=50, value=30)
        else:
            perplexity = 30  # Not used for PCA
        
        # Prepare labels for countries
        labels = []
        for idx in range(len(embeddings)):
            country_id = idx_to_country[idx]
            if country_id in country_id_to_name:
                labels.append(country_id_to_name[country_id])
            else:
                labels.append(str(country_id))
        
        # Create visualization
        st.subheader(f"{method} Visualization of Country Embeddings")
        with st.spinner(f"Generating {method} visualization..."):
            fig = visualize_embeddings(embeddings, labels, method.lower(), perplexity, dimensions)
            st.plotly_chart(fig)
        
        st.info("Countries positioned close together in this visualization have similar trading patterns according to the GAT model.")
    
    elif app_mode == "🤝 Trading Partner Predictions":
        st.header("🤝 Potential Trading Partner Predictions")
        
        if embeddings is None or predictor is None:
            st.error("Model or embeddings not available. Please upload or use a sample model first.")
            return
        
        # Country selection
        if country_id_to_name:
            # Create sorted list of countries with names
            countries = sorted([(country_id, country_id_to_name.get(country_id, str(country_id))) 
                            for country_id in country_to_idx.keys()], 
                            key=lambda x: x[1])
            
            country_options = {f"{name} ({id})": id for id, name in countries}
            selected_country_display = st.selectbox("Select a country:", list(country_options.keys()))
            selected_country = country_options[selected_country_display]
        else:
            # Just use country IDs
            country_list = sorted(country_to_idx.keys())
            selected_country = st.selectbox("Select a country:", country_list)
            # Define selected_country_display for the else branch too
            selected_country_display = str(selected_country)
        
        # Number of predictions to show
        top_n = st.slider("Number of predictions to show:", min_value=5, max_value=30, value=10)
        
        # Get country index
        country_idx = country_to_idx[selected_country]
        
        # Make predictions
        st.subheader(f"Top {top_n} Potential Trading Partners")
        with st.spinner("Generating predictions..."):
            predictions = predict_potential_trading_partners(
                predictor, embeddings, country_idx, idx_to_country, country_id_to_name, top_n)
            
            # Display results
            results_df = pd.DataFrame(predictions, columns=["Country ID", "Country Name", "Trade Probability"])
            results_df["Trade Probability"] = results_df["Trade Probability"].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(results_df)
            
            # Visualize as bar chart
            fig = px.bar(results_df, x="Trade Probability", y="Country Name", 
                        orientation='h',
                        title=f"Predicted Trading Partners for {selected_country_display}",
                        height=400)
            st.plotly_chart(fig)
    
    elif app_mode == "🌍 Trade Communities":
        st.header("🌍 Trade Community Detection")
        
        if embeddings is None:
            st.error("Embeddings not available. Please upload or use a sample model first.")
            return
        
        # Community detection parameters
        n_clusters = st.slider("Number of communities:", min_value=2, max_value=10, value=5)
        method = st.radio("Clustering method:", ["K-Means", "Spectral Clustering"])
        
        # Perform clustering
        with st.spinner("Detecting trade communities..."):
            clusters = find_trade_communities(graph, embeddings, n_clusters, 
                                             'kmeans' if method == "K-Means" else 'spectral')
            
            # Prepare results
            community_data = []
            for idx, cluster in enumerate(clusters):
                country_id = idx_to_country[idx]
                name = country_id_to_name.get(country_id, str(country_id))
                community_data.append({
                    "Country ID": country_id,
                    "Country Name": name,
                    "Community": int(cluster) + 1  # Make 1-indexed for display
                })
            
            community_df = pd.DataFrame(community_data)
            
            # Display results
            st.subheader("Detected Trade Communities")
            st.dataframe(community_df)
            
            # Visualize communities
            if embeddings is not None:
                st.subheader("Community Visualization")
                
                # Prepare data for visualization
                vis_data = pd.DataFrame({
                    "x": TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings.numpy())[:, 0],
                    "y": TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings.numpy())[:, 1],
                    "Country": [community_df.iloc[i]["Country Name"] for i in range(len(community_df))],
                    "Community": [f"Community {c}" for c in community_df["Community"]]
                })
                
                fig = px.scatter(vis_data, x="x", y="y", color="Community", hover_data=["Country"],
                                title="Trade Communities Visualization")
                st.plotly_chart(fig)
                
                # Community statistics
                st.subheader("Community Statistics")
                community_counts = community_df["Community"].value_counts().sort_index()
                fig = px.bar(x=community_counts.index, y=community_counts.values,
                            labels={"x": "Community", "y": "Number of Countries"},
                            title="Countries per Community")
                st.plotly_chart(fig)
    
    elif app_mode == "🦠 COVID-19 Impact":
        st.header("🦠 COVID-19 Impact Analysis")
        
        # Check if data spans COVID years
        years = sorted(df['year'].unique())
        if 2019 not in years or 2020 not in years:
            st.warning("Dataset does not contain both 2019 (pre-COVID) and 2020 (COVID) data. "
                      "Using the first two available years for demonstration.")
            if len(years) >= 2:
                pre_covid_year = years[0]
                covid_year = years[1]
            else:
                st.error("Not enough years in the dataset for comparison. Please upload data with at least two years.")
                return
        else:
            pre_covid_year = 2019
            covid_year = 2020
        
        # Perform analysis
        with st.spinner("Analyzing COVID-19 impact..."):
            covid_impact = analyze_covid_impact(df, pre_covid_year, covid_year)
            
            # Display overall impact
            st.subheader("Overall Trade Impact")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pre-COVID Trade Volume", f"${covid_impact['pre_covid_volume']:,.2f}")
            with col2:
                st.metric("COVID Trade Volume", f"${covid_impact['covid_volume']:,.2f}")
            with col3:
                st.metric("Percentage Change", f"{covid_impact['overall_change']:.2f}%",
                         delta=f"{covid_impact['overall_change']:.2f}%")
            
            # Export impact
            st.subheader("Export Impact by Country")
            
            # Add country names if available
            export_impact = covid_impact['export_change']
            if country_id_to_name:
                export_impact['Country Name'] = export_impact['country_id'].apply(
                    lambda x: country_id_to_name.get(x, str(x)))
            else:
                export_impact['Country Name'] = export_impact['country_id']
            
            # Sort by absolute percentage change
            export_impact = export_impact.sort_values('pct_change', key=abs, ascending=False)
            
            # Display top impacted exporters
            top_exporters = export_impact.head(10)
            fig = px.bar(top_exporters, x='Country Name', y='pct_change',
                        title="Top 10 Countries by Export Impact",
                        labels={'pct_change': 'Percentage Change', 'Country Name': 'Country'})
            st.plotly_chart(fig)
            
            # Import impact
            st.subheader("Import Impact by Country")
            
            # Add country names if available
            import_impact = covid_impact['import_change']
            if country_id_to_name:
                import_impact['Country Name'] = import_impact['country_id'].apply(
                    lambda x: country_id_to_name.get(x, str(x)))
            else:
                import_impact['Country Name'] = import_impact['country_id']
            
            # Sort by absolute percentage change
            import_impact = import_impact.sort_values('pct_change', key=abs, ascending=False)
            
            # Display top impacted importers
            top_importers = import_impact.head(10)
            fig = px.bar(top_importers, x='Country Name', y='pct_change',
                        title="Top 10 Countries by Import Impact",
                        labels={'pct_change': 'Percentage Change', 'Country Name': 'Country'})
            st.plotly_chart(fig)
    
    elif app_mode == "🌟 Trade Hub Analysis":
        st.header("🌟 Trade Hub Analysis")
        
        available_years = sorted(df['year'].unique())
        if not available_years:
            st.error("No year data available in the dataset.")
            return
        
        year = st.selectbox("Select year to analyze:", available_years)
        
        # Calculate centrality metrics
        with st.spinner("Calculating trade centrality metrics..."):
            centrality_df = calculate_centrality_metrics(df, year)
            
            # Add country names if available
            if country_id_to_name:
                centrality_df['Country Name'] = centrality_df['Country'].apply(
                    lambda x: country_id_to_name.get(x, str(x)))
            else:
                centrality_df['Country Name'] = centrality_df['Country']
            
            # Display centrality rankings
            st.subheader("Trade Hub Rankings")
            
            metric_type = st.radio("Select centrality metric:", 
                                ["Degree (Overall Connectivity)", 
                                 "In-Degree (Import Connectivity)",
                                 "Out-Degree (Export Connectivity)",
                                 "Betweenness (Bridge Role)",
                                 "Eigenvector (Connected to Important Hubs)"])
            
            metric_map = {
                "Degree (Overall Connectivity)": "Degree",
                "In-Degree (Import Connectivity)": "In-Degree",
                "Out-Degree (Export Connectivity)": "Out-Degree",
                "Betweenness (Bridge Role)": "Betweenness",
                "Eigenvector (Connected to Important Hubs)": "Eigenvector"
            }
            
            selected_metric = metric_map[metric_type]
            
            # Sort and display top countries
            sorted_df = centrality_df.sort_values(selected_metric, ascending=False).head(20)
            fig = px.bar(sorted_df, x='Country Name', y=selected_metric,
                        title=f"Top 20 Countries by {selected_metric} Centrality",
                        labels={selected_metric: 'Centrality Score', 'Country Name': 'Country'})
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
            
            # Display full rankings
            st.subheader("Complete Centrality Rankings")
            st.dataframe(centrality_df.sort_values(selected_metric, ascending=False))
            
            # Trade hub map visualization
            st.subheader("Global Trade Hub Map")
            st.info("This section would show a geographical map visualization of trade hubs based on centrality. "
                  "For a full implementation, you would need to integrate with a geo-mapping library.")
            
    elif app_mode == "🌎📉 Country Stats":
        st.header("Trade Statistics")
        country = st.text_input("Enter Country Code (e.g., USA, IND, CHN):", "IND")
        year = st.number_input("Enter Year:", min_value=2000, max_value=2024, value=2018, step=1)
        fetch_data = st.button("Fetch Trade Data")

        if fetch_data:
            summary = fetch_trade_summary(country, year)
            
            # Generate a clearer prompt for the AI to summarize and structure the country trade data as bullet points
            prompt = f"""
                Summarize the trade statistics for 
                {country} in {year} in a list of points. 
                Include key information like export/import values, 
                top trading partners, top export/import products, 
                product shares, and any other relevant data. 
                Here is the trade summary:\n\n{summary}
                \n\n 
                I will be displaying this information in a user-friendly streamlit app.
                Please ensure the summary is clear and concise.
                Do not include any unnecessary details or jargon.
                Do not do any formatting, just provide the text as points.
                Do not include any HTML or markdown formatting.

            """

            try:
                response = model_gen.generate_content(prompt)
                st.write("### Summary:")
                st.write(response.text)
                print("Generated response:", response.text)  # Debugging line to check the generated response
            except Exception as e:
                st.error(f"Error generating response: {e}")

    # News Scraping Section
    elif app_mode == "📰 Trade News":
        st.header("Latest Trade News")
        if st.button("Scrape Latest News"):
            scrape_news()  # Ensure you implement scrape_news function
            st.success("News scraped and summaries generated successfully!")

        news_items = load_news()
        if news_items:
            for news in news_items:
                st.markdown(f"### {news[1]}")
                st.write(news[2])
                st.markdown(f"[Read more]({news[3]})")
                st.write("---")
        else:
            st.write("No news available.")

        # Trade News Q&A
        st.header("Ask about Trade News")
        question = st.text_input("Enter your question about recent trade news:")
        if question:
            context = " ".join([news[2] for news in news_items])
            # Update the prompt to better guide the AI for answering trade-related questions
            prompt = f"Answer based on the following trade news:\n{context}\n\nQuestion: {question}\nPlease structure your answer in clear bullet points if possible."
            
            try:
                response = model_gen.generate_content(prompt)
                st.write(response.text)
            except Exception as e:
                st.error(f"Error generating response: {e}")


# Run the app
if __name__ == "__main__":
    main()