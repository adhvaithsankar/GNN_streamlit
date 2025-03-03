"""
Visualization tools for trade network and opportunities
"""

import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import numpy as np


class TradeOpportunityVisualizer:
    """
    Class to create interactive visualizations for trade opportunities.
    """
    
    def __init__(self, analyzer):
        """
        Initialize the visualizer.
        
        Args:
            analyzer: TradeAnalyzer instance
        """
        self.analyzer = analyzer
        self.country_names = {}  # Map from country ID to name
        
        # Extract country names from trade data
        for _, row in analyzer.data_processor.trade_data.iterrows():
            self.country_names[row['exporter_id']] = row['exporter_name']
            self.country_names[row['importer_id']] = row['importer_name']
    
    def plot_trade_network(self, country_id=None, highlight_opportunities=None):
        """
        Plot interactive trade network visualization.
        
        Args:
            country_id: Optional country ID to focus on
            highlight_opportunities: Optional opportunities to highlight
            
        Returns:
            fig: Plotly figure
        """
        G = self.analyzer.data_processor.G
        node_mapping = self.analyzer.data_processor.pyg_data.node_mapping
        
        # Create positions for the nodes using a layout algorithm
        pos = nx.spring_layout(G, seed=42)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node text
            name = self.country_names.get(node, f"Country {node}")
            node_text.append(name)
            
            # Node size based on degree
            size = 10 + G.degree(node)
            node_size.append(size)
            
            # Node color - highlight the selected country if any
            if country_id is not None and node == country_id:
                node_color.append('red')
            else:
                node_color.append('skyblue')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line=dict(width=2, color='white')
            )
        )
        
        # Create edge traces
        edge_traces = []
        
        # Regular edges
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge text
            exporter = self.country_names.get(edge[0], f"Country {edge[0]}")
            importer = self.country_names.get(edge[1], f"Country {edge[1]}")
            volume = G[edge[0]][edge[1]]['trade_volume']
            edge_text.extend([f"{exporter} → {importer}: ${volume:,.0f}", "", ""])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines'
        )
        
        edge_traces.append(edge_trace)
        
        # Highlight opportunity edges if provided
        if highlight_opportunities is not None:
            opp_edge_x = []
            opp_edge_y = []
            opp_edge_text = []
            
            for opp in highlight_opportunities:
                exporter_id = opp['exporter_id']
                importer_id = opp['importer_id']
                
                # Check if nodes exist in the graph
                if exporter_id in pos and importer_id in pos:
                    x0, y0 = pos[exporter_id]
                    x1, y1 = pos[importer_id]
                    opp_edge_x.extend([x0, x1, None])
                    opp_edge_y.extend([y0, y1, None])
                    
                    # Edge text
                    exporter = self.country_names.get(exporter_id, f"Country {exporter_id}")
                    importer = self.country_names.get(importer_id, f"Country {importer_id}")
                    prob = opp['probability']
                    vol = opp.get('potential_volume', 0)
                    opp_edge_text.extend([f"Opportunity: {exporter} → {importer}, Prob: {prob:.2f}, Est. Vol: ${vol:,.0f}", "", ""])
            
            opp_trace = go.Scatter(
                x=opp_edge_x, y=opp_edge_y,
                line=dict(width=2, color='green'),
                hoverinfo='text',
                text=opp_edge_text,
                mode='lines'
            )
            
            edge_traces.append(opp_trace)
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title={'text': 'International Trade Network', 'font': {'size': 16}},
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(
                    text="Trade network visualization",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig
    
    def plot_opportunity_rankings(self, opportunities):
        """
        Plot ranked opportunities as a bar chart.
        
        Args:
            opportunities: List of opportunity dictionaries
            
        Returns:
            fig: Plotly figure
        """
        # Extract data
        countries = []
        probabilities = []
        volumes = []
        colors = []
        
        for opp in opportunities:
            if opp['role'] == 'exporter':
                # We are exporting to this country
                country_id = opp['importer_id']
                direction = "Export to"
            else:
                # We are importing from this country
                country_id = opp['exporter_id']
                direction = "Import from"
            
            country_name = self.country_names.get(country_id, f"Country {country_id}")
            countries.append(f"{direction} {country_name}")
            probabilities.append(opp['probability'])
            volumes.append(opp.get('potential_volume', 0))
            
            # Color based on probability
            if opp['probability'] > 0.8:
                colors.append('green')
            elif opp['probability'] > 0.6:
                colors.append('yellowgreen')
            else:
                colors.append('orange')
        
        # Sort by probability
        indices = np.argsort(probabilities)[::-1]
        countries = [countries[i] for i in indices]
        probabilities = [probabilities[i] for i in indices]
        volumes = [volumes[i] for i in indices]
        colors = [colors[i] for i in indices]
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=countries,
            y=probabilities,
            name='Probability',
            marker_color=colors,
            text=[f"{p:.2f}" for p in probabilities],
            textposition='auto'
        ))
        
        # Add second y-axis for volume
        fig.add_trace(go.Scatter(
            x=countries,
            y=volumes,
            name='Est. Volume',
            mode='markers',
            marker=dict(size=10, color='black'),
            yaxis='y2'
        ))
        
        # Update layout
        fig.update_layout(
            title='Trade Opportunities Ranked by Probability',
            yaxis=dict(
                title='Probability',
                range=[0, 1]
            ),
            yaxis2=dict(
                title='Estimated Volume',
                overlaying='y',
                side='right'
            ),
            barmode='group',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def plot_underutilized_opportunities(self, opportunities):
        """
        Plot underutilized trade opportunities.
        
        Args:
            opportunities: List of underutilized opportunity dictionaries
            
        Returns:
            fig: Plotly figure
        """
        # Extract data
        countries = []
        actual_volumes = []
        potential_volumes = []
        utilization_ratios = []
        
        for opp in opportunities:
            exporter_id = opp['exporter_id']
            importer_id = opp['importer_id']
            
            exporter_name = self.country_names.get(exporter_id, f"Country {exporter_id}")
            importer_name = self.country_names.get(importer_id, f"Country {importer_id}")
            
            countries.append(f"{exporter_name} → {importer_name}")
            actual_volumes.append(opp['actual_volume'])
            potential_volumes.append(opp['potential_volume'])
            utilization_ratios.append(opp['utilization_ratio'])
        
        # Sort by growth potential (potential - actual)
        growth_potential = [p - a for p, a in zip(potential_volumes, actual_volumes)]
        indices = np.argsort(growth_potential)[::-1]
        
        countries = [countries[i] for i in indices]
        actual_volumes = [actual_volumes[i] for i in indices]
        potential_volumes = [potential_volumes[i] for i in indices]
        utilization_ratios = [utilization_ratios[i] for i in indices]
        
        # Limit to top 20 for visibility
        countries = countries[:20]
        actual_volumes = actual_volumes[:20]
        potential_volumes = potential_volumes[:20]
        utilization_ratios = utilization_ratios[:20]
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for actual volume
        fig.add_trace(go.Bar(
            x=countries,
            y=actual_volumes,
            name='Current Volume',
            marker_color='blue',
            text=[f"${v:,.0f}" for v in actual_volumes],
            textposition='auto'
        ))
        
        # Add bars for potential volume
        fig.add_trace(go.Bar(
            x=countries,
            y=potential_volumes,
            name='Potential Volume',
            marker_color='green',
            text=[f"${v:,.0f}" for v in potential_volumes],
            textposition='auto'
        ))
        
        # Add line for utilization ratio
        fig.add_trace(go.Scatter(
            x=countries,
            y=utilization_ratios,
            name='Utilization Ratio',
            mode='markers+lines',
            marker=dict(size=8, color='red'),
            yaxis='y2'
        ))
        
        # Update layout
        fig.update_layout(
            title='Underutilized Trade Opportunities',
            xaxis_tickangle=-45,
            yaxis=dict(
                title='Trade Volume'
            ),
            yaxis2=dict(
                title='Utilization Ratio',
                overlaying='y',
                side='right',
                range=[0, max(utilization_ratios) * 1.1]
            ),
            barmode='group'
        )
        
        return fig