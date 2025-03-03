"""
Retrieval-Augmented Generation (RAG) agent for enhancing trade predictions
with contextual reasoning and domain knowledge
"""

from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline  # Updated import
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import pandas as pd


class RAGAgent:
    """
    Retrieval-Augmented Generation Agent for enhancing trade predictions with
    contextual reasoning and domain knowledge.
    """
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Initialize the RAG agent.
        
        Args:
            model_name: Hugging Face model name
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.sentence_encoder = None
        self.llm_chain = None
        self.knowledge_base = None
        self.country_info = {}
        self.product_info = {}
        
    def initialize(self):
        """Initialize models and knowledge base."""
        try:
            # Initialize language model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Initialize sentence transformer for encoding
            self.sentence_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512
            )
            
            # Create LLM wrapper
            llm = HuggingFacePipeline(pipeline=pipe)
            
            # Create prompt template
            template = """
            You are a trade policy advisor. Given the following information about two countries and 
            their trade patterns, evaluate the potential for establishing or expanding trade relationship.
            
            Country 1 (Exporter): {country1_info}
            
            Country 2 (Importer): {country2_info}
            
            Product Categories to Consider: {product_categories}
            
            Existing Trade Relationship: {existing_relationship}
            
            Statistical Prediction Score: {prediction_score}
            
            Based on this information, analyze:
            1. What are the key factors that make this a good potential trade relationship?
            2. What might be potential barriers or considerations?
            3. Provide 3-5 specific product categories that would be most promising.
            4. Rate this potential trade relationship on a scale of 1-10 and explain why.
            
            Your analysis:
            """
            
            prompt = PromptTemplate(
                input_variables=["country1_info", "country2_info", "product_categories",
                                "existing_relationship", "prediction_score"],
                template=template
            )
            
            # Create chain
            self.llm_chain = LLMChain(llm=llm, prompt=prompt)
        
        except Exception as e:
            print(f"Error initializing RAG agent: {e}")
            # Create a fallback dummy chain that just returns a message
            self.llm_chain = DummyChain()
        
        return self
    
    def build_knowledge_base(self, country_metadata, trade_data, product_categories):
        """
        Build knowledge base with country and product information.
        
        Args:
            country_metadata: DataFrame with country information
            trade_data: DataFrame with trade data
            product_categories: DataFrame with product categories
        """
        # Country information
        for _, row in country_metadata.iterrows():
            country_id = row['country_id']
            self.country_info[country_id] = {
                'id': country_id,
                'gdp': row.get('gdp', 'Unknown'),
                'population': row.get('population', 'Unknown'),
                'gdp_per_capita': row.get('gdp_per_capita', 'Unknown'),
                'top_exports': [],
                'top_imports': []
            }
        
        # Add top exports and imports for each country
        for country_id in self.country_info:
            # Top exports
            exports = trade_data[trade_data['exporter_id'] == country_id]
            if not exports.empty:
                top_exports = exports.groupby('hs2')['value'].sum().nlargest(5).reset_index()
                for _, row in top_exports.iterrows():
                    matches = product_categories[product_categories['hs2'] == row['hs2']]
                    if not matches.empty:
                        product = matches['product_name'].iloc[0]
                        self.country_info[country_id]['top_exports'].append({
                            'hs2': row['hs2'],
                            'product': product,
                            'value': row['value']
                        })
            
            # Top imports
            imports = trade_data[trade_data['importer_id'] == country_id]
            if not imports.empty:
                top_imports = imports.groupby('hs2')['value'].sum().nlargest(5).reset_index()
                for _, row in top_imports.iterrows():
                    matches = product_categories[product_categories['hs2'] == row['hs2']]
                    if not matches.empty:
                        product = matches['product_name'].iloc[0]
                        self.country_info[country_id]['top_imports'].append({
                            'hs2': row['hs2'],
                            'product': product,
                            'value': row['value']
                        })
        
        # Product information
        for _, row in product_categories.iterrows():
            hs2 = row['hs2']
            self.product_info[hs2] = {
                'hs2': hs2,
                'name': row['product_name'],
                'top_exporters': [],
                'top_importers': []
            }
        
        # Add top exporters and importers for each product category
        for hs2 in self.product_info:
            # Top exporters
            product_exports = trade_data[trade_data['hs2'] == hs2]
            if not product_exports.empty:
                top_exporters = product_exports.groupby('exporter_id')['value'].sum().nlargest(5).reset_index()
                for _, row in top_exporters.iterrows():
                    self.product_info[hs2]['top_exporters'].append({
                        'country_id': row['exporter_id'],
                        'value': row['value']
                    })
            
            # Top importers
            product_imports = trade_data[trade_data['hs2'] == hs2]
            if not product_imports.empty:
                top_importers = product_imports.groupby('importer_id')['value'].sum().nlargest(5).reset_index()
                for _, row in top_importers.iterrows():
                    self.product_info[hs2]['top_importers'].append({
                        'country_id': row['importer_id'],
                        'value': row['value']
                    })
        
        return self
    
    def get_country_info_text(self, country_id):
        """
        Generate textual description of a country for the LLM.
        
        Args:
            country_id: Country identifier
            
        Returns:
            text: Textual description
        """
        if country_id not in self.country_info:
            return f"Country ID: {country_id}\nNo detailed information available."
        
        info = self.country_info[country_id]
        
        text = f"Country ID: {info['id']}\n"
        text += f"GDP: {info['gdp']}\n"
        text += f"Population: {info['population']}\n"
        text += f"GDP per capita: {info['gdp_per_capita']}\n\n"
        
        # Add top exports
        text += "Top Exports:\n"
        for i, export in enumerate(info['top_exports'], 1):
            text += f"{i}. {export['product']} (HS: {export['hs2']}) - Value: {export['value']}\n"
        
        # Add top imports
        text += "\nTop Imports:\n"
        for i, imp in enumerate(info['top_imports'], 1):
            text += f"{i}. {imp['product']} (HS: {imp['hs2']}) - Value: {imp['value']}\n"
        
        return text
    
    def get_product_recommendations(self, exporter_id, importer_id):
        """
        Generate product recommendations for a potential trade relationship.
        
        Args:
            exporter_id: Exporter country ID
            importer_id: Importer country ID
            
        Returns:
            products: List of recommended product categories
        """
        # Products that exporter exports a lot
        exporter_products = set()
        if exporter_id in self.country_info:
            for export in self.country_info[exporter_id]['top_exports']:
                exporter_products.add(export['hs2'])
        
        # Products that importer imports a lot
        importer_products = set()
        if importer_id in self.country_info:
            for imp in self.country_info[importer_id]['top_imports']:
                importer_products.add(imp['hs2'])
        
        # Find intersection (products that exporter exports and importer imports)
        common_products = exporter_products.intersection(importer_products)
        
        # If there are common products, return those first, then add others
        recommended_products = list(common_products)
        
        # Add other top exports from exporter
        for hs2 in exporter_products - common_products:
            recommended_products.append(hs2)
            if len(recommended_products) >= 5:
                break
        
        # If still fewer than 5, add top imports from importer
        if len(recommended_products) < 5:
            for hs2 in importer_products - common_products - set(recommended_products):
                recommended_products.append(hs2)
                if len(recommended_products) >= 5:
                    break
        
        # Convert to actual product names
        results = []
        for hs2 in recommended_products[:5]:  # Top 5 recommendations
            if hs2 in self.product_info:
                results.append(self.product_info[hs2]['name'])
            else:
                results.append(f"Product category HS {hs2}")
        
        return results
    
    def analyze_trade_relationship(
        self, 
        exporter_id, 
        importer_id, 
        prediction_score, 
        existing_relationship=None
    ):
        """
        Analyze potential trade relationship using RAG approach.
        
        Args:
            exporter_id: Exporter country ID
            importer_id: Importer country ID
            prediction_score: Model's prediction score
            existing_relationship: Description of existing trade (if any)
            
        Returns:
            analysis: Textual analysis from the LLM
        """
        try:
            # Get country information
            country1_info = self.get_country_info_text(exporter_id)
            country2_info = self.get_country_info_text(importer_id)
            
            # Get recommended products
            products = self.get_product_recommendations(exporter_id, importer_id)
            product_text = ", ".join(products)
            
            # Describe existing relationship
            if existing_relationship is None:
                existing_text = "No significant existing trade relationship identified."
            else:
                existing_text = existing_relationship
            
            # Run the LLM chain
            result = self.llm_chain.run({
                "country1_info": country1_info,
                "country2_info": country2_info,
                "product_categories": product_text,
                "existing_relationship": existing_text,
                "prediction_score": f"{prediction_score:.4f}"
            })
            
            return result
        except Exception as e:
            # Fallback response if the LLM fails
            return f"""
            Analysis of trade relationship between exporter ID: {exporter_id} and importer ID: {importer_id}
            
            This appears to be a promising trade relationship with a prediction score of {prediction_score:.4f}.
            
            Key factors:
            1. Economic complementarity between the countries
            2. Potential for market expansion
            3. Geographic and logistical feasibility
            
            Recommended product categories:
            - {', '.join(products)}
            
            Overall rating: {int(prediction_score*10)}/10
            
            Note: This is a simplified analysis. For a more detailed assessment, please ensure the LLM is properly configured.
            """


# Add this class to provide a fallback when LLM initialization fails
class DummyChain:
    """A dummy chain that returns a template response when LLM initialization fails"""
    
    def run(self, inputs):
        """Return a template response"""
        country1_info = inputs.get("country1_info", "")
        country2_info = inputs.get("country2_info", "")
        product_categories = inputs.get("product_categories", "")
        prediction_score = inputs.get("prediction_score", "0.5")
        
        # Extract country IDs from the info strings
        try:
            country1_id = country1_info.split("\n")[0].split(": ")[1]
            country2_id = country2_info.split("\n")[0].split(": ")[1]
        except:
            country1_id = "Unknown"
            country2_id = "Unknown"
        
        score = float(prediction_score) if isinstance(prediction_score, str) else prediction_score
        
        return f"""
        Analysis of trade relationship between exporter ID: {country1_id} and importer ID: {country2_id}
        
        This appears to be a promising trade relationship with a prediction score of {score:.4f}.
        
        Key factors:
        1. Economic complementarity between the countries
        2. Potential for market expansion
        3. Geographic and logistical feasibility
        
        Recommended product categories:
        - {product_categories}
        
        Overall rating: {int(score*10)}/10
        
        Note: This is a simplified analysis because the LLM could not be initialized.
        Please ensure all required packages are installed for a more detailed assessment.
        """