#!/usr/bin/env python3
"""
Simplified Enhanced AyurvaBot Runner
This script runs a simplified version of the enhanced AyurvaBot to avoid segmentation faults.
"""

import os
import sys
import gradio as gr
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class SimplifiedEnhancedAyurvaBot:
    """
    Simplified version of the Enhanced AyurvaBot.
    """
    
    def __init__(self, index_path=None, documents_path=None):
        """Initialize the bot with simpler components."""
        # Set up paths
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        self.index_path = index_path or os.path.join(self.output_dir, "ayurveda_index.faiss")
        self.documents_path = documents_path or os.path.join(self.output_dir, "ayurveda_documents.csv")
        
        # Initialize model
        print("Loading embedding model...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize QA pipeline
        print("Loading QA pipeline...")
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        
        # Try to load index or create fallback
        self.load_or_create_index()
        
        # Initialize documents
        self.documents = []
        if os.path.exists(self.documents_path):
            import pandas as pd
            try:
                self.documents_df = pd.read_csv(self.documents_path)
                print(f"Loaded {len(self.documents_df)} documents")
                self.documents = self.documents_df['content'].tolist()
            except Exception as e:
                print(f"Error loading documents: {e}")
                self.create_fallback_documents()
        else:
            print("Documents file not found, creating fallback")
            self.create_fallback_documents()
    
    
    
    
   
        # Get embeddings
        embeddings = self.model.encode(docs, convert_to_numpy=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        # Save fallback index
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        print("Fallback index created and saved")
        
        # Save documents
        import pandas as pd
        df = pd.DataFrame({
            'content': docs,
            'source': ['fallback'] * len(docs)
        })
        df.to_csv(self.documents_path, index=False)
        print("Fallback documents saved")
        
        self.index_loaded = True
        self.documents = docs
        self.documents_df = df
    
    def search_documents(self, query, k=3):
        """Search for documents related to the query."""
        if not hasattr(self, 'index') or not self.index:
            print("Index not loaded")
            return []
        
        # Get query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Normalize embedding
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.documents):
                content = self.documents[idx]
                source = self.documents_df.iloc[idx]['source'] if 'source' in self.documents_df.columns else 'unknown'
                score = float(distances[0][i])
                results.append((content, score, {'source': source}))
        
        return results
    
    def generate_response(self, query):
        """Generate a response to the user's query."""
        # Search for relevant documents
        docs = self.search_documents(query, k=3)
        
        if not docs:
            return "I couldn't find information about that in my Ayurvedic knowledge base. Could you ask about doshas, herbs, or treatments?"
        
        # Combine documents into context
        context = "\n\n".join([doc[0] for doc in docs])
        
        try:
            # Use QA pipeline to get answer
            result = self.qa_pipeline(question=query, context=context)
            answer = result['answer']
            
            # Format response
            response = f"Based on Ayurvedic principles:\n\n{answer}\n\n"
            response += "This information comes from traditional Ayurvedic knowledge. For health concerns, please consult a qualified practitioner."
            
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return context + "\n\n(I've provided the most relevant information I could find on this topic.)"

class AyurvaApp:
    """Application wrapper with feedback collection."""
    
    def __init__(self):
        """Initialize the application."""
        self.bot = SimplifiedEnhancedAyurvaBot()
        self.feedback_collector = FeedbackCollector()
        self.last_query = None
        self.last_response = None
    
    def answer_query(self, query):
        """Process a user query and return the response."""
        self.last_query = query
        self.last_response = self.bot.generate_response(query)
        return self.last_response
    
    def submit_feedback(self, feedback_text, rating):
        """Submit feedback on the last response."""
        if not self.last_query or not self.last_response:
            return "No recent query to provide feedback on."
        
        return self.feedback_collector.save_feedback(
            question=self.last_query,
            answer=self.last_response,
            feedback=feedback_text,
            rating=rating
        )

def chat_interface(message, history):
    """Gradio chat interface function."""
    app = getattr(chat_interface, "app", None)
    if app is None:
        app = AyurvaApp()
        chat_interface.app = app
    
    response = app.answer_query(message)
    return response

def submit_feedback(feedback_text, rating):
    """Submit feedback function for Gradio."""
    app = getattr(chat_interface, "app", None)
    if app is None:
        return "Error: Application not initialized"
    
    return app.submit_feedback(feedback_text, rating)

def main():
    """Main function to run the Gradio interface."""
    print("Starting Simplified Enhanced AyurvaBot...")
    
    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("# 🌿 Enhanced AyurvaBot - Ayurvedic Knowledge Assistant")
        gr.Markdown("Ask me questions about Ayurveda, herbs, treatments, and wellness practices.")
        
        # Chat interface
        chatbot = gr.ChatInterface(
            chat_interface,
            examples=[
                "What is Vata dosha?",
                "How do I balance Pitta dosha?",
                "What herbs are good for digestion?",
                "Tell me about Panchakarma treatment",
                "What's the Ayurvedic approach to treating anxiety?",
            ]
        )
        
        # Feedback section
        with gr.Accordion("Provide Feedback", open=False):
            gr.Markdown("### Help improve AyurvaBot by providing feedback on the last answer")
            
            with gr.Row():
                feedback_text = gr.Textbox(
                    label="Your Feedback", 
                    placeholder="What did you like or dislike about the answer?"
                )
                
                rating = gr.Slider(
                    minimum=1, 
                    maximum=5, 
                    step=1, 
                    label="Rating (1-5)", 
                    value=3
                )
            
            feedback_btn = gr.Button("Submit Feedback")
            feedback_result = gr.Textbox(label="Feedback Status")
            
            feedback_btn.click(
                submit_feedback, 
                inputs=[feedback_text, rating], 
                outputs=feedback_result
            )
    
    # Launch the interface
    demo.launch(share=True)

if __name__ == "__main__":
    main()