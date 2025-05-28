"""
RAG (Retrieval-Augmented Generation) System for Enhanced AyurvaBot
Handles vector search, embeddings, and AI model integration.
"""

import os
import numpy as np
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("⚠️ Some dependencies not available. Running in fallback mode.")

from knowledge_base import AyurvedaKnowledgeBase

class RAGSystem:
    """
    RAG system for intelligent document retrieval and response generation.
    """
    
    def __init__(self, config):
        self.config = config
        self.sentence_transformer = None
        self.hf_pipeline = None
        self.faiss_index = None
        self.knowledge_base = AyurvedaKnowledgeBase()
        self.document_chunks = []
        
        # Set Hugging Face token
        if DEPENDENCIES_AVAILABLE:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = config.HF_TOKEN
    
    def load_knowledge_base(self):
        """
        Load the Ayurvedic knowledge base.
        """
        try:
            chunk_count = self.knowledge_base.load_knowledge_chunks()
            self.document_chunks = self.knowledge_base.get_document_chunks()
            print(f"✅ Loaded {chunk_count} knowledge chunks")
            return True
        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")
            return False
    
    def initialize_sentence_transformer(self):
        """
        Initialize the sentence transformer for embeddings.
        """
        if not DEPENDENCIES_AVAILABLE:
            print("⚠️ Sentence Transformer not available. Using fallback mode.")
            return False
            
        try:
            print("Loading Sentence Transformer for RAG embeddings...")
            self.sentence_transformer = SentenceTransformer(self.config.EMBEDDING_MODEL)
            print("✅ Sentence Transformer loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading Sentence Transformer: {e}")
            return False
    
    def initialize_faiss_index(self):
        """
        Initialize FAISS index for vector search.
        """
        if not DEPENDENCIES_AVAILABLE or not self.sentence_transformer or not self.knowledge_base.get_knowledge_chunks():
            print("⚠️ Cannot initialize FAISS: missing dependencies or data")
            return False
        
        try:
            print("Creating FAISS index for RAG system...")
            
            # Generate embeddings for all knowledge chunks
            knowledge_chunks = self.knowledge_base.get_knowledge_chunks()
            print(f"Generating embeddings for {len(knowledge_chunks)} knowledge chunks...")
            embeddings = self.sentence_transformer.encode(knowledge_chunks)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            self.faiss_index.add(embeddings.astype('float32'))
            
            print(f"✅ FAISS index created with {self.faiss_index.ntotal} vectors!")
            return True
            
        except Exception as e:
            print(f"❌ Error creating FAISS index: {e}")
            return False
    
    def initialize_hf_model(self):
        """
        Initialize Hugging Face QA model.
        """
        if not DEPENDENCIES_AVAILABLE:
            print("⚠️ Hugging Face models not available. Using fallback mode.")
            return False
            
        try:
            print("Loading Hugging Face QA model...")
            self.hf_pipeline = pipeline(
                "question-answering", 
                model=self.config.QA_MODEL,
                token=self.config.HF_TOKEN
            )
            print("✅ Hugging Face QA model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading Hugging Face model: {e}")
            return False
    
    def retrieve(self, query, top_k=None):
        """
        Retrieve relevant documents using vector similarity search.
        """
        if top_k is None:
            top_k = self.config.TOP_K_RETRIEVAL
            
        if not self.faiss_index or not self.sentence_transformer:
            # Fallback: simple keyword matching
            return self._fallback_retrieve(query, top_k)
        
        try:
            # Generate embedding for the query
            query_embedding = self.sentence_transformer.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
            
            # Get the relevant document chunks
            relevant_chunks = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.document_chunks) and score > self.config.SIMILARITY_THRESHOLD:
                    chunk = self.document_chunks[idx].copy()
                    chunk['similarity_score'] = float(score)
                    chunk['rank'] = i + 1
                    relevant_chunks.append(chunk)
            
            # Sort by similarity score (descending)
            relevant_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return relevant_chunks
            
        except Exception as e:
            print(f"❌ Error in RAG retrieval: {e}")
            return self._fallback_retrieve(query, top_k)
    
    def _fallback_retrieve(self, query, top_k):
        """
        Fallback retrieval using simple keyword matching.
        """
        query_lower = query.lower()
        relevant_chunks = []
        
        for chunk in self.document_chunks:
            text_lower = chunk['text'].lower()
            # Simple keyword matching
            score = sum(1 for word in query_lower.split() if word in text_lower)
            if score > 0:
                chunk_copy = chunk.copy()
                chunk_copy['similarity_score'] = score / len(query_lower.split())
                relevant_chunks.append(chunk_copy)
        
        # Sort by score and return top_k
        relevant_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
        return relevant_chunks[:top_k]
    
    def generate(self, query, retrieved_chunks):
        """
        Generate response using retrieved chunks and Hugging Face model.
        """
        if not self.hf_pipeline or not retrieved_chunks:
            return None, []
        
        try:
            # Combine retrieved chunks into context
            context_parts = []
            for chunk in retrieved_chunks[:3]:  # Use top 3 chunks
                context_parts.append(chunk['text'])
            
            combined_context = " ".join(context_parts)
            
            # Limit context length for the model
            if len(combined_context) > self.config.MAX_CONTEXT_LENGTH:
                combined_context = combined_context[:self.config.MAX_CONTEXT_LENGTH]
            
            # Use the pipeline for question answering
            result = self.hf_pipeline(
                question=query,
                context=combined_context
            )
            
            if result and 'answer' in result:
                answer = result['answer'].strip()
                confidence = result.get('score', 0)
                
                # Only return answers with reasonable confidence and length
                if answer and len(answer) > 5 and confidence > self.config.CONFIDENCE_THRESHOLD:
                    return answer, retrieved_chunks
            
            return None, retrieved_chunks
            
        except Exception as e:
            print(f"❌ Error in RAG generation: {e}")
            return None, retrieved_chunks
    
    def rag_pipeline(self, query):
        """
        Complete RAG pipeline: retrieve + generate.
        """
        try:
            # Step 1: Retrieve relevant documents
            retrieved_chunks = self.retrieve(query)
            
            if not retrieved_chunks:
                return None, []
            
            # Step 2: Generate response
            answer, chunks_used = self.generate(query, retrieved_chunks)
            
            return answer, chunks_used
            
        except Exception as e:
            print(f"❌ Error in RAG pipeline: {e}")
            return None, []
    
    def is_ready(self):
        """
        Check if RAG system is fully initialized and ready.
        """
        return len(self.document_chunks) > 0
