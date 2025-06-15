import os
import json
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel, pipeline
import streamlit as st
import pandas as pd
from typing import List, Dict, Tuple
import re
from datetime import datetime
import pickle
import torch
from huggingface_hub import login
from src.config import HUGGINGFACE_TOKEN

class EnhancedAyurvaBot:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 llm_model='microsoft/DialoGPT-medium', index_path='ayurveda_index.faiss'):
        """
        Initialize the Enhanced Ayurveda Chatbot with FAISS vector search and Hugging Face models
        
        Args:
            model_name: Sentence transformer model for embeddings
            llm_model: Language model for response generation
            index_path: Path to save/load FAISS index
        """
        # Use Hugging Face token from config
        self.hf_token = HUGGINGFACE_TOKEN
        
        # Authenticate with Hugging Face
        if self.hf_token:
            login(token=self.hf_token)
            os.environ['HUGGINGFACE_HUB_TOKEN'] = self.hf_token
            print("✅ Authenticated with Hugging Face")
        else:
            print("⚠️ Warning: No Hugging Face token found. Some models may not be accessible.")
        
        # Initialize models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Load embedding model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.hf_token)
            self.embedding_model = AutoModel.from_pretrained(model_name, token=self.hf_token)
            self.embedding_model.to(self.device)
            print(f"✅ Loaded embedding model: {model_name}")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise
        
        # Initialize text generation pipeline
        try:
            self.text_generator = pipeline(
                "text-generation",
                model=llm_model,
                tokenizer=llm_model,
                device=0 if torch.cuda.is_available() else -1,
                token=self.hf_token,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256
            )
            print(f"✅ Loaded text generation model: {llm_model}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load text generation model: {e}")
            self.text_generator = None
        
        self.index_path = index_path
        self.metadata_path = index_path.replace('.faiss', '_metadata.pkl')
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Enhanced Ayurveda knowledge base
        self.ayurveda_knowledge = {
            'doshas': {
                'vata': {
                    'characteristics': ['dry', 'light', 'cold', 'rough', 'subtle', 'mobile', 'clear'],
                    'imbalance_symptoms': ['anxiety', 'insomnia', 'constipation', 'dry skin', 'joint pain', 'irregular digestion', 'restlessness'],
                    'balancing_foods': ['warm cooked foods', 'ghee', 'nuts', 'dates', 'warm milk', 'cooked grains', 'root vegetables'],
                    'lifestyle': ['regular routine', 'oil massage (abhyanga)', 'meditation', 'gentle yoga', 'early bedtime', 'warm environment'],
                    'herbs': ['ashwagandha', 'brahmi', 'jatamansi', 'shankhpushpi', 'bala'],
                    'season': 'autumn and early winter',
                    'time_of_day': 'dawn and dusk (2-6 AM/PM)'
                },
                'pitta': {
                    'characteristics': ['hot', 'sharp', 'light', 'oily', 'liquid', 'mobile', 'penetrating'],
                    'imbalance_symptoms': ['anger', 'heartburn', 'skin rashes', 'excessive heat', 'inflammation', 'hyperacidity', 'irritability'],
                    'balancing_foods': ['cooling foods', 'sweet fruits', 'coconut water', 'leafy greens', 'cucumber', 'melons', 'dairy'],
                    'lifestyle': ['avoid excessive heat', 'cooling pranayama', 'moderate exercise', 'shade and cool environments'],
                    'herbs': ['amalaki', 'neem', 'guduchi', 'shatavari', 'brahmi'],
                    'season': 'summer and late spring',
                    'time_of_day': 'midday and midnight (10 AM-2 PM, 10 PM-2 AM)'
                },
                'kapha': {
                    'characteristics': ['heavy', 'slow', 'steady', 'solid', 'cold', 'soft', 'smooth'],
                    'imbalance_symptoms': ['lethargy', 'weight gain', 'congestion', 'depression', 'attachment', 'sluggish digestion'],
                    'balancing_foods': ['spicy foods', 'light meals', 'ginger', 'turmeric', 'honey', 'warm beverages'],
                    'lifestyle': ['vigorous exercise', 'dry brushing', 'wake up early', 'stimulating activities', 'warm dry environments'],
                    'herbs': ['trikatu', 'guggulu', 'punarnava', 'chitrak', 'pippali'],
                    'season': 'late winter and spring',
                    'time_of_day': 'morning and evening (6-10 AM/PM)'
                }
            },
            'herbs': {
                'ashwagandha': 'Powerful adaptogen for stress, anxiety, vitality, and immune support. Rasayana (rejuvenative) herb.',
                'turmeric': 'Anti-inflammatory, digestive, liver support, and blood purifier. Sacred golden herb.',
                'triphala': 'Three-fruit combination for digestion, detoxification, and rejuvenation. Gentle daily tonic.',
                'brahmi': 'Supreme brain tonic for memory, concentration, and nervous system health.',
                'neem': 'Bitter blood purifier for skin health, diabetes, and immune support.',
                'tulsi': 'Holy basil for respiratory health, stress relief, and spiritual upliftment.',
                'ginger': 'Universal medicine for digestion, circulation, nausea, and inflammation.',
                'cardamom': 'Aromatic digestive spice, breath freshener, and heart tonic.',
                'shatavari': 'Female reproductive tonic, also beneficial for digestive and immune health.',
                'guduchi': 'Immune modulator and fever reducer, excellent for autoimmune conditions.',
                'amalaki': 'Richest natural source of Vitamin C, anti-aging, and tissue regeneration.',
                'licorice': 'Harmonizing herb for respiratory health, adrenals, and inflammation.'
            },
            'practices': {
                'dinacharya': 'Daily routine aligned with natural circadian rhythms for optimal health',
                'ritucharya': 'Seasonal routines to maintain balance throughout the year',
                'pranayama': 'Breathing practices for prana (life force) regulation and mental clarity',
                'meditation': 'Mental practices for inner peace, clarity, and spiritual growth',
                'yoga_asanas': 'Physical postures for strength, flexibility, and energy flow',
                'abhyanga': 'Daily self-massage with warm oil for circulation and nervous system',
                'tongue_scraping': 'Morning oral hygiene practice for detoxification',
                'oil_pulling': 'Swishing oil in mouth for oral and systemic health',
                'nasya': 'Nasal administration of medicated oils for respiratory and mental clarity',
                'panchakarma': 'Five purification procedures for deep detoxification and rejuvenation'
            },
            'concepts': {
                'agni': 'Digestive fire responsible for metabolism and transformation',
                'ama': 'Undigested toxins that cause disease when accumulated',
                'ojas': 'Vital essence, immunity, and spiritual energy',
                'tejas': 'Metabolic fire, intelligence, and discrimination',
                'prana': 'Life force energy that governs all bodily functions',
                'srotas': 'Channels of circulation in the body',
                'dhatus': 'Seven tissue layers of the body',
                'malas': 'Waste products that need proper elimination'
            }
        }
        
        # Initialize or load existing index
        self.load_or_create_index()
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Hugging Face transformer model"""
        embeddings = []
        
        for text in texts:
            # Tokenize and encode
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                  padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                # Use mean pooling of last hidden states
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def create_comprehensive_documents(self) -> List[Dict]:
        """Create comprehensive documents from enhanced Ayurveda knowledge base"""
        documents = []
        
        # Create detailed documents for doshas
        for dosha, info in self.ayurveda_knowledge['doshas'].items():
            # Main dosha document
            content = f"""
            {dosha.title()} Dosha - Complete Guide:
            
            Characteristics: {', '.join(info['characteristics'])}
            
            Imbalance Symptoms: {', '.join(info['imbalance_symptoms'])}
            
            Balancing Foods: {', '.join(info['balancing_foods'])}
            
            Lifestyle Recommendations: {', '.join(info['lifestyle'])}
            
            Beneficial Herbs: {', '.join(info['herbs'])}
            
            Predominant Season: {info['season']}
            
            Active Time of Day: {info['time_of_day']}
            
            {dosha.title()} dosha governs movement, transformation, or structure in the body and mind.
            """
            
            doc = {
                'content': self.preprocess_text(content),
                'type': 'dosha',
                'category': dosha,
                'title': f"{dosha.capitalize()} Dosha - Complete Guide",
                'keywords': [dosha] + info['characteristics'] + info['imbalance_symptoms']
            }
            documents.append(doc)
        
        # Create detailed herb documents
        for herb, description in self.ayurveda_knowledge['herbs'].items():
            content = f"""
            {herb.title()} - Ayurvedic Herb:
            
            {description}
            
            This herb is traditionally used in Ayurvedic medicine for its therapeutic properties.
            Always consult with a qualified practitioner before use.
            """
            
            doc = {
                'content': self.preprocess_text(content),
                'type': 'herb',
                'category': herb,
                'title': f"{herb.replace('_', ' ').title()} - Ayurvedic Herb",
                'keywords': [herb, 'herb', 'medicine', 'treatment']
            }
            documents.append(doc)
        
        # Create practice documents
        for practice, description in self.ayurveda_knowledge['practices'].items():
            content = f"""
            {practice.replace('_', ' ').title()} - Ayurvedic Practice:
            
            {description}
            
            This practice is an important part of Ayurvedic lifestyle and wellness approach.
            Regular practice leads to better health and balance.
            """
            
            doc = {
                'content': self.preprocess_text(content),
                'type': 'practice',
                'category': practice,
                'title': f"{practice.replace('_', ' ').title()} - Ayurvedic Practice",
                'keywords': [practice, 'practice', 'routine', 'lifestyle']
            }
            documents.append(doc)
        
        # Create concept documents
        for concept, description in self.ayurveda_knowledge['concepts'].items():
            content = f"""
            {concept.title()} - Ayurvedic Concept:
            
            {description}
            
            Understanding {concept} is fundamental to Ayurvedic health and healing.
            """
            
            doc = {
                'content': self.preprocess_text(content),
                'type': 'concept',
                'category': concept,
                'title': f"{concept.title()} - Ayurvedic Concept",
                'keywords': [concept, 'concept', 'principle', 'theory']
            }
            documents.append(doc)
        
        # Additional comprehensive documents for common queries
        additional_docs = [
            {
                'content': """
                Introduction to Ayurveda:
                
                Ayurveda is a 5000-year-old system of natural healing from India. It focuses on 
                achieving balance between mind, body, and spirit through personalized diet, lifestyle, 
                herbal remedies, and spiritual practices. The word Ayurveda means 'knowledge of life' 
                in Sanskrit. It emphasizes prevention and treatment through lifestyle practices and 
                natural therapies.
                """,
                'type': 'general',
                'category': 'introduction',
                'title': "Complete Introduction to Ayurveda",
                'keywords': ['ayurveda', 'introduction', 'basics', 'overview']
            },
            {
                'content': """
                Ayurvedic Diet Principles:
                
                Ayurvedic nutrition is based on your constitution (Prakriti) and current imbalances (Vikriti).
                Eat according to your dosha, favor fresh and seasonal foods, eat mindfully in peaceful 
                environment, make lunch your largest meal, avoid incompatible food combinations, 
                and include all six tastes (sweet, sour, salty, pungent, bitter, astringent) in your meals.
                """,
                'type': 'diet',
                'category': 'nutrition',
                'title': "Ayurvedic Diet and Nutrition Principles",
                'keywords': ['diet', 'nutrition', 'food', 'eating', 'digestion']
            },
            {
                'content': """
                Ayurvedic Treatment Approach:
                
                Ayurvedic treatment is highly individualized based on constitution, current imbalances,
                season, age, and lifestyle. Treatment includes dietary modifications, herbal medicines,
                lifestyle counseling, detoxification procedures (Panchakarma), yoga, meditation,
                and spiritual practices. The goal is to restore natural balance and strengthen 
                the body's healing capacity.
                """,
                'type': 'treatment',
                'category': 'healing',
                'title': "Ayurvedic Treatment and Healing Approach",
                'keywords': ['treatment', 'healing', 'therapy', 'medicine', 'cure']
            }
        ]
        
        for doc in additional_docs:
            doc['content'] = self.preprocess_text(doc['content'])
        
        documents.extend(additional_docs)
        return documents
    
    def build_vector_index(self, documents: List[Dict]):
        """Build FAISS vector index from documents using Hugging Face embeddings"""
        texts = [doc['content'] for doc in documents]
        print(f"Generating embeddings for {len(texts)} documents...")
        
        embeddings = self.get_embeddings(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents = texts
        self.metadata = documents
        
        # Save index and metadata
        self.save_index()
        print("Vector index built and saved successfully!")
    
    def save_index(self):
        """Save FAISS index and metadata"""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata
                }, f)
    
    def load_index(self) -> bool:
        """Load existing FAISS index and metadata"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadata = data['metadata']
                print("Existing vector index loaded successfully!")
                return True
        except Exception as e:
            print(f"Error loading index: {e}")
        return False
    
    def load_or_create_index(self):
        """Load existing index or create new one"""
        if not self.load_index():
            print("Creating new vector index with Hugging Face embeddings...")
            documents = self.create_comprehensive_documents()
            self.build_vector_index(documents)
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search for similar documents using FAISS and Hugging Face embeddings"""
        query_embedding = self.get_embeddings([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents) and score > 0.1:  # Minimum similarity threshold
                results.append((
                    self.documents[idx],
                    float(score),
                    self.metadata[idx]
                ))
        
        return results
    
    def classify_query_intent(self, query: str) -> str:
        """Enhanced query intent classification"""
        query_lower = query.lower()
        
        # Dosha-related queries
        if any(word in query_lower for word in ['vata', 'pitta', 'kapha', 'dosha', 'constitution', 'prakriti']):
            return 'dosha'
        
        # Herb and medicine queries
        elif any(word in query_lower for word in ['herb', 'medicine', 'remedy', 'treatment', 'ashwagandha', 
                                                 'turmeric', 'triphala', 'brahmi', 'neem']):
            return 'herb'
        
        # Practice and lifestyle queries
        elif any(word in query_lower for word in ['practice', 'routine', 'lifestyle', 'yoga', 'meditation', 
                                                 'pranayama', 'abhyanga', 'dinacharya']):
            return 'practice'
        
        # Diet and nutrition queries
        elif any(word in query_lower for word in ['diet', 'food', 'eat', 'nutrition', 'digestion', 'agni']):
            return 'diet'
        
        # Symptom and health queries
        elif any(word in query_lower for word in ['symptom', 'problem', 'issue', 'pain', 'sick', 'disease', 
                                                 'health', 'cure', 'heal']):
            return 'symptom'
        
        # Concept queries
        elif any(word in query_lower for word in ['concept', 'principle', 'theory', 'philosophy', 'ojas', 
                                                 'prana', 'ama', 'tejas']):
            return 'concept'
        
        else:
            return 'general'
    
    def generate_enhanced_response(self, query: str) -> str:
        """Generate enhanced response using retrieved documents and optional LLM"""
        intent = self.classify_query_intent(query)
        similar_docs = self.search_similar_documents(query, k=3)
        
        if not similar_docs:
            return """I apologize, but I couldn't find specific information about your query in my Ayurvedic knowledge base. 
            Could you please rephrase your question or ask about specific topics like doshas (Vata, Pitta, Kapha), 
            Ayurvedic herbs, practices, or general health principles?"""
        
        # Build comprehensive response
        response_parts = []
        response_parts.append("🕉️ **Based on Ayurvedic wisdom:**\n")
        
        # Add relevant information from retrieved documents
        for i, (doc, score, metadata) in enumerate(similar_docs):
            if score > 0.3:  # Higher similarity threshold
                # Extract key information
                lines = doc.split('\n')
                relevant_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 20][:3]
                
                if relevant_lines:
                    response_parts.append(f"📚 **{metadata['title']}:**")
                    for line in relevant_lines:
                        if not line.endswith(':'):
                            response_parts.append(f"• {line}")
                    response_parts.append("")
        
        # Add intent-specific guidance
        guidance = self.get_intent_specific_guidance(intent)
        if guidance:
            response_parts.append(f"💡 **Additional Guidance:**\n{guidance}")
        
        # Use LLM for enhanced response if available
        if self.text_generator and len(' '.join(response_parts)) < 300:
            try:
                context = ' '.join(response_parts)
                prompt = f"Based on Ayurvedic principles, {query}? {context}"
                
                generated = self.text_generator(prompt, max_length=200, num_return_sequences=1)
                if generated and len(generated) > 0:
                    generated_text = generated[0]['generated_text'].replace(prompt, '').strip()
                    if generated_text and len(generated_text) > 20:
                        response_parts.append(f"\n🤖 **AI Enhancement:**\n{generated_text}")
            except Exception as e:
                print(f"LLM generation error: {e}")
        
        return '\n'.join(response_parts)
    
    def get_intent_specific_guidance(self, intent: str) -> str:
        """Get specific guidance based on query intent"""
        guidance_map = {
            'dosha': """Remember that everyone has all three doshas, but in different proportions. 
            Understanding your unique constitution (Prakriti) and current imbalances (Vikriti) is key to 
            applying Ayurvedic principles effectively. Consider consulting an Ayurvedic practitioner for 
            personalized assessment.""",
            
            'herb': """⚠️ Please consult with a qualified Ayurvedic practitioner or healthcare provider before 
            using any herbal remedies, especially if you have existing health conditions, are pregnant, 
            or are taking medications. Herbs can interact with medications and may not be suitable for everyone.""",
            
            'practice': """Consistency and gradual implementation are key to experiencing the benefits of 
            Ayurvedic practices. Start with one or two practices and build them into your routine before 
            adding more. Listen to your body and adjust practices according to your constitution and current needs.""",
            
            'diet': """Ayurvedic nutrition is highly individualized. What works for one person may not work 
            for another. Pay attention to how different foods make you feel and adjust your diet accordingly. 
            Eating mindfully and in a peaceful environment is as important as what you eat.""",
            
            'symptom': """While Ayurveda offers valuable insights for health and wellness, persistent or 
            serious health issues should be evaluated by qualified healthcare providers. Ayurvedic principles 
            can complement conventional medical care but should not replace professional medical advice for 
            serious conditions.""",
            
            'concept': """Ayurvedic concepts are interconnected and form a comprehensive system of understanding 
            health and life. Take time to understand these foundational principles as they will help you apply 
            Ayurvedic wisdom more effectively in your daily life."""
        }
        
        return guidance_map.get(intent, "")
    
    def get_enhanced_dosha_questions(self) -> List[Dict]:
        """Enhanced dosha assessment questions"""
        return [
            {
                'question': 'What is your natural body build and weight?',
                'options': [
                    'Thin, light frame, hard to gain weight (Vata)',
                    'Medium build, moderate weight, muscular (Pitta)', 
                    'Large frame, heavy build, easy weight gain (Kapha)'
                ],
                'dosha_weights': {'vata': [3, 1, 0], 'pitta': [0, 3, 1], 'kapha': [0, 1, 3]}
            },
            {
                'question': 'How is your skin and hair naturally?',
                'options': [
                    'Dry, rough skin; thin, dry hair (Vata)',
                    'Warm, oily skin; fine, oily hair, early graying (Pitta)',
                    'Thick, oily skin; thick, wavy, lustrous hair (Kapha)'
                ],
                'dosha_weights': {'vata': [3, 1, 0], 'pitta': [0, 3, 1], 'kapha': [0, 1, 3]}
            },
            {
                'question': 'How do you typically handle stress and emotions?',
                'options': [
                    'Anxious, worried, changeable emotions (Vata)',
                    'Irritated, angry, intense emotions (Pitta)',
                    'Calm, withdrawn, steady emotions (Kapha)'
                ],
                'dosha_weights': {'vata': [3, 1, 0], 'pitta': [0, 3, 1], 'kapha': [0, 1, 3]}
            },
            {
                'question': 'What is your natural energy and activity pattern?',
                'options': [
                    'Variable energy, bursts of activity, tire easily (Vata)',
                    'Intense, focused energy, goal-oriented (Pitta)',
                    'Steady, enduring energy, slow and deliberate (Kapha)'
                ],
                'dosha_weights': {'vata': [3, 1, 0], 'pitta': [0, 3, 1], 'kapha': [0, 1, 3]}
            },
            {
                'question': 'How is your appetite and digestion?',
                'options': [
                    'Variable appetite, irregular digestion, gas/bloating (Vata)',
                    'Strong appetite, intense digestion, heartburn tendency (Pitta)',
                    'Slow appetite, heavy digestion, sluggish metabolism (Kapha)'
                ],
                'dosha_weights': {'vata': [3, 1, 0], 'pitta': [0, 3, 1], 'kapha': [0, 1, 3]}
            },
            {
                'question': 'What is your sleep pattern?',
                'options': [
                    'Light sleeper, difficulty falling asleep, restless (Vata)',
                    'Moderate sleep, vivid dreams, wake up refreshed (Pitta)',
                    'Deep sleeper, long sleep, hard to wake up (Kapha)'
                ],
                'dosha_weights': {'vata': [3, 1, 0], 'pitta': [0, 3, 1], 'kapha': [0, 1, 3]}
            },
            {
                'question': 'How do you respond to weather?',
                'options': [
                    'Dislike cold and wind, prefer warmth (Vata)',
                    'Dislike heat and sun, prefer cool weather (Pitta)',
                    'Dislike cold and damp, prefer warm dry weather (Kapha)'
                ],
                'dosha_weights': {'vata': [3, 1, 0], 'pitta': [0, 3, 1], 'kapha': [0, 1, 3]}
            },
            {
                'question': 'What is your mental and learning style?',
                'options': [
                    'Quick to learn, quick to forget, creative but scattered (Vata)',
                    'Sharp intellect, focused learning, good memory (Pitta)',
                    'Slow to learn but excellent retention, methodical (Kapha)'
                ],
                'dosha_weights': {'vata': [3, 1, 0], 'pitta': [0, 3, 1], 'kapha': [0, 1, 3]}
            }
        ]

def create_streamlit_app():
    pass