"""
Knowledge Base module for Enhanced AyurvaBot
Contains all Ayurvedic knowledge chunks and management functions.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any

from .chunking import semantic_chunk
from .entity_extractor import EntityExtractor
from .dataset_ingestion import ingest_dataset


class AyurvedaKnowledgeBase:
    """
    Class to manage Ayurvedic knowledge base for RAG system.
    """
    
    def __init__(self, config=None):
        self.config = config
        self.knowledge_chunks: List[str] = []
        self.document_chunks: List[Dict[str, Any]] = []
        use_spacy = False
        max_chars = 4000
        if config is not None:
            use_spacy = getattr(config, 'ENABLE_SPACY_NER', False)
            max_chars = getattr(config, 'NER_MAX_CHARS', 4000)
        self.entity_extractor = EntityExtractor(use_spacy=use_spacy, max_chars=max_chars)
        if not use_spacy:
            # Force disable heavy model just in case
            self.entity_extractor.model = None
        
    def load_knowledge_chunks(self, use_csv: bool = False, use_synthetic_json: bool = True, config: Any | None = None):
        """
        Load comprehensive Ayurvedic knowledge chunks from multiple sources.
        """
        all_texts = []
        
        # Optionally load from old CSV (disabled by default as per user request)
        if use_csv:
            try:
                csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'ayurveda_documents.csv')
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    for _, row in df.iterrows():
                        all_texts.append({
                            'text': row['content'],
                            'source': row['source'],
                            'type': 'csv_dataset'
                        })
                    print(f"✅ Loaded {len(df)} chunks from CSV dataset")
            except Exception as e:
                print(f"⚠️ Could not load CSV dataset: {e}")
        
        # Load from synthetic JSON dataset (optional)
        if use_synthetic_json:
            try:
                json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'synthetic_ayurveda_dataset.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    for item in json_data:
                        all_texts.append({
                            'text': f"Q: {item['question']} A: {item['answer']}",
                            'source': 'synthetic_dataset',
                            'type': 'qa_pair'
                        })
                        if 'context' in item and item['context']:
                            all_texts.append({
                                'text': item['context'],
                                'source': 'synthetic_dataset',
                                'type': 'context'
                            })
                    print(f"✅ Loaded {len(json_data)} Q&A pairs from JSON dataset")
            except Exception as e:
                print(f"⚠️ Could not load JSON dataset: {e}")

        # Ingest primary Ayurveda dataset directory (PDF/TXT) per user requirement
        if config and getattr(config, 'DATASET_PATH', None):
            try:
                dataset_docs = ingest_dataset(
                    dataset_path=config.DATASET_PATH,
                    cache_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output'),
                    max_pages=config.MAX_PAGES_PER_PDF,
                    min_chars=config.MIN_CHARS_DOCUMENT,
                    max_files=getattr(config, 'MAX_INGEST_FILES', None),
                )
                all_texts.extend(dataset_docs)
                print(f"✅ Added {len(dataset_docs)} documents from primary dataset")
            except Exception as e:
                print(f"⚠️ Dataset ingestion failed: {e}")
        
        # Add curated knowledge chunks for comprehensive coverage
        curated_texts = [
            # Fever (Jwara) knowledge
            "Fever in Ayurveda is called Jwara and is considered the king of all diseases. It is caused by aggravated Pitta dosha and accumulated toxins (ama) in the body.",
            "Tulsi (Holy Basil) is the most effective natural antipyretic herb that reduces body temperature and boosts immunity during fever.",
            "Ginger promotes sweating which helps break fever naturally by eliminating toxins through perspiration.",
            "Coriander seeds are cooling herbs that pacify aggravated Pitta dosha and reduce fever effectively.",
            "Neem has powerful antibacterial properties and treats fever caused by bacterial infections.",
            "Giloy (Guduchi) is an excellent immunity booster that helps the body fight fever-causing pathogens.",
            "Sudarshan Churna is a classical Ayurvedic formulation used for treating all types of fever including chronic and intermittent fevers.",
            
            # Heart health knowledge
            "Heart health in Ayurveda is governed by Vyana Vata which controls circulation and Sadhaka Pitta which governs emotions and heart function.",
            "Arjuna (Terminalia arjuna) is the most important heart tonic in Ayurveda that strengthens heart muscles and improves cardiac function.",
            "Brahmi reduces mental stress and anxiety that negatively affect heart health through the nervous system.",
            "Ashwagandha enhances overall heart strength and reduces stress-related cardiac issues by balancing cortisol levels.",
            "Punarnava improves circulation and helps with fluid retention around the heart area.",
            "Guggulu is traditional medicine for managing cholesterol levels and supporting overall cardiovascular health.",
            
            # Respiratory health knowledge
            "Cold and cough in Ayurveda are caused by aggravated Kapha dosha and weakened digestive fire (Agni).",
            "Tulsi is the best herb for respiratory health, natural immunity, and treating cold and cough symptoms.",
            "Ginger and honey combination provides warming effect that soothes throat and reduces congestion.",
            "Turmeric milk has anti-inflammatory properties that reduce throat irritation and respiratory inflammation.",
            "Black pepper helps clear respiratory passages and reduces mucus accumulation in lungs.",
            "Licorice (Mulethi) is a natural cough suppressant and throat soother that reduces respiratory irritation.",
            "Trikatu (three spices: ginger, black pepper, long pepper) is excellent for respiratory health and reducing Kapha.",
            
            # Digestive health knowledge
            "Digestive problems stem from weak digestive fire (Agni) and imbalanced doshas, particularly Pitta and Vata.",
            "Ginger is the universal digestive herb that kindles Agni and improves appetite and digestion.",
            "Fennel is a cooling herb that reduces gas, bloating, and stomach discomfort effectively.",
            "Cumin enhances digestion and helps with proper nutrient absorption in the intestines.",
            "Ajwain (Carom seeds) is excellent for stomach pain, indigestion, and gas-related problems.",
            "Triphala is a three-fruit combination that provides comprehensive digestive wellness and detoxification.",
            "Hing (Asafoetida) is a powerful anti-flatulent herb and digestive stimulant that reduces bloating.",
            
            # Comprehensive Ayurvedic concepts
            "Ojas is the vital essence that provides immunity, strength, and vitality to the body and mind.",
            "Tejas is the subtle fire element that governs metabolism, digestion, and mental clarity.",
            "Prana is the life force energy that controls all vital functions including breathing and circulation.",
            "Ama refers to undigested toxins that accumulate in the body due to weak digestive fire.",
            "Agni is the digestive fire responsible for all metabolic processes and transformation in the body.",
            "Srotas are the channels or pathways through which nutrients, waste, and energy flow in the body.",
            "Dhatus are the seven body tissues: plasma, blood, muscle, fat, bone, nerve, and reproductive tissue.",
            "Malas are the waste products of the body including urine, feces, and sweat that need regular elimination.",
            
            # Lifestyle and daily routine
            "Dinacharya is the daily routine that aligns with natural rhythms to maintain health and prevent disease.",
            "Ritucharya is the seasonal routine that helps adapt to changing environmental conditions throughout the year.",
            "Brahma muhurta (4-6 AM) is the ideal time for waking up, meditation, and spiritual practices.",
            "Oil massage (Abhyanga) should be done daily to nourish the skin, improve circulation, and calm the nervous system.",
            "Yoga and pranayama are essential practices for maintaining physical flexibility and mental balance.",
            "Meditation helps calm the mind, reduce stress, and develop inner awareness and peace.",
            
            # Diet and nutrition principles
            "Food should be fresh, seasonal, and prepared with love and positive intention for optimal nourishment.",
            "Eating in a calm, peaceful environment aids proper digestion and nutrient absorption.",
            "The largest meal should be consumed at midday when digestive fire is strongest.",
            "Incompatible food combinations (Viruddha Ahara) can create toxins and disturb digestion.",
            "Six tastes (sweet, sour, salty, pungent, bitter, astringent) should be included in every meal for balance.",
            "Drinking warm water throughout the day helps maintain proper hydration and supports digestion.",
            
            # Mental health and emotional well-being
            "Sattva, Rajas, and Tamas are the three mental qualities that influence psychological health and behavior.",
            "Satvavajaya Chikitsa is psychotherapy in Ayurveda that addresses mental and emotional imbalances.",
            "Positive thinking, gratitude, and contentment are essential for mental health and spiritual growth.",
            "Excessive desires, anger, and attachment are considered root causes of mental suffering.",
            "Regular spiritual practices help develop equanimity and inner peace regardless of external circumstances.",
            
            # Women's health
            "Shatavari is the primary herb for women's reproductive health and hormonal balance.",
            "Menstrual health depends on proper Apana Vata function and adequate nourishment of reproductive tissues.",
            "Pregnancy requires special care with appropriate diet, herbs, and lifestyle practices for mother and child.",
            "Postpartum care focuses on rebuilding strength, supporting lactation, and restoring hormonal balance.",
            
            # Men's health
            "Ashwagandha and Safed Musli are important herbs for male reproductive health and vitality.",
            "Shukra dhatu (reproductive tissue) requires proper nourishment through appropriate diet and lifestyle.",
            "Stress management is crucial for maintaining healthy testosterone levels and reproductive function.",
            
            # Skin and beauty
            "Healthy skin reflects internal health and proper functioning of liver, kidneys, and digestive system.",
            "Natural skincare uses herbs like turmeric, neem, rose, and sandalwood for different skin types.",
            "Beauty in Ayurveda comes from inner radiance achieved through balanced doshas and pure mind.",
            "Premature aging is caused by excessive stress, poor diet, and imbalanced lifestyle habits.",
            
            # Immunity and disease prevention
            "Strong immunity (Ojas) depends on proper digestion, adequate sleep, and balanced emotional state.",
            "Rasayana therapy uses rejuvenative herbs and practices to enhance immunity and longevity.",
            "Seasonal cleansing helps remove accumulated toxins and maintain optimal health throughout the year.",
            "Prevention is always better than cure, focusing on maintaining health rather than treating disease.",
            
            # Pain and inflammation
            "Joint pain and arthritis are primarily Vata disorders requiring warming, nourishing treatments.",
            "Inflammation is usually a Pitta imbalance that responds well to cooling, anti-inflammatory herbs.",
            "Chronic pain often involves multiple doshas and requires comprehensive, individualized treatment.",
            "Natural pain relief uses herbs like turmeric, ginger, boswellia, and guggulu without side effects.",
            
            # Sleep and rest
            "Quality sleep is essential for physical recovery, mental clarity, and emotional balance.",
            "Insomnia is often caused by excess Vata or Pitta and requires calming, grounding practices.",
            "Sleep hygiene includes regular bedtime, comfortable environment, and avoiding stimulants before bed.",
            "Natural sleep aids include warm milk with nutmeg, brahmi, and jatamansi herbs.",
            
            # Energy and vitality
            "Low energy often results from weak digestion, poor sleep, or emotional stress and imbalance.",
            "Natural energy boosters include proper nutrition, regular exercise, and stress management techniques.",
            "Chronic fatigue may indicate deeper imbalances requiring comprehensive Ayurvedic evaluation and treatment.",
            "Sustainable energy comes from balanced lifestyle rather than artificial stimulants or quick fixes."
        ]
        
        # Add curated texts
        for text in curated_texts:
            all_texts.append({
                'text': text,
                'source': 'curated_knowledge',
                'type': 'curated'
            })
        
        # Advanced chunking: break longer texts into semantic chunks, keep short ones as-is
        self.document_chunks = []
        idx = 0
        for text_data in all_texts:
            raw_text = text_data['text']
            # decide if we chunk (heuristic: > 220 words)
            words = raw_text.split()
            if len(words) > 220:
                sem_chunks = semantic_chunk(raw_text)
                for ch in sem_chunks:
                    entities = self.entity_extractor.extract(ch['text'])
                    chunk = {
                        'id': idx,
                        'text': ch['text'],
                        'summary': ch.get('summary', ch['text'][:160]),
                        'entities': entities,
                        'source': text_data['source'],
                        'type': text_data['type'],
                        'topic': self._extract_topic_from_text(ch['text'])
                    }
                    self.document_chunks.append(chunk)
                    idx += 1
            else:
                entities = self.entity_extractor.extract(raw_text)
                self.document_chunks.append({
                    'id': idx,
                    'text': raw_text,
                    'summary': raw_text[:160],
                    'entities': entities,
                    'source': text_data['source'],
                    'type': text_data['type'],
                    'topic': self._extract_topic_from_text(raw_text)
                })
                idx += 1
        # finalize knowledge chunks list
        self.knowledge_chunks = [chunk['text'] for chunk in self.document_chunks]
        # Deduplicate & limit
        self._deduplicate_and_limit()
        self.knowledge_chunks = [chunk['text'] for chunk in self.document_chunks]
        print(f"✅ Total knowledge chunks loaded: {len(self.knowledge_chunks)}")
        # Compatibility line the user expects (explicit loaded line)
        print(f"✅ Loaded {len(self.knowledge_chunks)} knowledge chunks")
        print(f"Docs: {len(self.knowledge_chunks)}")
        return len(self.knowledge_chunks)

    def _deduplicate_and_limit(self):
        if not self.document_chunks:
            return
        if not self.config:
            return
        # Skip any limiting/dedup if UNLIMITED_INGEST is enabled
        if getattr(self.config, 'UNLIMITED_INGEST', False):
            return
        max_chunks = getattr(self.config, 'MAX_TOTAL_CHUNKS', None)
        min_chars = getattr(self.config, 'DEDUPE_MIN_CHARS', 100)
        hash_len = getattr(self.config, 'DEDUPE_HASH_LEN', 300)
        seen = set()
        filtered = []
        for ch in self.document_chunks:
            txt = ch.get('text', '')
            if len(txt) < min_chars:
                continue
            norm = ' '.join(txt.lower().split())[:hash_len]
            if norm in seen:
                continue
            seen.add(norm)
            filtered.append(ch)
            if max_chunks and len(filtered) >= max_chunks:
                break
        removed = len(self.document_chunks) - len(filtered)
        if removed > 0:
            print(f"ℹ️ Chunk reduction: removed {removed} duplicates/overflow (kept {len(filtered)})")
        self.document_chunks = filtered
    
    def _extract_topic_from_text(self, text):
        """
        Extract topic from text for better organization.
        """
        text_lower = text.lower()
        if any(word in text_lower for word in ['fever', 'jwara', 'temperature']):
            return 'fever'
        elif any(word in text_lower for word in ['heart', 'cardiac', 'arjuna']):
            return 'heart'
        elif any(word in text_lower for word in ['cold', 'cough', 'respiratory']):
            return 'respiratory'
        elif any(word in text_lower for word in ['digestion', 'digestive', 'stomach']):
            return 'digestive'
        elif any(word in text_lower for word in ['dosha', 'vata', 'pitta', 'kapha']):
            return 'doshas'
        elif any(word in text_lower for word in ['panchakarma', 'vamana', 'virechana']):
            return 'panchakarma'
        elif any(word in text_lower for word in ['amla', 'ashwagandha', 'brahmi', 'herb']):
            return 'herbs'
        else:
            return 'general'
    
    def get_knowledge_chunks(self):
        """
        Get all knowledge chunks as list of strings.
        """
        return self.knowledge_chunks
    
    def get_document_chunks(self):
        """
        Get all document chunks with metadata.
        """
        return self.document_chunks
    
    def get_chunks_by_topic(self, topic):
        """
        Get chunks filtered by topic.
        """
        return [chunk for chunk in self.document_chunks if chunk['topic'] == topic]
