"""
Knowledge Base module for Enhanced AyurvaBot
Contains all Ayurvedic knowledge chunks and management functions.
"""

class AyurvedaKnowledgeBase:
    """
    Class to manage Ayurvedic knowledge base for RAG system.
    """
    
    def __init__(self):
        self.knowledge_chunks = []
        self.document_chunks = []
        
    def load_knowledge_chunks(self):
        """
        Load comprehensive Ayurvedic knowledge chunks.
        """
        ayurveda_texts = [
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
            
            # Dosha knowledge
            "The three doshas - Vata, Pitta, and Kapha - are fundamental energies that govern all physiological and psychological functions.",
            "Vata dosha is composed of air and space elements and controls movement, circulation, breathing, and nervous system functions.",
            "Pitta dosha is composed of fire and water elements and governs digestion, metabolism, body temperature, and transformation processes.",
            "Kapha dosha is composed of earth and water elements and provides structure, stability, immunity, and lubrication to the body.",
            "Balanced doshas result in good health while imbalanced doshas lead to various diseases and health problems.",
            
            # Panchakarma knowledge
            "Panchakarma is a comprehensive detoxification and rejuvenation program consisting of five therapeutic procedures.",
            "Vamana is therapeutic vomiting used to eliminate excess Kapha dosha from the upper respiratory tract.",
            "Virechana is purgation therapy used to eliminate excess Pitta dosha from the small intestine.",
            "Basti involves medicated enemas to eliminate excess Vata dosha from the colon.",
            "Nasya is nasal administration of medicines to treat disorders of head and neck region.",
            "Raktamokshana is bloodletting therapy used to eliminate toxins from blood and treat skin disorders.",
            
            # General Ayurveda principles
            "Ayurveda literally means 'science of life' and is one of the world's oldest healing systems focusing on prevention.",
            "The fundamental principle of Ayurveda is that health depends on delicate balance between mind, body, and spirit.",
            "Ayurveda emphasizes prevention and holistic healing through natural remedies, lifestyle practices, and dietary guidelines.",
            "Individual constitution (Prakriti) determines the most suitable diet, lifestyle, and treatment for each person.",
            "Ayurvedic treatment focuses on removing the root cause of disease rather than just treating symptoms.",
            
            # Additional herb knowledge
            "Amla (Indian Gooseberry) is rich in Vitamin C and is excellent for immunity, hair health, and overall vitality.",
            "Ashwagandha is an adaptogenic herb that helps the body manage stress and improves energy levels.",
            "Brahmi is a brain tonic that enhances memory, concentration, and mental clarity.",
            "Shankhpushpi is another brain tonic used for improving cognitive function and reducing anxiety.",
            "Jatamansi is used for treating insomnia, anxiety, and nervous disorders in Ayurveda."
        ]
        
        # Create document chunks for RAG
        self.document_chunks = []
        for i, text in enumerate(ayurveda_texts):
            chunk = {
                'id': i,
                'text': text,
                'source': 'Ayurveda Knowledge Base',
                'topic': self._extract_topic_from_text(text)
            }
            self.document_chunks.append(chunk)
        
        self.knowledge_chunks = [chunk['text'] for chunk in self.document_chunks]
        return len(self.knowledge_chunks)
    
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
