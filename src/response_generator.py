"""
Response Generator module for Enhanced AyurvaBot
Handles response formatting and generation logic.
"""

class ResponseGenerator:
    """
    Generates structured responses using RAG system and templates.
    """
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
    
    def generate_response(self, message):
        """
        Generate a comprehensive response to the user's message.
        """
        # Handle empty or very short queries
        if not message or len(message.strip()) < 3:
            return "Please ask a more specific question about Ayurveda, such as 'What is Ayurveda?' or 'What are the three doshas?'"

        message_lower = message.lower()
        
        # Try RAG approach first if system is ready
        if self.rag_system.is_ready():
            try:
                # Use RAG pipeline
                rag_answer, retrieved_chunks = self.rag_system.rag_pipeline(message)
                
                if rag_answer and retrieved_chunks:
                    # Format enhanced RAG response
                    enhanced_response = self._format_rag_response(message, rag_answer, retrieved_chunks)
                    return enhanced_response
            except Exception as e:
                print(f"RAG processing error: {e}")
                # Fall back to template-based responses
        
        # Fallback to template-based responses
        return self._generate_template_response(message, message_lower)
    
    def _format_rag_response(self, query, rag_answer, retrieved_chunks):
        """
        Format a structured RAG response.
        """
        try:
            # Determine query category
            category = self._categorize_query(query.lower())
            
            # Get appropriate template response
            template_response = self._get_template_response(category, query)
            
            # Add RAG information
            rag_info = f"""

---
**📊 RAG Analysis**: Retrieved {len(retrieved_chunks)} relevant sources with {retrieved_chunks[0]['similarity_score']:.2f} similarity score

**🤖 AI-Generated Answer**: {rag_answer}

**📚 Retrieved Sources**:
"""
            
            for i, chunk in enumerate(retrieved_chunks[:3], 1):
                rag_info += f"• **Source {i}** (Similarity: {chunk['similarity_score']:.2f}): {chunk['text'][:100]}...\n"
            
            return template_response + rag_info
            
        except Exception as e:
            print(f"Error formatting RAG response: {e}")
            return f"Based on Ayurvedic principles, {rag_answer}.\n\n💡 *Generated using RAG technology*"
    
    def _generate_template_response(self, message, message_lower):
        """
        Generate response using templates when RAG is not available.
        """
        category = self._categorize_query(message_lower)
        return self._get_template_response(category, message)
    
    def _categorize_query(self, query_lower):
        """
        Categorize the query based on keywords.
        """
        categories = {
            'fever': ['fever', 'jwara', 'temperature', 'pyrexia'],
            'heart': ['heart', 'cardiac', 'arjuna', 'cardiovascular'],
            'respiratory': ['cold', 'cough', 'respiratory', 'breathing'],
            'digestive': ['digestion', 'digestive', 'stomach', 'acidity', 'gastric'],
            'doshas': ['dosha', 'vata', 'pitta', 'kapha'],
            'ayurveda': ['ayurveda', 'what is ayurveda'],
            'panchakarma': ['panchakarma', 'vamana', 'virechana', 'basti', 'nasya'],
            'herbs': ['herb', 'medicine', 'remedy', 'treatment']
        }
        
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def _get_template_response(self, category, query):
        """
        Get template response for the category.
        """
        if category == 'ayurveda':
            return self._ayurveda_template()
        elif category == 'fever':
            return self._fever_template()
        elif category == 'respiratory':
            return self._respiratory_template()
        elif category == 'heart':
            return self._heart_template()
        elif category == 'digestive':
            return self._digestive_template()
        elif category == 'doshas':
            return self._doshas_template()
        elif category == 'panchakarma':
            return self._panchakarma_template()
        elif category == 'herbs':
            return self._herbs_template()
        else:
            return self._general_template(query)
    
    def _ayurveda_template(self):
        """Template for Ayurveda overview questions."""
        return """In Ayurveda, the term literally means "science of life" (Ayur = life, Veda = knowledge) and represents one of the world's oldest healing systems. Ayurveda focuses on prevention and holistic healing through natural remedies, lifestyle practices, and maintaining balance between mind, body, and spirit.

**🔶 Core Principles of Ayurveda**

1. **Three Doshas (Bio-energies)**
   • **Vata** (Air + Space): Controls movement, circulation, breathing
   • **Pitta** (Fire + Water): Governs digestion, metabolism, transformation
   • **Kapha** (Earth + Water): Provides structure, stability, immunity

2. **Individual Constitution (Prakriti)**
   • Each person has a unique combination of doshas
   • Treatment is personalized based on constitution
   • Diet and lifestyle recommendations vary by individual

3. **Prevention Focus**
   • Emphasis on maintaining health rather than treating disease
   • Daily routines (Dinacharya) and seasonal practices (Ritucharya)
   • Balance through proper diet, exercise, and mental practices

**🔷 Ayurvedic Treatment Approaches**

1. **Panchakarma (Detoxification)**
   • Five therapeutic procedures for deep cleansing
   • Removes accumulated toxins (ama) from the body

2. **Herbal Medicine**
   • Natural herbs and formulations
   • Minimal side effects when used properly

3. **Lifestyle Medicine**
   • Yoga, meditation, pranayama
   • Proper sleep, eating habits, daily routines

**⚖️ Fundamental Philosophy**
• Health is a state of balance between doshas, tissues, and waste products
• Disease occurs when this balance is disturbed
• Treatment focuses on restoring natural balance

**🧘‍♂️ Modern Relevance**
• Integrative approach combining traditional wisdom with modern science
• Growing recognition in preventive and lifestyle medicine
• Emphasis on personalized healthcare

💡 *Ayurveda offers a comprehensive system for understanding health and disease, providing practical tools for maintaining wellness throughout life.*"""

    def _fever_template(self):
        """Template for fever-related questions."""
        return """In Ayurveda, fever is called "Jwara" and is considered the "king of all diseases." It typically results from aggravated Pitta dosha combined with accumulated toxins (ama) in the body.

**🔶 Ayurvedic Herbs and Remedies for Fever**

1. **Tulsi (Holy Basil)**
   • **How to use**: Boil 10-15 fresh leaves in water, drink as tea 2-3 times daily
   • **Effect**: Natural antipyretic, reduces body temperature, boosts immunity

2. **Ginger (Adrak)**
   • **How to use**: Mix fresh ginger juice with honey, or boil with tea
   • **Effect**: Promotes sweating to break fever naturally, eliminates toxins

3. **Coriander (Dhania)**
   • **How to use**: Soak 1 tsp coriander seeds overnight, drink the water in morning
   • **Effect**: Cooling herb that pacifies aggravated Pitta dosha

4. **Neem**
   • **How to use**: Boil neem leaves in water, drink when cool (bitter taste)
   • **Effect**: Powerful antibacterial, treats infection-related fevers

5. **Giloy (Guduchi)**
   • **How to use**: Boil giloy stem in water or take as powder with honey
   • **Effect**: Excellent immunity booster, helps fight fever-causing pathogens

**🔷 Ayurvedic Lifestyle Practices**

1. **Rest and Hydration**
   • Complete bed rest to conserve energy for healing
   • Drink warm water, herbal teas, and fresh fruit juices

2. **Light Diet**
   • Easily digestible foods like rice porridge (khichdi)
   • Avoid heavy, oily, or cold foods during fever

**⚖️ Dietary Guidelines During Fever**
• **Eat**: Light soups, herbal teas, fresh fruit juices, rice water
• **Avoid**: Heavy meals, dairy products, fried foods, cold drinks

**⚠️ Important Note**
If fever persists for more than 3 days, reaches above 103°F (39.4°C), or is accompanied by severe symptoms, consult a qualified healthcare practitioner immediately.

💡 *Would you like a specific herbal decoction (kadha) recipe for fever management?*"""

    def _general_template(self, query):
        """Template for general questions."""
        return f"""I understand you are asking about: "{query}"

**🌿 This is a comprehensive Ayurvedic knowledge system. Here are specific topics I can help with:**

**🔹 Fundamental Concepts:**
• **What is Ayurveda?** - Learn about the ancient healing system
• **What are the three doshas?** - Understand Vata, Pitta, and Kapha
• **What is Panchakarma?** - Discover the detoxification process

**🔹 Health & Treatment:**
• **Fever treatment** - Natural remedies for fever (Jwara)
• **Heart medicines** - Ayurvedic cardiovascular treatments
• **Cold and cough** - Respiratory health treatments
• **Digestive herbs** - Natural remedies for stomach problems
• **Immunity boosters** - Herbs for strong immune system

**🔹 Specific Doshas:**
• **What is Vata dosha?** - Learn about the air element
• **What is Pitta dosha?** - Understand the fire element
• **What is Kapha dosha?** - Discover the earth element

**💡 Try asking about specific health concerns, herbs, or treatments for detailed Ayurvedic guidance!**"""

    def _respiratory_template(self):
        return "Respiratory health information..."
    
    def _heart_template(self):
        return "Heart health information..."
    
    def _digestive_template(self):
        return "Digestive health information..."
    
    def _doshas_template(self):
        return "Dosha information..."
    
    def _panchakarma_template(self):
        return "Panchakarma information..."
    
    def _herbs_template(self):
        return "Herbs information..."
