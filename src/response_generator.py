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
                
                if retrieved_chunks:  # If we have relevant chunks, use RAG approach
                    # Format enhanced RAG response
                    enhanced_response = self._format_rag_response(message, rag_answer, retrieved_chunks)
                    return enhanced_response
                else:
                    print("No relevant chunks found, falling back to templates")
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
            # Check if we have a good RAG answer
            has_good_rag_answer = rag_answer and len(rag_answer.strip()) > 20 and not rag_answer.lower().startswith('i don')
            
            # Determine query category for fallback
            category = self._categorize_query(query.lower())
            is_specific_category = category != 'general'
            
            # Decision logic for response format
            if has_good_rag_answer and not is_specific_category:
                # For generic queries with good RAG answers, use RAG as primary
                enhanced_response = f"""**🌿 Ayurvedic Knowledge Response**

{rag_answer}

---
**📊 Sources**: Based on {len(retrieved_chunks)} relevant Ayurvedic references (Top similarity: {retrieved_chunks[0]['similarity_score']:.2f})

**📚 Key References**:
"""
                
                for i, chunk in enumerate(retrieved_chunks[:3], 1):
                    source_type = chunk.get('type', 'knowledge')
                    enhanced_response += f"• **Source {i}** ({source_type}): {chunk['text'][:120]}...\n"
                
                enhanced_response += "\n💡 *This response is generated from traditional Ayurvedic knowledge using AI analysis.*"
                return enhanced_response
                
            elif is_specific_category:
                # For specific categories, use comprehensive template with RAG enhancement
                template_response = self._get_template_response(category, query)
                
                if has_good_rag_answer:
                    # Add RAG insights to template
                    rag_enhancement = f"""

---
**🔍 Additional AI Insights**: {rag_answer}

**📚 Supporting References** (from {len(retrieved_chunks)} sources):
"""
                    for i, chunk in enumerate(retrieved_chunks[:2], 1):
                        rag_enhancement += f"• {chunk['text'][:100]}...\n"
                    
                    return template_response + rag_enhancement
                else:
                    # Just add source info
                    source_info = f"""

---
**📚 Enhanced with Knowledge Base**: {len(retrieved_chunks)} relevant sources found
💡 *Response combines curated templates with retrieved knowledge*"""
                    return template_response + source_info
            
            else:
                # Generic query without good RAG answer - create response from retrieved chunks
                if retrieved_chunks:
                    combined_info = self._create_response_from_chunks(query, retrieved_chunks)
                    return combined_info
                else:
                    # Final fallback
                    return self._get_template_response('general', query)
            
        except Exception as e:
            print(f"Error formatting RAG response: {e}")
            # Fallback to template response
            category = self._categorize_query(query.lower())
            return self._get_template_response(category, query)
    
    def _create_response_from_chunks(self, query, retrieved_chunks):
        """
        Create a response by intelligently combining retrieved chunks.
        """
        try:
            # Group chunks by relevance
            high_relevance = [c for c in retrieved_chunks if c['similarity_score'] > 0.7]
            medium_relevance = [c for c in retrieved_chunks if 0.4 <= c['similarity_score'] <= 0.7]
            
            response = f"""**🌿 Ayurvedic Knowledge on: "{query}"**

"""
            
            # Use high relevance chunks as primary content
            if high_relevance:
                response += "**Key Information:**\n"
                for i, chunk in enumerate(high_relevance[:3], 1):
                    response += f"{i}. {chunk['text']}\n\n"
            
            # Add medium relevance as additional context
            if medium_relevance and len(high_relevance) < 3:
                response += "**Related Information:**\n"
                remaining_slots = 3 - len(high_relevance)
                for i, chunk in enumerate(medium_relevance[:remaining_slots], 1):
                    response += f"• {chunk['text']}\n\n"
            
            # Add source information
            response += f"""---
**📊 Knowledge Sources**: {len(retrieved_chunks)} relevant sources found
**🔍 Relevance Score**: {retrieved_chunks[0]['similarity_score']:.2f} (highest match)

💡 *This response is compiled from traditional Ayurvedic knowledge sources.*"""
            
            return response
            
        except Exception as e:
            print(f"Error creating response from chunks: {e}")
            return f"Based on Ayurvedic principles related to '{query}', here are some relevant insights from our knowledge base. Please ask a more specific question for detailed guidance."
    
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
        """Template for respiratory health questions."""
        return """In Ayurveda, respiratory health is primarily governed by **Prana Vata** (a sub-type of Vata dosha) and **Kapha dosha**. Cold, cough, and breathing issues often result from imbalanced Kapha and weakened immunity.

**🔶 Ayurvedic Herbs for Cold & Cough**

**1. Tulsi (Holy Basil)**
• **Benefits**: Natural expectorant, antimicrobial, immunity booster
• **Usage**: Chew 5-7 fresh leaves daily or drink tulsi tea 2-3 times
• **Effect**: Clears respiratory passages, reduces cough

**2. Ginger (Adrak)**
• **Benefits**: Warming herb, breaks down mucus, improves circulation
• **Usage**: Fresh ginger tea with honey, or ginger-turmeric milk
• **Effect**: Relieves congestion, soothes throat

**3. Turmeric (Haldi)**
• **Benefits**: Anti-inflammatory, antimicrobial, immunity enhancer
• **Usage**: Golden milk (turmeric + warm milk + honey) before bed
• **Effect**: Reduces inflammation, fights infection

**4. Licorice (Mulethi)**
• **Benefits**: Soothes throat, expectorant, anti-inflammatory
• **Usage**: Chew small piece or make herbal tea
• **Effect**: Calms cough, heals throat irritation

**5. Black Pepper (Kali Mirch)**
• **Benefits**: Breaks down mucus, improves digestion
• **Usage**: Pinch of black pepper powder with honey
• **Effect**: Clears respiratory congestion

**🔷 Ayurvedic Home Remedies**

**1. Steam Inhalation**
• Boil water with eucalyptus leaves or ajwain (carom seeds)
• Inhale steam for 5-10 minutes, 2-3 times daily
• Clears nasal congestion and sinuses

**2. Herbal Kadha (Decoction)**
• **Ingredients**: Ginger, tulsi, black pepper, cinnamon, honey
• **Method**: Boil all ingredients, strain, add honey when warm
• **Frequency**: 2-3 times daily

**3. Throat Gargling**
• Warm salt water or turmeric water
• Gargle 2-3 times daily
• Reduces throat inflammation

**🔸 Lifestyle Recommendations**

**Do's:**
• Drink warm water throughout the day
• Eat warm, cooked foods
• Practice pranayama (breathing exercises)
• Get adequate rest and sleep
• Use humidifier in dry environments

**Don'ts:**
• Avoid cold drinks and ice cream
• Reduce dairy products during congestion
• Avoid exposure to cold wind
• Don't suppress natural urges (cough, sneeze)

**⚖️ Dosha-Specific Approach**

**For Vata-Type Respiratory Issues:**
• Warm, moist treatments
• Oil massage on chest
• Gentle steam inhalation

**For Kapha-Type Congestion:**
• Warming, drying herbs
• Vigorous steam inhalation
• Spicy, light foods

**🧘‍♂️ Yoga & Pranayama**
• **Bhastrika** (Bellows Breath): Strengthens lungs
• **Anulom Vilom** (Alternate Nostril): Balances respiratory system
• **Kapalbhati**: Clears respiratory passages

💡 *For chronic respiratory issues or severe symptoms, consult an Ayurvedic practitioner for personalized treatment.*"""
    
    def _heart_template(self):
        """Template for heart health questions."""
        return """In Ayurveda, heart health is governed by **Sadhaka Pitta** (a sub-type of Pitta dosha) and **Vyana Vata** (circulation). The heart is considered the seat of consciousness (Ojas) and requires special care for optimal cardiovascular function.

**🔶 Key Ayurvedic Herbs for Heart Health**

**1. Arjuna (Terminalia arjuna)**
• **Primary Benefits**: Strengthens heart muscle, improves circulation
• **Usage**: 1-2 grams powder with warm water twice daily
• **Effect**: Reduces cholesterol, supports cardiac function
• **Special**: Most important herb for heart in Ayurveda

**2. Brahmi (Bacopa monnieri)**
• **Benefits**: Reduces stress, calms nervous system
• **Usage**: 500mg powder with milk or ghee
• **Effect**: Lowers blood pressure, reduces anxiety

**3. Ashwagandha (Withania somnifera)**
• **Benefits**: Adaptogenic, reduces stress hormones
• **Usage**: 1-2 grams with warm milk before bed
• **Effect**: Strengthens heart, improves stress tolerance

**4. Guggulu (Commiphora mukul)**
• **Benefits**: Cholesterol management, circulation improvement
• **Usage**: As prescribed by practitioner (potent herb)
• **Effect**: Reduces bad cholesterol, prevents arterial blockage

**5. Pushkarmool (Inula racemosa)**
• **Benefits**: Heart tonic, respiratory support
• **Usage**: Under professional guidance
• **Effect**: Strengthens heart muscle, improves breathing

**🔷 Ayurvedic Heart-Healthy Practices**

**1. Pranayama (Breathing Exercises)**
• **Anulom Vilom**: Balances nervous system
• **Bhramari**: Calms mind, reduces stress
• **Sheetali**: Cooling breath, reduces Pitta

**2. Meditation & Yoga**
• **Shavasana**: Deep relaxation for heart
• **Gentle Asanas**: Avoid strenuous poses
• **Regular Practice**: 20-30 minutes daily

**3. Oil Massage (Abhyanga)**
• **Sesame Oil**: For Vata constitution
• **Coconut Oil**: For Pitta constitution
• **Mustard Oil**: For Kapha constitution
• **Frequency**: 2-3 times per week

**🔸 Heart-Healthy Diet (Ayurvedic)**

**Foods to Include:**
• **Whole Grains**: Oats, quinoa, brown rice
• **Fresh Fruits**: Pomegranate, grapes, apples
• **Vegetables**: Leafy greens, beetroot, carrots
• **Healthy Fats**: Ghee (in moderation), nuts, seeds
• **Spices**: Turmeric, coriander, fennel

**Foods to Avoid:**
• **Excessive Salt**: Increases blood pressure
• **Fried Foods**: Clogs arteries, increases Kapha
• **Processed Foods**: High in preservatives
• **Excessive Sweets**: Increases Kapha, weight gain
• **Cold Drinks**: Weakens digestive fire

**⚖️ Dosha-Specific Heart Care**

**For Vata Heart Issues (Irregular heartbeat, anxiety):**
• Warm, nourishing foods
• Regular meal times
• Calming herbs like Brahmi
• Oil massage and warm baths

**For Pitta Heart Issues (High BP, anger, stress):**
• Cooling foods and herbs
• Avoid spicy, sour foods
• Practice cooling pranayama
• Meditation and stress management

**For Kapha Heart Issues (High cholesterol, weight):**
• Light, warm foods
• Regular exercise
• Stimulating herbs
• Avoid heavy, oily foods

**🧘‍♂️ Lifestyle Recommendations**

**Daily Routine:**
• Wake up early (before sunrise)
• Light exercise or yoga
• Regular meal times
• Early dinner (before 7 PM)
• Early sleep (by 10 PM)

**Stress Management:**
• Regular meditation
• Adequate sleep (7-8 hours)
• Avoid overwork
• Maintain work-life balance
• Practice gratitude

**⚠️ Important Notes**
• Heart conditions require professional medical care
• Ayurvedic herbs should complement, not replace, medical treatment
• Consult both Ayurvedic practitioner and cardiologist
• Monitor blood pressure and cholesterol regularly

💡 *Heart health in Ayurveda emphasizes prevention through lifestyle, diet, and stress management rather than just treating symptoms.*"""
    
    def _digestive_template(self):
        """Template for digestive health questions."""
        return """In Ayurveda, digestion is governed by **Agni** (digestive fire), primarily controlled by **Samana Vata** and **Pachaka Pitta**. Strong digestion is the foundation of good health, while weak digestion leads to toxin accumulation (Ama).

**🔶 Key Digestive Herbs in Ayurveda**

**1. Ginger (Adrak)**
• **Benefits**: Kindles digestive fire, reduces gas and bloating
• **Usage**: Fresh ginger slice with rock salt before meals
• **Effect**: Stimulates appetite, improves digestion

**2. Cumin (Jeera)**
• **Benefits**: Carminative, reduces gas, improves absorption
• **Usage**: Cumin water (boil 1 tsp cumin in water) or cumin powder
• **Effect**: Soothes stomach, prevents bloating

**3. Fennel (Saunf)**
• **Benefits**: Cooling digestive, reduces acidity
• **Usage**: Chew 1 tsp after meals or fennel tea
• **Effect**: Freshens breath, aids digestion

**4. Ajwain (Carom Seeds)**
• **Benefits**: Strong digestive stimulant, anti-spasmodic
• **Usage**: 1/2 tsp with warm water for gas/bloating
• **Effect**: Quick relief from digestive discomfort

**5. Triphala**
• **Benefits**: Gentle laxative, digestive tonic, detoxifier
• **Usage**: 1-2 tsp powder with warm water before bed
• **Effect**: Regulates bowel movements, cleanses system

**🔷 Common Digestive Issues & Remedies**

**1. Acidity/Heartburn (Pitta Imbalance)**
• **Herbs**: Amla, licorice, fennel, coriander
• **Diet**: Avoid spicy, sour, fried foods
• **Remedy**: Coconut water, cucumber juice, mint tea

**2. Gas/Bloating (Vata Imbalance)**
• **Herbs**: Ginger, ajwain, hing (asafoetida)
• **Diet**: Warm, cooked foods; avoid raw, cold foods
• **Remedy**: Ginger-ajwain tea, warm oil massage on abdomen

**3. Constipation (Vata Imbalance)**
• **Herbs**: Triphala, isabgol (psyllium), castor oil
• **Diet**: Increase fiber, healthy fats, warm water
• **Remedy**: Triphala at night, morning warm water with lemon

**4. Loose Motions (Pitta/Kapha Imbalance)**
• **Herbs**: Kutaj, bilva, pomegranate peel
• **Diet**: Light, easily digestible foods
• **Remedy**: Buttermilk with cumin, rice water

**🔸 Ayurvedic Digestive Principles**

**Agni (Digestive Fire) Types:**
• **Sama Agni**: Balanced digestion (ideal)
• **Vishama Agni**: Irregular digestion (Vata type)
• **Tikshna Agni**: Sharp digestion (Pitta type)
• **Manda Agni**: Slow digestion (Kapha type)

**🔹 Digestive Guidelines**

**Before Meals:**
• Drink warm water 30 minutes before eating
• Take ginger with rock salt to kindle Agni
• Avoid cold drinks that weaken digestive fire

**During Meals:**
• Eat in calm, peaceful environment
• Chew food thoroughly
• Eat until 3/4 full, leave 1/4 for digestion
• Sip warm water, avoid cold drinks

**After Meals:**
• Walk 100 steps to aid digestion
• Sit in Vajrasana (thunderbolt pose) for 5-10 minutes
• Chew fennel seeds or drink fennel tea
• Rest for 15-20 minutes before activity

**⚖️ Dosha-Specific Digestive Care**

**For Vata Digestion (Irregular, gas, bloating):**
• Regular meal times
• Warm, moist, well-cooked foods
• Digestive spices: ginger, cumin, ajwain
• Oil massage on abdomen

**For Pitta Digestion (Acidity, heartburn, loose stools):**
• Cool, fresh foods
• Avoid spicy, sour, fried foods
• Cooling herbs: fennel, coriander, mint
• Eat at regular times, don't skip meals

**For Kapha Digestion (Slow, heavy feeling, poor appetite):**
• Light, warm, spicy foods
• Stimulating spices: ginger, black pepper, turmeric
• Avoid heavy, oily, cold foods
• Exercise before meals to stimulate appetite

**🧘‍♂️ Lifestyle for Healthy Digestion**

**Daily Routine:**
• Eat largest meal at midday (when Agni is strongest)
• Light breakfast and dinner
• 3-4 hour gap between meals
• Early dinner (before sunset if possible)

**Yoga & Pranayama:**
• **Pawanmuktasana**: Releases gas
• **Bhujangasana**: Stimulates digestive organs
• **Kapalabhati**: Strengthens digestive fire
• **Vajrasana**: Practice after meals

**Foods to Favor:**
• Freshly cooked, warm foods
• Seasonal fruits and vegetables
• Whole grains, legumes (well-cooked)
• Digestive spices and herbs

**Foods to Avoid:**
• Processed, packaged foods
• Cold, frozen foods and drinks
• Overeating or eating too fast
• Incompatible food combinations

💡 *Remember: In Ayurveda, proper digestion is more important than what you eat. Focus on strengthening your Agni for optimal health.*"""
    
    def _doshas_template(self):
        """Template for dosha-related questions."""
        return """In Ayurveda, the **three doshas** are the fundamental bio-energies that govern all physiological and psychological functions in the body. Understanding your dosha constitution is key to maintaining optimal health.

**🔶 The Three Doshas**

**1. VATA DOSHA (Air + Space)**
• **Primary Functions**: Movement, circulation, breathing, nervous system
• **Physical Characteristics**: Thin build, dry skin, cold hands/feet, variable appetite
• **Mental Qualities**: Creative, quick thinking, enthusiastic, but prone to anxiety
• **When Balanced**: Good circulation, regular elimination, sound sleep, creativity
• **When Imbalanced**: Anxiety, insomnia, constipation, dry skin, joint pain

**2. PITTA DOSHA (Fire + Water)**
• **Primary Functions**: Digestion, metabolism, body temperature, intelligence
• **Physical Characteristics**: Medium build, warm body, strong appetite, sharp features
• **Mental Qualities**: Intelligent, focused, ambitious, natural leaders
• **When Balanced**: Strong digestion, good metabolism, sharp intellect, courage
• **When Imbalanced**: Anger, irritability, acidity, skin rashes, inflammation

**3. KAPHA DOSHA (Earth + Water)**
• **Primary Functions**: Structure, stability, immunity, lubrication
• **Physical Characteristics**: Sturdy build, soft skin, slow metabolism, strong stamina
• **Mental Qualities**: Calm, patient, loving, stable, but can be lethargic
• **When Balanced**: Strong immunity, stable emotions, good strength, healthy weight
• **When Imbalanced**: Weight gain, congestion, lethargy, depression, attachment

**🔷 Understanding Your Constitution (Prakriti)**

**Individual Dosha Combinations:**
• **Single Dosha**: One dosha predominates (rare)
• **Dual Dosha**: Two doshas are prominent (most common)
• **Tri-Dosha**: All three doshas are balanced (rare)

**🔸 Balancing Your Doshas**

**For Vata Imbalance:**
• Warm, cooked foods; regular meals
• Oil massage, warm baths
• Regular sleep schedule
• Gentle, grounding exercises like yoga

**For Pitta Imbalance:**
• Cool, fresh foods; avoid spicy/sour
• Cooling activities, avoid excessive heat
• Moderate exercise, swimming
• Meditation to calm the mind

**For Kapha Imbalance:**
• Light, warm, spicy foods
• Regular vigorous exercise
• Stimulating activities
• Avoid heavy, oily foods

**⚖️ Key Principles**
• **Like increases like**: Similar qualities aggravate a dosha
• **Opposites balance**: Opposite qualities pacify a dosha
• **Individual approach**: Treatment varies based on personal constitution

**🧘‍♂️ Practical Application**
• Eat according to your dosha type
• Follow seasonal routines (Ritucharya)
• Practice appropriate exercise for your constitution
• Use herbs and treatments specific to your needs

💡 *Understanding your unique dosha combination helps you make lifestyle choices that support your natural constitution and maintain optimal health.*"""
    
    def _panchakarma_template(self):
        """Template for Panchakarma questions."""
        return """**Panchakarma** is Ayurveda's premier detoxification and rejuvenation therapy, literally meaning "five actions." It's a comprehensive cleansing process that removes deep-seated toxins (Ama) and restores natural balance to the body and mind.

**🔶 The Five Panchakarma Procedures**

**1. Vamana (Therapeutic Vomiting)**
• **Purpose**: Eliminates excess Kapha dosha from chest and stomach
• **Conditions**: Asthma, bronchitis, chronic cough, obesity
• **Process**: Induced vomiting using herbal preparations
• **Duration**: Single session with preparation and follow-up

**2. Virechana (Purgation Therapy)**
• **Purpose**: Cleanses Pitta dosha from liver, gallbladder, small intestine
• **Conditions**: Skin diseases, liver disorders, chronic fever, diabetes
• **Process**: Controlled purgation using herbal laxatives
• **Duration**: 1-3 days with preparation phase

**3. Basti (Medicated Enemas)**
• **Purpose**: Balances Vata dosha, cleanses colon
• **Types**: Niruha Basti (cleansing) and Anuvasana Basti (nourishing)
• **Conditions**: Arthritis, paralysis, constipation, neurological disorders
• **Process**: Herbal decoctions or oils administered rectally

**4. Nasya (Nasal Administration)**
• **Purpose**: Cleanses head, neck, and respiratory passages
• **Conditions**: Sinusitis, headaches, hair loss, mental disorders
• **Process**: Medicated oils or powders through nasal passages
• **Duration**: 7-14 days typically

**5. Raktamokshana (Bloodletting)**
• **Purpose**: Purifies blood, removes Pitta-related toxins
• **Methods**: Leeches, cupping, or controlled blood donation
• **Conditions**: Skin diseases, hypertension, blood disorders
• **Note**: Rarely practiced, requires expert supervision

**🔷 Panchakarma Process Phases**

**1. Purva Karma (Preparation Phase)**
• **Snehana (Oleation)**: Internal and external oil therapy
• **Swedana (Sudation)**: Steam therapy to loosen toxins
• **Duration**: 3-7 days
• **Purpose**: Prepares body for main detox procedures

**2. Pradhana Karma (Main Procedures)**
• **Implementation**: The five main cleansing procedures
• **Duration**: Varies by procedure and individual needs
• **Supervision**: Requires qualified Ayurvedic physician

**3. Paschat Karma (Post-Treatment)**
• **Samsarjana Krama**: Gradual return to normal diet
• **Rasayana Therapy**: Rejuvenation and strengthening
• **Duration**: 7-14 days
• **Purpose**: Rebuilds strength and immunity

**🔸 Benefits of Panchakarma**

**Physical Benefits:**
• Deep detoxification at cellular level
• Improved digestion and metabolism
• Enhanced immunity and vitality
• Better sleep and energy levels
• Reduced inflammation and pain

**Mental Benefits:**
• Stress reduction and mental clarity
• Emotional balance and stability
• Improved concentration and memory
• Spiritual awareness and inner peace

**🔹 Who Should Consider Panchakarma**

**Ideal Candidates:**
• Chronic health conditions
• High stress and lifestyle disorders
• Preventive health maintenance
• Seasonal detoxification
• Pre-conception cleansing

**Contraindications:**
• Pregnancy and menstruation
• Severe weakness or debility
• Active infections or fever
• Heart conditions (without supervision)
• Children under 12 and elderly over 70

**⚖️ Seasonal Panchakarma**

**Spring (Vasant Ritu)**: Vamana for Kapha disorders
**Summer (Grishma Ritu)**: Virechana for Pitta conditions
**Monsoon (Varsha Ritu)**: Basti for Vata imbalances
**Autumn (Sharad Ritu)**: Raktamokshana if needed
**Winter (Shishir Ritu)**: Nourishing therapies

**🧘‍♂️ Preparation Guidelines**

**Before Panchakarma:**
• Consult qualified Ayurvedic physician
• Complete health assessment and dosha analysis
• Gradual dietary modifications
• Mental preparation and positive mindset

**During Treatment:**
• Follow practitioner's instructions strictly
• Maintain light, easily digestible diet
• Avoid physical and mental stress
• Practice meditation and gentle yoga

**After Treatment:**
• Gradual return to normal activities
• Follow prescribed diet and lifestyle
• Take recommended herbal supplements
• Regular follow-up consultations

**⚠️ Important Considerations**
• Panchakarma should only be performed by qualified practitioners
• Requires proper assessment of individual constitution
• Not a quick fix but a comprehensive healing process
• Results may take weeks to months to fully manifest

💡 *Panchakarma is not just detoxification but a complete reset for body, mind, and spirit. It's best done during seasonal transitions for optimal results.*"""
    
    def _herbs_template(self):
        """Template for herbs and medicine questions."""
        return """Ayurveda utilizes thousands of medicinal herbs, each with specific properties and therapeutic actions. Here are some of the most important and commonly used Ayurvedic herbs for various health conditions.

**🔶 Top Ayurvedic Herbs & Their Uses**

**1. Ashwagandha (Withania somnifera)**
• **Properties**: Adaptogenic, rejuvenative, nervine tonic
• **Uses**: Stress, anxiety, insomnia, weakness, immunity
• **Dosage**: 1-3 grams powder with milk or water
• **Benefits**: Reduces cortisol, improves strength and vitality

**2. Turmeric (Curcuma longa)**
• **Properties**: Anti-inflammatory, antimicrobial, hepatoprotective
• **Uses**: Inflammation, wounds, liver health, skin conditions
• **Dosage**: 1-2 grams powder with warm milk or water
• **Benefits**: Powerful antioxidant, natural antibiotic

**3. Brahmi (Bacopa monnieri)**
• **Properties**: Medhya rasayana (brain tonic), nervine
• **Uses**: Memory, concentration, anxiety, mental fatigue
• **Dosage**: 500mg-1g powder with ghee or milk
• **Benefits**: Enhances cognitive function, calms mind

**4. Triphala (Three Fruits)**
• **Composition**: Amla, Bibhitaki, Haritaki
• **Properties**: Digestive, detoxifying, rejuvenative
• **Uses**: Constipation, digestion, eye health, immunity
• **Dosage**: 1-2 tsp powder with warm water before bed
• **Benefits**: Gentle laxative, complete body cleanser

**5. Amla (Emblica officinalis)**
• **Properties**: Highest natural source of Vitamin C, rasayana
• **Uses**: Immunity, hair health, digestion, anti-aging
• **Dosage**: 1-2 tsp powder or fresh juice daily
• **Benefits**: Powerful antioxidant, rejuvenative

**🔷 Herbs by Health Categories**

**For Immunity & Strength:**
• **Giloy (Guduchi)**: Fever, immunity, liver health
• **Shatavari**: Women's health, reproductive system
• **Bala**: Strength, muscle building, nervous system
• **Amalaki**: Vitamin C, antioxidant, anti-aging

**For Digestive Health:**
• **Ginger**: Digestive fire, nausea, circulation
• **Cumin**: Gas, bloating, appetite
• **Fennel**: Acidity, digestion, breath freshener
• **Ajwain**: Stomach pain, gas, respiratory issues

**For Mental Health & Stress:**
• **Jatamansi**: Anxiety, insomnia, mental stress
• **Shankhpushpi**: Memory, concentration, brain tonic
• **Mandukaparni**: Mental clarity, nervous system
• **Saraswatarishta**: Intelligence, speech, memory

**For Respiratory Health:**
• **Vasaka**: Cough, bronchitis, respiratory congestion
• **Kantakari**: Asthma, cough, throat problems
• **Bharangi**: Chronic cough, breathing difficulties
• **Pushkarmool**: Heart, lungs, circulation

**🔸 Herbal Preparations & Forms**

**1. Churna (Powders)**
• **Usage**: Mixed with water, milk, honey, or ghee
• **Examples**: Triphala churna, Sitopaladi churna
• **Benefits**: Easy absorption, customizable dosage

**2. Kwatha (Decoctions)**
• **Usage**: Boiled herbal extracts, taken warm
• **Examples**: Dashmoola kwatha, Saraswatarishta
• **Benefits**: Concentrated potency, quick action

**3. Ghrita (Medicated Ghee)**
• **Usage**: Herbal ghee preparations
• **Examples**: Brahmi ghrita, Saraswata ghrita
• **Benefits**: Nourishing, good for Vata conditions

**4. Asava & Arishta (Fermented Preparations)**
• **Usage**: Self-generated alcohol-based extracts
• **Examples**: Dashamoolarishta, Saraswatarishta
• **Benefits**: Long shelf life, enhanced bioavailability

**🔹 Herb Selection by Dosha**

**For Vata Dosha:**
• **Herbs**: Ashwagandha, Bala, Shatavari, Brahmi
• **Properties**: Nourishing, grounding, calming
• **Preparations**: With ghee, milk, or warm water

**For Pitta Dosha:**
• **Herbs**: Amla, Shatavari, Brahmi, Guduchi
• **Properties**: Cooling, bitter, sweet
• **Preparations**: With cool water, milk, or coconut water

**For Kapha Dosha:**
• **Herbs**: Ginger, Turmeric, Guggulu, Trikatu
• **Properties**: Warming, stimulating, light
• **Preparations**: With honey, warm water, or ginger

**⚖️ Important Guidelines for Herb Usage**

**Dosage Principles:**
• Start with small doses and gradually increase
• Take herbs at appropriate times (empty stomach, with food, etc.)
• Consider individual constitution and condition
• Adjust dosage based on season and age

**Quality Considerations:**
• Use authentic, high-quality herbs from reputable sources
• Check for proper storage and expiration dates
• Prefer organic and sustainably sourced herbs
• Avoid adulterated or contaminated products

**Safety Precautions:**
• Consult qualified Ayurvedic practitioner before starting
• Inform about existing medications and health conditions
• Monitor for any adverse reactions or side effects
• Pregnant and nursing women should exercise extra caution

**🧘‍♂️ Herbal Lifestyle Integration**

**Daily Routine:**
• Morning herbs: Energizing and digestive herbs
• Evening herbs: Calming and nourishing herbs
• Seasonal adjustments: Change herbs based on season
• Consistency: Regular use for best results

**Combination Principles:**
• Synergistic herbs enhance each other's effects
• Avoid incompatible combinations
• Use Anupana (vehicles) like honey, ghee, milk appropriately
• Consider timing and food interactions

**⚠️ Professional Guidance**
• Complex health conditions require expert consultation
• Herb-drug interactions need professional assessment
• Personalized formulations are more effective
• Regular monitoring ensures safety and efficacy

💡 *Remember: Ayurvedic herbs work best when used as part of a holistic lifestyle approach including proper diet, exercise, and stress management.*"""
