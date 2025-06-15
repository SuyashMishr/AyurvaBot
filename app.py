"""
Enhanced AyurvaBot - Main Application
RAG-Powered Ayurvedic Assistant with Gradio Interface

This is the main entry point for the AyurvaBot application.
"""

import gradio as gr
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_system import RAGSystem
from src.response_generator import ResponseGenerator
from src.feedback_manager import FeedbackManager
from src.config import Config

def main():
    """
    Main function to initialize and launch the AyurvaBot application.
    """
    print("🌿 Initializing Enhanced AyurvaBot with RAG Technology...")
    
    # Initialize components
    config = Config()
    rag_system = RAGSystem(config)
    response_generator = ResponseGenerator(rag_system)
    feedback_manager = FeedbackManager()
    
    # Initialize RAG system
    print("Step 1: Loading Ayurveda knowledge base...")
    rag_system.load_knowledge_base()
    
    print("Step 2: Loading Sentence Transformer for embeddings...")
    rag_system.initialize_sentence_transformer()
    
    print("Step 3: Creating FAISS index for vector search...")
    rag_system.initialize_faiss_index()
    
    print("Step 4: Loading Hugging Face QA model...")
    rag_system.initialize_hf_model()
    
    print("✅ RAG system initialization complete!")
    
    # Create Gradio interface
    with gr.Blocks(title="Enhanced AyurvaBot", theme=gr.themes.Soft()) as demo:
        # State variables
        current_question = gr.State("")
        current_answer = gr.State("")
        
        # Header
        gr.Markdown("# 🌿 Enhanced AyurvaBot - RAG-Powered Ayurvedic Assistant")
        gr.Markdown("### 🚀 Powered by RAG Technology + Hugging Face AI + FAISS Vector Search")
        gr.Markdown("Ask questions about Ayurvedic medicine and get intelligent responses using Retrieval-Augmented Generation!")

        with gr.Row():
            with gr.Column(scale=2):
                # Question input
                question_input = gr.Textbox(
                    placeholder="Ask a question about Ayurveda...",
                    label="Your Question",
                    lines=2
                )

                # Ask button
                ask_button = gr.Button("Ask Question", variant="primary", size="lg")

                # Example questions
                gr.Examples(
                    examples=[
                        ["What is Ayurveda?"],
                        ["What are the three doshas?"],
                        ["What is fever?"],
                        ["What is heart medicines?"],
                        ["What herbs are used for digestion?"],
                        ["What is cold and cough treatment?"],
                        ["What are immunity boosters?"],
                        ["What is Panchakarma?"]
                    ],
                    inputs=[question_input]
                )

            with gr.Column(scale=3):
                # Response output
                response_output = gr.Textbox(
                    label="AyurvaBot Response (RAG-Enhanced)",
                    lines=15,
                    max_lines=20,
                    interactive=False
                )

        # Feedback section
        gr.Markdown("## 📝 Rate This Response")
        gr.Markdown("Your feedback helps us improve AyurvaBot!")

        with gr.Row():
            with gr.Column():
                # Rating component
                rating_input = gr.Radio(
                    choices=[1, 2, 3, 4, 5],
                    label="Rate this response (1 = Poor, 5 = Excellent)",
                    value=None
                )

                # Additional feedback text
                feedback_text_input = gr.Textbox(
                    placeholder="Optional: Provide additional feedback to help us improve...",
                    label="Additional Feedback (Optional)",
                    lines=3
                )

                # Submit feedback button
                feedback_button = gr.Button("Submit Feedback", variant="secondary")

                # Feedback status
                feedback_status = gr.Textbox(
                    label="Feedback Status",
                    interactive=False,
                    visible=False
                )

        # Event handlers
        def handle_question(question):
            if not question.strip():
                return "", "", ""
            
            response = response_generator.generate_response(question)
            return response, question, response

        def handle_feedback(question, answer, rating, feedback_text):
            if not question or not answer:
                return "Please ask a question first before submitting feedback."
            
            if rating is None:
                return "Please select a rating before submitting feedback."
            
            success = feedback_manager.save_feedback(question, answer, rating, feedback_text)
            
            if success:
                return f"Thank you for your feedback! You rated this response {rating}/5 stars. Your feedback helps us improve AyurvaBot."
            else:
                return "Sorry, there was an error saving your feedback. Please try again."

        # Connect events
        ask_button.click(
            fn=handle_question,
            inputs=[question_input],
            outputs=[response_output, current_question, current_answer]
        )

        question_input.submit(
            fn=handle_question,
            inputs=[question_input],
            outputs=[response_output, current_question, current_answer]
        )

        feedback_button.click(
            fn=handle_feedback,
            inputs=[current_question, current_answer, rating_input, feedback_text_input],
            outputs=[feedback_status]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[feedback_status]
        )

    # Launch the interface
    print("🚀 Starting Enhanced AyurvaBot...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=config.PORT,
        share=True,
        show_error=True
    )

if __name__ == "__main__":
    main()
