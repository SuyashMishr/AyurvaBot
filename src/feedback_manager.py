"""
Feedback Manager module for Enhanced AyurvaBot
Handles user feedback collection and storage.
"""

import json
import os
from datetime import datetime

class FeedbackManager:
    """
    Manages user feedback collection and storage.
    """
    
    def __init__(self, feedback_file="output/user_feedback.json"):
        self.feedback_file = feedback_file
        # Ensure output directory exists
        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
    
    def save_feedback(self, question, answer, rating, feedback_text=""):
        """
        Save user feedback to a JSON file.
        
        Args:
            question (str): The user's question
            answer (str): The bot's response
            rating (int): User rating (1-5)
            feedback_text (str): Optional additional feedback
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load existing feedback
            feedback_data = self._load_existing_feedback()
            
            # Create new feedback entry
            new_feedback = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                "rating": rating,
                "feedback_text": feedback_text,
                "session_id": self._generate_session_id()
            }
            
            # Add to feedback data
            feedback_data.append(new_feedback)
            
            # Save updated feedback
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Feedback saved: Rating {rating}/5 for question: {question[:50]}...")
            return True
            
        except Exception as e:
            print(f"❌ Error saving feedback: {e}")
            return False
    
    def _load_existing_feedback(self):
        """
        Load existing feedback from file.
        
        Returns:
            list: Existing feedback data or empty list
        """
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return []
        except Exception as e:
            print(f"❌ Error loading existing feedback: {e}")
            return []
    
    def _generate_session_id(self):
        """
        Generate a simple session ID based on timestamp.
        
        Returns:
            str: Session ID
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_feedback_stats(self):
        """
        Get basic statistics about collected feedback.
        
        Returns:
            dict: Feedback statistics
        """
        try:
            feedback_data = self._load_existing_feedback()
            
            if not feedback_data:
                return {"total_feedback": 0}
            
            ratings = [entry["rating"] for entry in feedback_data]
            
            stats = {
                "total_feedback": len(feedback_data),
                "average_rating": sum(ratings) / len(ratings),
                "rating_distribution": {
                    "5_star": ratings.count(5),
                    "4_star": ratings.count(4),
                    "3_star": ratings.count(3),
                    "2_star": ratings.count(2),
                    "1_star": ratings.count(1)
                },
                "latest_feedback": feedback_data[-1]["timestamp"] if feedback_data else None
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Error calculating feedback stats: {e}")
            return {"error": str(e)}
