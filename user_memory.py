import os
import json
import logging
import time
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class UserMemory:
    """
    Manages user memory and preferences for personalized recommendations.
    
    This class handles storing, retrieving, and applying user feedback to improve
    recommendation quality over time.
    """
    
    def __init__(self, user_id: str):
        """
        Initialize the user memory system.
        
        Args:
            user_id (str): Unique identifier for the user
        """
        self.user_id = user_id
        self.memory_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data")
        self.memory_file = os.path.join(self.memory_dir, f"{user_id}_memory.json")
        self.memory_cache = None
        self.last_loaded = 0
        
        # Create memory directory if it doesn't exist
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Load existing memory or create a new one
        self._load_memory()
    
    def _load_memory(self) -> None:
        """Load user memory from disk or initialize a new one if not found."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    self.memory_cache = json.load(f)
                self.last_loaded = time.time()
                logger.info(f"Loaded memory for user {self.user_id}")
            else:
                # Initialize new memory structure
                self.memory_cache = {
                    "user_id": self.user_id,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "company_preferences": {
                        "liked": {},
                        "disliked": {},
                        "neutral": {}
                    },
                    "feature_preferences": {
                        "industries": {
                            "preferred": [],
                            "avoided": []
                        },
                        "company_sizes": {
                            "preferred": [],
                            "avoided": []
                        },
                        "keywords": {
                            "preferred": [],
                            "avoided": []
                        }
                    },
                    "feedback_history": [],
                    "recommendation_history": []
                }
                self._save_memory()
                logger.info(f"Created new memory for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error loading memory for user {self.user_id}: {str(e)}")
            raise
    
    def _save_memory(self) -> None:
        """Save user memory to disk."""
        try:
            # Update the last modified timestamp
            self.memory_cache["updated_at"] = datetime.now().isoformat()
            
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory_cache, f, indent=2)
            
            logger.info(f"Saved memory for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error saving memory for user {self.user_id}: {str(e)}")
            raise
    
    def refresh_memory(self) -> None:
        """Reload memory from disk if it might have changed."""
        if time.time() - self.last_loaded > 60:  # Refresh if older than 60 seconds
            self._load_memory()
    
    def store_company_preference(self, company_name: str, preference: str, reason: Optional[str] = None) -> None:
        """
        Store user preference for a specific company.
        
        Args:
            company_name (str): Name of the company
            preference (str): One of 'liked', 'disliked', or 'neutral'
            reason (str, optional): Reason for the preference
        """
        self.refresh_memory()
        
        # Validate preference
        if preference not in ['liked', 'disliked', 'neutral']:
            raise ValueError("Preference must be one of 'liked', 'disliked', or 'neutral'")
        
        # Remove from other preference categories if present
        for category in ['liked', 'disliked', 'neutral']:
            if category != preference and company_name in self.memory_cache["company_preferences"][category]:
                del self.memory_cache["company_preferences"][category][company_name]
        
        # Add to the specified preference category
        self.memory_cache["company_preferences"][preference][company_name] = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason
        }
        
        # Add to feedback history
        self.memory_cache["feedback_history"].append({
            "timestamp": datetime.now().isoformat(),
            "company": company_name,
            "action": f"marked_as_{preference}",
            "reason": reason
        })
        
        self._save_memory()
        logger.info(f"Stored {preference} preference for company {company_name}")
    
    def get_company_preference(self, company_name: str) -> Dict:
        """
        Get user preference for a specific company.
        
        Args:
            company_name (str): Name of the company
            
        Returns:
            Dict: Preference information or None if not found
        """
        self.refresh_memory()
        
        for category in ['liked', 'disliked', 'neutral']:
            if company_name in self.memory_cache["company_preferences"][category]:
                return {
                    "preference": category,
                    "timestamp": self.memory_cache["company_preferences"][category][company_name]["timestamp"],
                    "reason": self.memory_cache["company_preferences"][category][company_name].get("reason")
                }
        
        return None
    
    def store_feature_preference(self, feature_type: str, feature_value: str, preference: str) -> None:
        """
        Store user preference for a specific feature (industry, company size, keyword).
        
        Args:
            feature_type (str): Type of feature ('industries', 'company_sizes', 'keywords')
            feature_value (str): Value of the feature
            preference (str): One of 'preferred' or 'avoided'
        """
        self.refresh_memory()
        
        # Validate feature type
        if feature_type not in ['industries', 'company_sizes', 'keywords']:
            raise ValueError("Feature type must be one of 'industries', 'company_sizes', 'keywords'")
        
        # Validate preference
        if preference not in ['preferred', 'avoided']:
            raise ValueError("Preference must be one of 'preferred' or 'avoided'")
        
        # Remove from opposite preference if present
        opposite = 'avoided' if preference == 'preferred' else 'preferred'
        if feature_value in self.memory_cache["feature_preferences"][feature_type][opposite]:
            self.memory_cache["feature_preferences"][feature_type][opposite].remove(feature_value)
        
        # Add to specified preference if not already present
        if feature_value not in self.memory_cache["feature_preferences"][feature_type][preference]:
            self.memory_cache["feature_preferences"][feature_type][preference].append(feature_value)
        
        # Add to feedback history
        self.memory_cache["feedback_history"].append({
            "timestamp": datetime.now().isoformat(),
            "feature_type": feature_type,
            "feature_value": feature_value,
            "action": f"marked_as_{preference}"
        })
        
        self._save_memory()
        logger.info(f"Stored {preference} preference for {feature_type} feature {feature_value}")
    
    def store_recommendation_feedback(self, recommendation: Dict, feedback: Dict) -> None:
        """
        Store feedback about a specific recommendation.
        
        Args:
            recommendation (Dict): The recommendation that was shown to the user
            feedback (Dict): User feedback about the recommendation
        """
        self.refresh_memory()
        
        # Add to recommendation history
        entry = {
            "timestamp": datetime.now().isoformat(),
            "recommendation": recommendation,
            "feedback": feedback
        }
        
        self.memory_cache["recommendation_history"].append(entry)
        
        # If feedback includes a preference, store it
        if "preference" in feedback and feedback["preference"] in ['liked', 'disliked', 'neutral']:
            self.store_company_preference(
                recommendation["name"],
                feedback["preference"],
                feedback.get("reason")
            )
        
        self._save_memory()
        logger.info(f"Stored feedback for recommendation of company {recommendation['name']}")
    
    def apply_preferences_to_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """
        Apply user preferences to a list of recommendations.
        
        This function:
        1. Removes disliked companies
        2. Prioritizes companies in preferred industries/sizes
        3. Adjusts fit scores based on user preferences
        
        Args:
            recommendations (List[Dict]): List of company recommendations
            
        Returns:
            List[Dict]: Modified recommendations with user preferences applied
        """
        self.refresh_memory()
        
        filtered_recommendations = []
        
        for rec in recommendations:
            company_name = rec["name"]
            
            # Skip disliked companies
            preference = self.get_company_preference(company_name)
            if preference and preference["preference"] == "disliked":
                logger.info(f"Filtering out disliked company {company_name}")
                continue
            
            # Apply preference adjustments
            if preference:
                # Add preference information to the recommendation
                rec["user_preference"] = preference
                
                # Boost score for liked companies
                if preference["preference"] == "liked":
                    if "fit_score" in rec:
                        for key in rec["fit_score"]:
                            if key != "overall_score":
                                rec["fit_score"][key] = min(100, rec["fit_score"][key] + 15)
                        
                        # Recalculate overall score
                        scores = [v for k, v in rec["fit_score"].items() if k != "overall_score"]
                        rec["fit_score"]["overall_score"] = sum(scores) / len(scores)
            
            filtered_recommendations.append(rec)
        
        # Sort recommendations based on preference-adjusted scores
        filtered_recommendations.sort(
            key=lambda x: (
                1 if self.get_company_preference(x["name"]) and 
                   self.get_company_preference(x["name"])["preference"] == "liked" else 0,
                x.get("fit_score", {}).get("overall_score", 0)
            ),
            reverse=True
        )
        
        return filtered_recommendations
    
    def extract_preferences_from_feedback(self, feedback: str) -> Dict:
        """
        Extract structured preferences from natural language feedback.
        
        Args:
            feedback (str): Natural language feedback from the user
            
        Returns:
            Dict: Structured preferences extracted from the feedback
        """
        # This would ideally use NLP/LLM to extract structured preferences
        # For now, we'll implement a simple keyword-based approach
        
        feedback_lower = feedback.lower()
        result = {
            "preference": None,
            "reason": None,
            "features": {
                "industries": [],
                "company_sizes": [],
                "keywords": []
            }
        }
        
        # Detect preference
        if any(term in feedback_lower for term in ["like", "good", "great", "excellent", "perfect"]):
            result["preference"] = "liked"
        elif any(term in feedback_lower for term in ["dislike", "don't like", "bad", "poor", "not good", "irrelevant"]):
            result["preference"] = "disliked"
        else:
            result["preference"] = "neutral"
        
        # Extract reason (simplified approach)
        result["reason"] = feedback
        
        return result
    
    def get_memory_summary(self) -> Dict:
        """
        Get a summary of the user's memory for use in recommendation prompts.
        
        Returns:
            Dict: Summary of user preferences
        """
        self.refresh_memory()
        
        liked_companies = list(self.memory_cache["company_preferences"]["liked"].keys())
        disliked_companies = list(self.memory_cache["company_preferences"]["disliked"].keys())
        
        preferred_industries = self.memory_cache["feature_preferences"]["industries"]["preferred"]
        avoided_industries = self.memory_cache["feature_preferences"]["industries"]["avoided"]
        
        preferred_sizes = self.memory_cache["feature_preferences"]["company_sizes"]["preferred"]
        avoided_sizes = self.memory_cache["feature_preferences"]["company_sizes"]["avoided"]
        
        preferred_keywords = self.memory_cache["feature_preferences"]["keywords"]["preferred"]
        avoided_keywords = self.memory_cache["feature_preferences"]["keywords"]["avoided"]
        
        return {
            "liked_companies": liked_companies,
            "disliked_companies": disliked_companies,
            "preferred_industries": preferred_industries,
            "avoided_industries": avoided_industries,
            "preferred_sizes": preferred_sizes,
            "avoided_sizes": avoided_sizes,
            "preferred_keywords": preferred_keywords,
            "avoided_keywords": avoided_keywords
        }
    
    def get_llm_preference_prompt(self) -> str:
        """
        Generate a prompt section describing user preferences for the LLM.
        
        Returns:
            str: Prompt section describing user preferences
        """
        summary = self.get_memory_summary()
        
        prompt_parts = ["USER PREFERENCES:"]
        
        if summary["liked_companies"]:
            prompt_parts.append(f"Previously liked companies: {', '.join(summary['liked_companies'])}")
        
        if summary["disliked_companies"]:
            prompt_parts.append(f"Previously disliked companies: {', '.join(summary['disliked_companies'])}")
        
        if summary["preferred_industries"]:
            prompt_parts.append(f"Preferred industries: {', '.join(summary['preferred_industries'])}")
        
        if summary["avoided_industries"]:
            prompt_parts.append(f"Avoided industries: {', '.join(summary['avoided_industries'])}")
        
        if summary["preferred_sizes"]:
            prompt_parts.append(f"Preferred company sizes: {', '.join(summary['preferred_sizes'])}")
        
        if summary["preferred_keywords"]:
            prompt_parts.append(f"Preferred keywords/topics: {', '.join(summary['preferred_keywords'])}")
        
        if summary["avoided_keywords"]:
            prompt_parts.append(f"Avoided keywords/topics: {', '.join(summary['avoided_keywords'])}")
        
        # Add recent feedback history (last 3 items)
        if self.memory_cache["feedback_history"]:
            recent_feedback = self.memory_cache["feedback_history"][-3:]
            prompt_parts.append("Recent feedback:")
            for feedback in recent_feedback:
                if "company" in feedback:
                    prompt_parts.append(f"- {feedback['action']} for {feedback['company']}: {feedback.get('reason', 'No reason provided')}")
                elif "feature_type" in feedback:
                    prompt_parts.append(f"- {feedback['action']} for {feedback['feature_type']}: {feedback['feature_value']}")
        
        return "\n".join(prompt_parts)
