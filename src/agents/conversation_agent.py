import os
from typing import Dict, Any, Optional
from datetime import datetime
import json
from openai import OpenAI
from loguru import logger
from agents.location_agent import Location, LocationAgent, TransportMode  
from agents.recommendation_agent import RecommendationAgent

class ConversationAgent:
    """Conversation agent with LLM directly leading the conversation."""
    
    def __init__(self, openai_api_key: str, memory_manager, preference_extractor=None, location_agent=None, recommendation_agent=None):
        self.client = OpenAI(api_key=openai_api_key)
        self.memory = memory_manager
        self.preference_extractor = preference_extractor
        self.location_agent = location_agent
        self.recommendation_agent = recommendation_agent
        logger.info("Conversation Agent (LLM-direct) initialized")

    def process_message(self, session_id, user_id, message, location=None, ip_address: Optional[str] = None):
        context = self.memory.get_full_context(session_id, user_id)

        # ✅ Step 1: Get user location properly
        user_loc = None
        if location:
    # Use provided location
            user_loc = Location(
                lat=location.get('lat', 59.8586),
                lng=location.get('lng', 17.6389)
            )
        elif self.location_agent:
            # Use IP if provided
            user_loc = self.location_agent.get_user_location(ip_address=ip_address)
        else:
            # Fallback to default location
            user_loc = Location(
                lat=59.8586,
                lng=17.6389,
                address="Uppsala, Sweden"
            )

        # Step 2: Get recommendations if needed
        recommended_places = []
        if self.recommendation_agent and any(keyword in message.lower() for keyword in 
                                           ['recommend', 'suggest', 'find', 'restaurant', 'food', 'eat', 'place']):
            try:
                location_dict = {'lat': user_loc.lat, 'lng': user_loc.lng}
                recommendations = self.recommendation_agent.get_recommendations(
                    user_id=user_id,
                    session_id=session_id,
                    location=location_dict,
                    num_recommendations=5
                )
                recommended_places = [r.name for r in recommendations[:3]]  # Get top 3 names
            except Exception as e:
                logger.error(f"Error getting recommendations: {e}")
                recommended_places = []

        # Step 3: Send everything to LLM with proper context
        system_prompt = """You are a friendly and knowledgeable tourism assistant for Uppsala, Sweden. 
        Help users find restaurants, attractions, and activities. Be conversational and helpful.
        If the user mentions specific preferences (like cuisine type, budget, etc.), acknowledge them.
        Keep responses concise but informative."""
        
        user_prompt = f"""User message: {message}

Current location: {user_loc.address or f"Lat: {user_loc.lat}, Lng: {user_loc.lng}"}

Previous conversation: {context.get('conversation', 'No previous conversation')}"""

        if recommended_places:
            user_prompt += f"\nAvailable recommendations nearby: {', '.join(recommended_places)}"

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )

            final_message = response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            if "italian" in message.lower():
                final_message = "I'd be happy to help you find Italian restaurants in Uppsala! Let me search for some great options near your location."
            elif "chinese" in message.lower():
                final_message = "Great choice! I'll find the best Chinese restaurants in Uppsala for you."
            else:
                final_message = f"I understand you're looking for: {message}. Let me help you find great places in Uppsala!"

        # Save to memory
        self.memory.conversation.add_message(session_id, {"role": "user", "content": message})
        self.memory.conversation.add_message(session_id, {"role": "assistant", "content": final_message})

        return {"message": final_message, "state": "conversation"}
