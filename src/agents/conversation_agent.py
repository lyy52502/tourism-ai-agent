# src/agents/conversation_agent.py - FIXED VERSION

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
        """Process user message and generate response."""
        
        # Get conversation context
        context = self.memory.get_full_context(session_id, user_id)

        # ✅ Step 1: Get user location properly - USE PROVIDED LOCATION FIRST
        user_loc = None
        
        # CRITICAL: If frontend sends location, use it (this is GPS location)
        if location and isinstance(location, dict) and location.get('lat') and location.get('lng'):
            user_loc = Location(
                lat=location.get('lat'),
                lng=location.get('lng'),
                address=location.get('address', f"Lat: {location.get('lat')}, Lng: {location.get('lng')}")
            )
            logger.info(f"✅ Using provided location: {user_loc.lat}, {user_loc.lng}")
        else:
            # Fallback to default Uppsala location
            user_loc = Location(lat=59.8586, lng=17.6389, address="Uppsala, Sweden")
            logger.info(f"📍 No location provided, using default Uppsala: {user_loc.lat}, {user_loc.lng}")

        # ✅ Step 2: Analyze what user is asking for (more intelligent detection)
        message_lower = message.lower()
        
        # Define specific categories
        asking_for_food = any(word in message_lower for word in [
            'restaurant', 'food', 'eat', 'dining', 'lunch', 'dinner', 'breakfast',
            'italian', 'chinese', 'japanese', 'pizza', 'sushi', 'cafe', 'bar'
        ])
        
        asking_for_attractions = any(word in message_lower for word in [
            'attraction', 'visit', 'see', 'sightseeing', 'landmark', 'monument',
            'cathedral', 'museum', 'castle', 'park', 'garden', 'top', 'best'
        ])
        
        asking_for_activities = any(word in message_lower for word in [
            'activity', 'do', 'experience', 'tour', 'fun', 'adventure', 'outdoor'
        ])
        
        asking_for_recommendations = any(word in message_lower for word in [
            'recommend', 'suggest', 'find', 'show', 'where', 'nearby', 'around'
        ])
        
        # Determine if we should get recommendations and what type
        should_recommend = (asking_for_food or asking_for_attractions or 
                           asking_for_activities or asking_for_recommendations)
        
        # Determine primary intent
        if asking_for_food:
            intent = "restaurants"
            logger.info(f"🍴 Detected food/restaurant query")
        elif asking_for_attractions:
            intent = "attractions"
            logger.info(f"🏛️ Detected attraction query")
        elif asking_for_activities:
            intent = "activities"
            logger.info(f"🎯 Detected activity query")
        else:
            intent = "general"
            logger.info(f"💬 General conversation - no specific intent")
        
        logger.info(f"🔍 Should fetch recommendations? {should_recommend} (intent: {intent})")
        
        # ✅ Step 3: Get recommendations ONLY if appropriate
        recommended_places = []
        recommendations_data = []
        
        if should_recommend and self.recommendation_agent and intent != "general":
            try:
                logger.info(f"🎯 Fetching {intent} for: '{message}'")
                
                location_dict = {'lat': user_loc.lat, 'lng': user_loc.lng}
                
                # Create context with intent
                recommendation_context = {
                    'intent': intent,
                    'query': message
                }
                
                # ✅ CRITICAL: Call recommendation agent
                recommendations = self.recommendation_agent.get_recommendations(
                    user_id=user_id,
                    session_id=session_id,
                    location=location_dict,
                    context=recommendation_context,
                    num_recommendations=5
                )
                
                logger.info(f"✅ Got {len(recommendations)} recommendations from agent")
                
                if len(recommendations) == 0:
                    logger.warning(f"⚠️ Recommendation agent returned 0 results!")
                
                # Store recommendations data for response
                recommendations_data = recommendations
                
                # Extract names for LLM context
                recommended_places = [
                    {
                        'name': r.name,
                        'type': r.type,
                        'rating': r.rating,
                        'distance': round(r.distance, 2),
                        'category': r.category,
                        'match_score': round(r.match_score, 2)
                    } 
                    for r in recommendations[:5]
                ]
                
                logger.info(f"📋 Place names for LLM: {[p['name'] for p in recommended_places]}")
                
            except Exception as e:
                logger.error(f"❌ Error getting recommendations: {e}")
                import traceback
                traceback.print_exc()
        elif should_recommend and not self.recommendation_agent:
            logger.error(f"❌ Recommendation query detected but recommendation_agent is None!")
        else:
            logger.info(f"ℹ️ Not a recommendation query, responding conversationally")

        # ✅ Step 4: Generate LLM response with proper context
        system_prompt = """You are a friendly and knowledgeable tourism assistant for Uppsala, Sweden. 

Your role:
- Help users discover restaurants, attractions, and activities
- Answer general questions about Uppsala and tourism
- Be conversational, warm, and helpful
- Keep responses concise (2-3 sentences)
- Don't use lists or bullet points - speak naturally

Important: 
- When specific places are provided, mention them by name
- When no places are provided, give general helpful information
- Match your response to what the user is actually asking about"""

        user_prompt = f"""User message: {message}

Current location: {user_loc.address or f"Lat: {user_loc.lat}, Lng: {user_loc.lng}"}"""

        # Add conversation context if available
        if context.get('conversation'):
            recent_context = context.get('conversation').split('\n')[-4:]  # Last 2 exchanges
            user_prompt += f"\n\nRecent conversation:\n" + "\n".join(recent_context)

        # Add recommendations to prompt ONLY if we have them
        if recommended_places:
            places_text = "\n".join([
                f"{i+1}. {p['name']} ({p['type']}, {p['rating']}⭐, {p['distance']}km away)"
                for i, p in enumerate(recommended_places)
            ])
            
            # Determine what kind of places they are
            place_category = intent if intent != "general" else "places"
            
            user_prompt += f"""

I found these {place_category} nearby:
{places_text}

You MUST mention only places from this list in your reply. 
Do not invent or add new ones.
Write 2–3 friendly sentences that highlight 2–3 of them naturally. 
Example: “If you’re craving sushi, {recommended_places[0]['name']} and {recommended_places[1]['name']} are both great options nearby!”
"""

        else:
            # No recommendations - respond conversationally
            user_prompt += """

No specific recommendations available for this query. Respond helpfully to the user's question about Uppsala or tourism in general. Keep it friendly and informative."""

        try:
            # Call OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=200  # Keep responses short
            )

            final_message = response.choices[0].message.content
            logger.info(f"💬 LLM Response: {final_message[:100]}...")

        except Exception as e:
            logger.error(f"❌ OpenAI API error: {e}")
            
            # Fallback response
            if recommended_places:
                place_names = [p['name'] for p in recommended_places[:3]]
                final_message = f"I found some great {intent} for you: {', '.join(place_names)}! Check the recommendations panel for details."
            else:
                final_message = f"I'm here to help you explore Uppsala! Feel free to ask about restaurants, attractions, or activities."

        # ✅ Save to memory
        self.memory.conversation.add_message(session_id, {"role": "user", "content": message})
        self.memory.conversation.add_message(session_id, {"role": "assistant", "content": final_message})

        # ✅ Return response with recommendations
        result = {
            "message": final_message,
            "state": "conversation",
            "recommendations": recommendations_data,  # Return actual recommendation objects
            "location": {"lat": user_loc.lat, "lng": user_loc.lng, "address": user_loc.address}
        }
        
        logger.info(f"📤 Returning {len(recommendations_data)} recommendations to main.py")
        
        return result
