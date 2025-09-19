# src/agents/conversation_agent.py

import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import re
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from loguru import logger
from enum import Enum

class ConversationState(Enum):
    """Conversation flow states."""
    GREETING = "greeting"
    PREFERENCE_GATHERING = "preference_gathering"
    RECOMMENDATION = "recommendation"
    ROUTE_PLANNING = "route_planning"
    BOOKING_ASSISTANCE = "booking_assistance"
    FEEDBACK = "feedback"
    FAREWELL = "farewell"

class IntentClassifier:
    """Classify user intent from messages."""
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.intents = [
            "greeting",
            "ask_recommendation",
            "provide_preference",
            "ask_route",
            "ask_details",
            "give_feedback",
            "booking_help",
            "farewell",
            "other"
        ]
        
    def classify(self, message: str, context: str = "") -> Dict[str, Any]:
        """Classify user intent."""
        prompt = f"""
        Classify the user's intent from their message. 
        Context from previous conversation: {context}
        
        User message: {message}
        
        Possible intents: {', '.join(self.intents)}
        
        Also extract any entities mentioned (locations, cuisines, activities, times, etc.)
        
        Return JSON format:
        {{
            "intent": "intent_name",
            "confidence": 0.95,
            "entities": {{
                "locations": [],
                "cuisines": [],
                "activities": [],
                "time_constraints": [],
                "budget": null,
                "group_size": null
            }}
        }}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an intent classifier for a tourism chatbot."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)

class ConversationAgent:
    """Main conversation management agent."""
    
    def __init__(self, 
                 openai_api_key: str,
                 memory_manager,
                 preference_extractor=None):
        
        # Initialize OpenAI
        self.client = OpenAI(api_key=openai_api_key)
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4-turbo-preview",
            temperature=0.7
        )
        
        # Components
        self.memory = memory_manager
        self.intent_classifier = IntentClassifier(self.client)
        self.preference_extractor = preference_extractor
        
        # State management
        self.conversation_states = {}
        
        # Prompts
        self.prompts = self._initialize_prompts()
        
        logger.info("Conversation Agent initialized")
        
    def _initialize_prompts(self) -> Dict[str, PromptTemplate]:
        """Initialize conversation prompts."""
        prompts = {}
        
        # Greeting prompt
        prompts['greeting'] = PromptTemplate(
            input_variables=["user_name", "location", "time_of_day"],
            template="""
            You are a friendly and knowledgeable tourism assistant. 
            Greet the user warmly and ask how you can help them explore {location}.
            
            Time of day: {time_of_day}
            User name (if known): {user_name}
            
            Be concise but warm. Ask an engaging question about their travel preferences.
            """
        )
        
        # Preference gathering prompt
        prompts['preference_gathering'] = PromptTemplate(
            input_variables=["context", "extracted_preferences", "missing_info"],
            template="""
            You are gathering travel preferences from the user.
            
            Conversation context: {context}
            
            Already extracted preferences: {extracted_preferences}
            
            Still need to know: {missing_info}
            
            Ask a natural follow-up question to gather ONE piece of missing information.
            Make it conversational, not like a form. Show enthusiasm about their choices.
            """
        )
        
        # Recommendation prompt
        prompts['recommendation'] = PromptTemplate(
            input_variables=["preferences", "recommendations", "context"],
            template="""
            Based on the user's preferences, present these recommendations in an engaging way.
            
            User preferences: {preferences}
            
            Recommendations to present: {recommendations}
            
            Context: {context}
            
            Present each option with:
            1. Why it matches their preferences
            2. Key highlights
            3. Practical info (distance, time needed, cost indicator)
            
            Be enthusiastic but informative. Use emojis sparingly for visual appeal.
            Ask which option interests them most or if they'd like other suggestions.
            """
        )
        
        # Route planning prompt
        prompts['route_planning'] = PromptTemplate(
            input_variables=["selected_places", "constraints", "route_info"],
            template="""
            Create an engaging itinerary presentation for the user.
            
            Selected places: {selected_places}
            User constraints: {constraints}
            Route information: {route_info}
            
            Present the route as a journey with:
            1. Clear timeline
            2. Travel methods between spots
            3. Suggested time at each location
            4. Tips for the best experience
            5. Alternative options if time permits
            
            Make it feel like an adventure, not just a list.
            """
        )
        
        return prompts
    
    def process_message(self, 
                       session_id: str,
                       user_id: str,
                       message: str,
                       location: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a user message and generate response."""
        
        # Get context
        context = self.memory.get_full_context(session_id, user_id)
        
        # Classify intent
        intent_result = self.intent_classifier.classify(
            message, 
            context.get('conversation', '')
        )
        
        # Get or initialize conversation state
        state = self.conversation_states.get(session_id, ConversationState.GREETING)
        
        # Store message in memory
        self.memory.conversation.add_message(session_id, {
            'role': 'user',
            'content': message,
            'intent': intent_result
        })
        
        # Process based on intent and state
        response = self._handle_conversation_flow(
            session_id,
            user_id,
            intent_result,
            state,
            context,
            location
        )
        
        # Store response in memory
        self.memory.conversation.add_message(session_id, {
            'role': 'assistant',
            'content': response['message']
        })
        
        return response
    
    def _handle_conversation_flow(self,
                                 session_id: str,
                                 user_id: str,
                                 intent: Dict,
                                 state: ConversationState,
                                 context: Dict,
                                 location: Optional[Dict]) -> Dict[str, Any]:
        """Handle conversation flow based on intent and state."""
        
        intent_type = intent.get('intent')
        entities = intent.get('entities', {})
        
        # Update preferences if entities extracted
        if self.preference_extractor and entities:
            self.preference_extractor.update_from_entities(user_id, entities)
        
        # Route to appropriate handler
        if intent_type == 'greeting' or state == ConversationState.GREETING:
            return self._handle_greeting(session_id, user_id, location)
            
        elif intent_type == 'ask_recommendation':
            return self._handle_recommendation_request(
                session_id, user_id, context, location
            )
            
        elif intent_type == 'provide_preference':
            return self._handle_preference_update(
                session_id, user_id, entities, context
            )
            
        elif intent_type == 'ask_route':
            return self._handle_route_request(
                session_id, user_id, context, location
            )
            
        elif intent_type == 'give_feedback':
            return self._handle_feedback(
                session_id, user_id, intent.get('message'), context
            )
            
        else:
            return self._handle_general_query(
                session_id, user_id, intent.get('message'), context
            )
    
    def _handle_greeting(self, 
                        session_id: str,
                        user_id: str,
                        location: Optional[Dict]) -> Dict[str, Any]:
        """Handle greeting interaction."""
        
        # Get user profile if exists
        profile = self.memory.user_profile.get_profile(user_id)
        user_name = profile.get('name', '') if profile else ''
        
        # Determine time of day
        hour = datetime.now().hour
        if hour < 12:
            time_of_day = "morning"
        elif hour < 18:
            time_of_day = "afternoon"
        else:
            time_of_day = "evening"
        
        # Get location name
        location_name = "your area"
        if location:
            location_name = location.get('city', 'your area')
        
        # Generate greeting
        chain = LLMChain(llm=self.llm, prompt=self.prompts['greeting'])
        message = chain.run(
            user_name=user_name,
            location=location_name,
            time_of_day=time_of_day
        )
        
        # Update state
        self.conversation_states[session_id] = ConversationState.PREFERENCE_GATHERING
        
        return {
            'message': message,
            'state': 'greeting',
            'suggestions': [
                "Find restaurants near me",
                "Popular tourist attractions",
                "Plan a day trip",
                "Local hidden gems"
            ]
        }
    
    def _handle_recommendation_request(self,
                                      session_id: str,
                                      user_id: str,
                                      context: Dict,
                                      location: Optional[Dict]) -> Dict[str, Any]:
        """Handle recommendation requests."""
        
        # Get user preferences
        profile = self.memory.user_profile.get_profile(user_id)
        preferences = profile.get('preferences', {}) if profile else {}
        
        # Check if we have enough preferences
        missing_info = self._check_missing_preferences(preferences)
        
        if missing_info:
            # Ask for more information
            chain = LLMChain(llm=self.llm, prompt=self.prompts['preference_gathering'])
            message = chain.run(
                context=context.get('conversation', ''),
                extracted_preferences=json.dumps(preferences),
                missing_info=', '.join(missing_info)
            )
            
            return {
                'message': message,
                'state': 'gathering_preferences',
                'missing_info': missing_info
            }
        
        # Generate recommendations (this would call recommendation_agent)
        recommendations = self._get_recommendations(preferences, location)
        
        # Format response
        chain = LLMChain(llm=self.llm, prompt=self.prompts['recommendation'])
        message = chain.run(
            preferences=json.dumps(preferences),
            recommendations=json.dumps(recommendations),
            context=context.get('conversation', '')
        )
        
        # Update state
        self.conversation_states[session_id] = ConversationState.RECOMMENDATION
        
        return {
            'message': message,
            'state': 'recommendations',
            'recommendations': recommendations,
            'actions': ['Show on map', 'Get directions', 'More details', 'Other options']
        }
    
    def _check_missing_preferences(self, preferences: Dict) -> List[str]:
        """Check what preference information is missing."""
        missing = []
        
        essential_prefs = {
            'activity_type': 'what type of activities you enjoy',
            'budget': 'your budget range',
            'group_size': 'how many people are in your group',
            'duration': 'how much time you have'
        }
        
        for key, description in essential_prefs.items():
            if key not in preferences or preferences[key] is None:
                missing.append(description)
        
        return missing[:2]  # Ask for max 2 at a time
    
    def _get_recommendations(self, preferences: Dict, location: Optional[Dict]) -> List[Dict]:
        """Get recommendations based on preferences."""
        # Placeholder - would integrate with recommendation_agent
        return [
            {
                'name': 'Central Park',
                'type': 'attraction',
                'match_score': 0.92,
                'distance': '1.2 km',
                'estimated_time': '2-3 hours',
                'price': 'Free',
                'highlights': ['Great for walks', 'Photo opportunities', 'Family friendly']
            },
            {
                'name': 'Local Food Market',
                'type': 'restaurant',
                'match_score': 0.88,
                'distance': '800 m',
                'estimated_time': '1-2 hours',
                'price': '$$',
                'highlights': ['Local cuisine', 'Vegetarian options', 'Cultural experience']
            }
        ]
    
    def _handle_preference_update(self,
                                 session_id: str,
                                 user_id: str,
                                 entities: Dict,
                                 context: Dict) -> Dict[str, Any]:
        """Handle preference updates from user."""
        
        # Update preferences
        self.memory.user_profile.update_preferences(user_id, entities)
        
        # Acknowledge and continue conversation
        response = f"Great! I've noted your preferences. "
        
        # Check if we have enough info for recommendations
        profile = self.memory.user_profile.get_profile(user_id)
        preferences = profile.get('preferences', {})
        missing_info = self._check_missing_preferences(preferences)
        
        if missing_info:
            response += f"Could you tell me about {missing_info[0]}?"
        else:
            response += "Based on what you've told me, I have some great suggestions for you!"
            self.conversation_states[session_id] = ConversationState.RECOMMENDATION
        
        return {
            'message': response,
            'state': 'preference_gathering',
            'updated_preferences': entities
        }
    
    def _handle_route_request(self,
                             session_id: str,
                             user_id: str,
                             context: Dict,
                             location: Optional[Dict]) -> Dict[str, Any]:
        """Handle route planning requests."""
        
        # This would integrate with route_planner agent
        route_info = {
            'total_distance': '5.2 km',
            'total_time': '4-5 hours',
            'transport_mode': 'walking + public transit'
        }
        
        selected_places = ["Central Park", "Local Food Market", "Art Museum"]
        constraints = {'time_available': '5 hours', 'transport': 'public'}
        
        # Generate route presentation
        chain = LLMChain(llm=self.llm, prompt=self.prompts['route_planning'])
        message = chain.run(
            selected_places=selected_places,
            constraints=constraints,
            route_info=route_info
        )
        
        self.conversation_states[session_id] = ConversationState.ROUTE_PLANNING
        
        return {
            'message': message,
            'state': 'route_planning',
            'route': route_info,
            'places': selected_places
        }
    
    def _handle_feedback(self,
                        session_id: str,
                        user_id: str,
                        message: str,
                        context: Dict) -> Dict[str, Any]:
        """Handle user feedback."""
        
        # Analyze feedback sentiment
        sentiment = self._analyze_sentiment(message)
        
        response = ""
        if sentiment > 0.5:
            response = "I'm glad you enjoyed that! Your feedback helps me provide better recommendations."
        else:
            response = "I appreciate your feedback. Let me find something that better matches your preferences."
        
        # Store feedback
        self.memory.user_profile.add_feedback(
            user_id,
            session_id,
            sentiment,
            'general'
        )
        
        return {
            'message': response,
            'state': 'feedback',
            'sentiment': sentiment
        }
    
    def _handle_general_query(self,
                             session_id: str,
                             user_id: str,
                             message: str,
                             context: Dict) -> Dict[str, Any]:
        """Handle general queries using LLM."""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful tourism assistant. Provide concise, friendly responses."
                },
                {
                    "role": "user",
                    "content": f"Context: {context.get('conversation', '')}\n\nUser: {message}"
                }
            ],
            temperature=0.7
        )
        
        return {
            'message': response.choices[0].message.content,
            'state': 'conversation'
        }
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of feedback text."""
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "Analyze the sentiment of this feedback. Return a score between 0 (very negative) and 1 (very positive)."
                },
                {"role": "user", "content": text}
            ],
            temperature=0
        )
        
        try:
            score = float(response.choices[0].message.content)
            return max(0, min(1, score))
        except:
            return 0.5
