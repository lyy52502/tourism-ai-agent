# src/agents/preference_agent.py

import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from openai import OpenAI
from sklearn.preprocessing import StandardScaler
import pickle
from loguru import logger

class PreferenceType(Enum):
    """Types of user preferences."""
    EXPLICIT = "explicit"  # Directly stated
    IMPLICIT = "implicit"  # Inferred from behavior
    CONTEXTUAL = "contextual"  # Based on context (time, weather, etc.)

@dataclass
class UserPreference:
    """User preference structure."""
    category: str
    value: Any
    confidence: float
    type: PreferenceType
    timestamp: datetime
    source: str  # Where preference was extracted from
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['type'] = self.type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

class PreferenceExtractor:
    """Extract and manage user preferences from conversations."""
    
    def __init__(self, openai_client: OpenAI, memory_manager):
        self.client = openai_client
        self.memory = memory_manager
        
        # Preference categories and their extraction patterns
        self.preference_schema = self._init_preference_schema()
        
        # Preference learning model
        self.preference_model = None
        self.scaler = StandardScaler()
        
        logger.info("Preference Extractor initialized")
        
    def _init_preference_schema(self) -> Dict:
        """Initialize preference extraction schema."""
        return {
            'cuisine': {
                'keywords': ['food', 'eat', 'restaurant', 'cuisine', 'meal', 'lunch', 'dinner'],
                'examples': ['Italian', 'Chinese', 'vegetarian', 'seafood', 'local'],
                'questions': ['What type of food do you enjoy?']
            },
            'activities': {
                'keywords': ['do', 'visit', 'see', 'activity', 'attraction', 'fun'],
                'examples': ['museums', 'hiking', 'shopping', 'beaches', 'nightlife', 'cultural'],
                'questions': ['What activities interest you?']
            },
            'budget': {
                'keywords': ['budget', 'spend', 'cost', 'price', 'expensive', 'cheap', 'free'],
                'examples': ['budget-friendly', 'moderate', 'luxury', '$', '$$', '$$$'],
                'questions': ['What\'s your budget range?']
            },
            'pace': {
                'keywords': ['pace', 'rush', 'relaxed', 'busy', 'slow', 'quick'],
                'examples': ['relaxed', 'moderate', 'packed', 'leisurely'],
                'questions': ['Do you prefer a relaxed or packed schedule?']
            },
            'group_composition': {
                'keywords': ['family', 'friends', 'solo', 'couple', 'kids', 'group'],
                'examples': ['family with kids', 'couple', 'solo traveler', 'friend group'],
                'questions': ['Who are you traveling with?']
            },
            'interests': {
                'keywords': ['interested', 'like', 'love', 'prefer', 'enjoy', 'passion'],
                'examples': ['history', 'art', 'nature', 'technology', 'sports', 'music'],
                'questions': ['What are your main interests?']
            },
            'accessibility': {
                'keywords': ['accessible', 'wheelchair', 'mobility', 'stairs', 'walking'],
                'examples': ['wheelchair accessible', 'limited mobility', 'can walk long distances'],
                'questions': ['Do you have any accessibility requirements?']
            },
            'time_constraints': {
                'keywords': ['time', 'hours', 'days', 'schedule', 'duration', 'long'],
                'examples': ['2 hours', 'half day', 'full day', 'multiple days'],
                'questions': ['How much time do you have?']
            }
        }
    
    def extract_preferences(self, 
                           user_id: str,
                           message: str,
                           context: Optional[str] = None) -> List[UserPreference]:
        """Extract preferences from user message."""
        
        # Build extraction prompt
        prompt = self._build_extraction_prompt(message, context)
        
        # Call OpenAI for extraction
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at extracting travel preferences from conversations. Return valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        # Parse extracted preferences
        extracted = json.loads(response.choices[0].message.content)
        
        # Convert to UserPreference objects
        preferences = []
        for pref in extracted.get('preferences', []):
            preference = UserPreference(
                category=pref['category'],
                value=pref['value'],
                confidence=pref.get('confidence', 0.8),
                type=PreferenceType[pref.get('type', 'EXPLICIT').upper()],
                timestamp=datetime.now(),
                source='conversation'
            )
            preferences.append(preference)
            
        # Store in user profile
        self._update_user_preferences(user_id, preferences)
        
        return preferences
    
    def _build_extraction_prompt(self, message: str, context: Optional[str]) -> str:
        """Build prompt for preference extraction."""
        schema_description = json.dumps(
            {cat: info['examples'] for cat, info in self.preference_schema.items()},
            indent=2
        )
        
        prompt = f"""
        Extract user preferences from this message.
        
        Message: {message}
        
        {f"Conversation context: {context}" if context else ""}
        
        Preference categories and examples:
        {schema_description}
        
        For each preference found, provide:
        - category: from the schema above
        - value: the specific preference value
        - confidence: 0.0 to 1.0
        - type: "explicit" if directly stated, "implicit" if inferred
        
        Return JSON:
        {{
            "preferences": [
                {{
                    "category": "cuisine",
                    "value": "Italian",
                    "confidence": 0.95,
                    "type": "explicit"
                }}
            ]
        }}
        """
        
        return prompt
    
    def _update_user_preferences(self, user_id: str, preferences: List[UserPreference]):
        """Update user profile with new preferences."""
        profile = self.memory.user_profile.get_profile(user_id) or {'preferences': {}}
        
        for pref in preferences:
            category = pref.category
            
            # Handle multiple values for same category
            if category not in profile['preferences']:
                profile['preferences'][category] = []
            
            # Check if this value already exists
            existing = False
            for existing_pref in profile['preferences'][category]:
                if existing_pref.get('value') == pref.value:
                    # Update confidence if higher
                    if pref.confidence > existing_pref.get('confidence', 0):
                        existing_pref['confidence'] = pref.confidence
                        existing_pref['timestamp'] = pref.timestamp.isoformat()
                    existing = True
                    break
            
            if not existing:
                profile['preferences'][category].append(pref.to_dict())
        
        # Update profile
        self.memory.user_profile.update_preferences(user_id, profile['preferences'])
        
    def infer_implicit_preferences(self, 
                                  user_id: str,
                                  interactions: List[Dict]) -> List[UserPreference]:
        """Infer preferences from user behavior."""
        inferred = []
        
        # Analyze interaction patterns
        liked_items = [i for i in interactions if i.get('feedback', 0) > 3]
        
        if liked_items:
            # Extract common patterns
            categories = {}
            for item in liked_items:
                cat = item.get('category')
                if cat:
                    categories[cat] = categories.get(cat, 0) + 1
            
            # Create implicit preferences
            for cat, count in categories.items():
                if count >= 2:  # At least 2 positive interactions
                    confidence = min(count / 10, 0.9)  # Cap at 0.9
                    
                    pref = UserPreference(
                        category='interests',
                        value=cat,
                        confidence=confidence,
                        type=PreferenceType.IMPLICIT,
                        timestamp=datetime.now(),
                        source='behavior_analysis'
                    )
                    inferred.append(pref)
        
        return inferred
    
    def update_from_entities(self, user_id: str, entities: Dict):
        """Update preferences from extracted entities."""
        preferences = []
        
        # Map entities to preferences
        mapping = {
            'cuisines': 'cuisine',
            'activities': 'activities',
            'budget': 'budget',
            'group_size': 'group_composition'
        }
        
        for entity_type, pref_category in mapping.items():
            if entity_type in entities and entities[entity_type]:
                values = entities[entity_type]
                if not isinstance(values, list):
                    values = [values]
                    
                for value in values:
                    if value:
                        pref = UserPreference(
                            category=pref_category,
                            value=value,
                            confidence=0.85,
                            type=PreferenceType.EXPLICIT,
                            timestamp=datetime.now(),
                            source='entity_extraction'
                        )
                        preferences.append(pref)
        
        if preferences:
            self._update_user_preferences(user_id, preferences)
    
    def get_preference_vector(self, user_id: str) -> np.ndarray:
        """Convert user preferences to vector for ML models."""
        profile = self.memory.user_profile.get_profile(user_id)
        
        if not profile:
            return np.zeros(50)  # Default vector size
        
        preferences = profile.get('preferences', {})
        
        # Create feature vector
        features = []
        
        # Budget encoding (0-3)
        budget_map = {'$': 1, '$$': 2, '$$$': 3, 'free': 0}
        budget = preferences.get('budget', [])
        if budget:
            budget_val = budget_map.get(budget[0].get('value', '$$'), 2)
        else:
            budget_val = 2
        features.append(budget_val)
        
        # Activity preferences (binary for each type)
        activity_types = ['cultural', 'nature', 'adventure', 'relaxation', 'nightlife', 'shopping']
        user_activities = preferences.get('activities', [])
        user_activity_values = [a.get('value', '').lower() for a in user_activities]
        
        for activity in activity_types:
            features.append(1 if activity in user_activity_values else 0)
        
        # Group size encoding
        group_sizes = {'solo': 1, 'couple': 2, 'small group': 4, 'large group': 8}
        group = preferences.get('group_composition', [])
        if group:
            group_size = group_sizes.get(group[0].get('value', 'couple'), 2)
        else:
            group_size = 2
        features.append(group_size)
        
        # Pace preference (0-2: relaxed, moderate, packed)
        pace_map = {'relaxed': 0, 'moderate': 1, 'packed': 2}
        pace = preferences.get('pace', [])
        if pace:
            pace_val = pace_map.get(pace[0].get('value', 'moderate'), 1)
        else:
            pace_val = 1
        features.append(pace_val)
        
        # Cuisine preferences (binary for common types)
        cuisine_types = ['italian', 'asian', 'american', 'local', 'vegetarian', 'seafood']
        user_cuisines = preferences.get('cuisine', [])
        user_cuisine_values = [c.get('value', '').lower() for c in user_cuisines]
        
        for cuisine in cuisine_types:
            features.append(1 if cuisine in user_cuisine_values else 0)
        
        # Interest areas (weighted by confidence)
        interest_areas = ['history', 'art', 'nature', 'technology', 'sports', 'music']
        user_interests = preferences.get('interests', [])
        
        for interest in interest_areas:
            interest_score = 0
            for ui in user_interests:
                if ui.get('value', '').lower() == interest:
                    interest_score = ui.get('confidence', 0.5)
                    break
            features.append(interest_score)
        
        # Pad or truncate to fixed size
        feature_vector = np.array(features)
        if len(feature_vector) < 50:
            feature_vector = np.pad(feature_vector, (0, 50 - len(feature_vector)))
        else:
            feature_vector = feature_vector[:50]
            
        return feature_vector
    
    def get_preference_summary(self, user_id: str) -> Dict[str, Any]:
        """Get human-readable preference summary."""
        profile = self.memory.user_profile.get_profile(user_id)
        
        if not profile:
            return {"message": "No preferences found"}
        
        preferences = profile.get('preferences', {})
        
        summary = {
            "strong_preferences": [],
            "moderate_preferences": [],
            "contextual_factors": []
        }
        
        for category, prefs in preferences.items():
            for pref in prefs:
                confidence = pref.get('confidence', 0.5)
                value = pref.get('value')
                
                pref_str = f"{category}: {value}"
                
                if confidence > 0.8:
                    summary["strong_preferences"].append(pref_str)
                elif confidence > 0.5:
                    summary["moderate_preferences"].append(pref_str)
                    
                if pref.get('type') == 'contextual':
                    summary["contextual_factors"].append(pref_str)
        
        return summary
    
    def calculate_preference_similarity(self, user_id1: str, user_id2: str) -> float:
        """Calculate similarity between two users' preferences."""
        vec1 = self.get_preference_vector(user_id1)
        vec2 = self.get_preference_vector(user_id2)
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def adapt_preferences_to_context(self, 
                                    user_id: str,
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt preferences based on current context."""
        base_preferences = self.get_preference_summary(user_id)
        
        # Context factors
        time_of_day = context.get('time_of_day', 'afternoon')
        weather = context.get('weather', 'clear')
        day_of_week = context.get('day_of_week', 'weekday')
        season = context.get('season', 'summer')
        
        adapted = base_preferences.copy()
        contextual_adjustments = []
        
        # Time-based adjustments
        if time_of_day == 'morning':
            contextual_adjustments.append("breakfast spots and morning activities prioritized")
        elif time_of_day == 'evening':
            contextual_adjustments.append("dinner venues and evening entertainment prioritized")
        
        # Weather-based adjustments
        if weather in ['rain', 'snow']:
            contextual_adjustments.append("indoor activities prioritized")
        elif weather == 'clear' and season == 'summer':
            contextual_adjustments.append("outdoor activities recommended")
        
        # Day of week adjustments
        if day_of_week in ['saturday', 'sunday']:
            contextual_adjustments.append("popular weekend spots considered")
        
        adapted['contextual_adjustments'] = contextual_adjustments
        
        return adapted
    
    def learn_from_feedback(self, 
                           user_id: str,
                           item_id: str,
                           feedback: float,
                           context: Dict[str, Any]):
        """Update preference model based on feedback."""
        # Store feedback
        self.memory.user_profile.add_feedback(
            user_id,
            item_id,
            feedback,
            'recommendation'
        )
        
        # If negative feedback, reduce confidence in related preferences
        if feedback < 3:
            profile = self.memory.user_profile.get_profile(user_id)
            preferences = profile.get('preferences', {})
            
            # Get item details to understand what was rejected
            item_category = context.get('item_category')
            if item_category and item_category in preferences:
                for pref in preferences[item_category]:
                    if pref.get('value') == context.get('item_value'):
                        # Reduce confidence
                        pref['confidence'] = max(0.3, pref.get('confidence', 0.5) * 0.8)
                        
                self.memory.user_profile.update_preferences(user_id, preferences)
        
        # If positive feedback, increase confidence
        elif feedback >= 4:
            profile = self.memory.user_profile.get_profile(user_id)
            preferences = profile.get('preferences', {})
            
            item_category = context.get('item_category')
            if item_category:
                if item_category not in preferences:
                    preferences[item_category] = []
                    
                # Add or update preference
                found = False
                for pref in preferences[item_category]:
                    if pref.get('value') == context.get('item_value'):
                        pref['confidence'] = min(1.0, pref.get('confidence', 0.5) * 1.2)
                        found = True
                        break
                
                if not found:
                    # Add new implicit preference
                    new_pref = UserPreference(
                        category=item_category,
                        value=context.get('item_value'),
                        confidence=0.7,
                        type=PreferenceType.IMPLICIT,
                        timestamp=datetime.now(),
                        source='positive_feedback'
                    )
                    preferences[item_category].append(new_pref.to_dict())
                
                self.memory.user_profile.update_preferences(user_id, preferences)
        
        logger.info(f"Updated preferences for user {user_id} based on feedback: {feedback}")


class PreferenceLearningModel:
    """ML model for preference prediction and learning."""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.is_trained = False
        
    def train(self, user_vectors: np.ndarray, item_vectors: np.ndarray, ratings: np.ndarray):
        """Train preference prediction model."""
        from sklearn.neural_network import MLPRegressor
        
        # Combine user and item features
        X = np.concatenate([user_vectors, item_vectors], axis=1)
        y = ratings
        
        # Train model
        self.model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=500
        )
        
        self.model.fit(X, y)
        self.is_trained = True
        
        logger.info("Preference learning model trained")
        
    def predict(self, user_vector: np.ndarray, item_vector: np.ndarray) -> float:
        """Predict user's rating for an item."""
        if not self.is_trained:
            return 3.0  # Default neutral rating
            
        X = np.concatenate([user_vector, item_vector]).reshape(1, -1)
        prediction = self.model.predict(X)[0]
        
        # Bound between 1 and 5
        return max(1.0, min(5.0, prediction))
    
    def save_model(self, path: str):
        """Save trained model."""
        if self.is_trained:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model."""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
            self.is_trained = True
        logger.info(f"Model loaded from {path}")
