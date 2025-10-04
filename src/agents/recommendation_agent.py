# src/agents/recommendation_agent.py

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import random
from dataclasses import dataclass, asdict
from collections import defaultdict
from agents.location_agent import Location, TransportMode
from datetime import timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger


@dataclass
class RecommendationItem:
    """Structure for a recommendation."""
    id: str
    name: str
    type: str  # restaurant, attraction, activity
    location: Dict[str, float]  # lat, lng
    rating: float
    price_range: str
    category: str
    features: List[str]
    distance: float  # from user in km
    popularity: float  # 0-1 score
    match_score: float  # How well it matches preferences
    
    def to_dict(self) -> Dict:
        return asdict(self)


class DeepQNetwork(nn.Module):
    """Deep Q-Network for recommendation learning."""
    
    def __init__(self, state_size: int, action_size: int):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class RecommendationAgent:
    """Agent for generating personalized recommendations with RL."""
    
    def __init__(self,
                 rag_system,
                 preference_extractor,
                 memory_manager,
                 location_agent=None):
        
        self.rag = rag_system
        self.preference_extractor = preference_extractor
        self.memory = memory_manager
        self.location_agent = location_agent
        
        # RL components
        self.state_size = 100  # User + context features
        self.action_size = 50   # Number of possible recommendations
        self.dqn = DeepQNetwork(self.state_size, self.action_size)
        self.target_dqn = DeepQNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        
        # RL hyperparameters
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95  # Discount factor
        self.batch_size = 32
        
        # Experience replay buffer
        self.replay_buffer = []
        self.buffer_size = 10000
        
        # Recommendation cache
        self.recommendation_cache = {}
        
        # Multi-objective weights
        self.objective_weights = {
            'relevance': 0.35,
            'distance': 0.25,
            'popularity': 0.20,
            'novelty': 0.10,
            'diversity': 0.10
        }
        
        logger.info("Recommendation Agent initialized with RL")
    
    def _generate_recommendations(self,
                              user_id: str,
                              session_id: str,
                              location: Dict[str, float],
                              context: Optional[Dict] = None,
                              num_recommendations: int = 5) -> List[RecommendationItem]:
        """Generate scored and RL-reranked recommendations with detailed info."""

        from agents.location_agent import Location, TransportMode

        user_location = Location(lat=location['lat'], lng=location['lng'])

        # 1️⃣ 获取用户偏好
        preferences = self.preference_extractor.get_preference_summary(user_id)
        preference_vector = self.preference_extractor.get_preference_vector(user_id)

        # 2️⃣ 上下文特征
        context_features = self._extract_context_features(context or {})
        state = self._create_state_vector(preference_vector, context_features, location)

        # 3️⃣ 获取候选项
        candidates = []

        # 3a. RAG / KB
        search_terms = []
        for pref_type in ['strong_preferences', 'moderate_preferences']:
            for pref in preferences.get(pref_type, []):
                search_terms.append(pref.split(':')[1].strip())
        query = ' '.join(search_terms[:5])
        filters = {'max_distance_km': 10}

        rag_results = self.rag.hybrid_search(query, top_k=num_recommendations*2, filters=filters)
        for doc in rag_results:
            metadata = doc.metadata
            item_loc = metadata.get('location', {'lat': 0, 'lng': 0})
            distance = self.location_agent.calculate_distance(user_location, Location(item_loc['lat'], item_loc['lng'])) \
                    if self.location_agent else self._calculate_distance(location['lat'], location['lng'], item_loc['lat'], item_loc['lng'])
            candidates.append(
                RecommendationItem(
                    id=doc.id,
                    name=metadata.get('name', 'Unknown'),
                    type=metadata.get('type', 'attraction'),
                    location=item_loc,
                    rating=metadata.get('rating', 3.5),
                    price_range=metadata.get('price_range', '$'),
                    category=metadata.get('category', 'general'),
                    features=metadata.get('features', []),
                    distance=distance,
                    popularity=metadata.get('popularity', 0.5),
                    match_score=0.0
                )
            )

        # 3b. Google Places
        if self.location_agent and len(candidates) < num_recommendations:
            place_types = self._extract_place_types(preferences)
            for place_type in place_types[:3]:
                try:
                    places = self.location_agent.find_nearby_places(
                        location=user_location,
                        place_type=place_type,
                        radius=5000,
                        min_rating=3.5,
                        max_results=5
                    )
                    for place in places:
                        if any(c.id == place.place_id for c in candidates):
                            continue
                        candidates.append(
                            RecommendationItem(
                                id=place.place_id,
                                name=place.name,
                                type=self._map_google_type(place.types[0] if place.types else 'point_of_interest'),
                                location={'lat': place.location.lat, 'lng': place.location.lng},
                                rating=place.rating or 0,
                                price_range=self._map_price_level(place.price_level),
                                category=place.types[0] if place.types else 'general',
                                features=place.types[:5] if place.types else [],
                                distance=self.location_agent.calculate_distance(user_location, place.location),
                                popularity=min((place.user_ratings_total or 0)/1000, 1.0),
                                match_score=0.0
                            )
                        )
                        if len(candidates) >= num_recommendations*2:
                            break
                except Exception as e:
                    logger.warning(f"Google Places fetch error: {e}")

        # 4️⃣ 打分
        candidates = self._score_candidates(candidates, state, preferences, location)

        # 5️⃣ RL reranking
        recommendations = self._apply_rl_reranking(candidates, state, num_recommendations)

        # 6️⃣ 补充 travel_time / is_open
        for rec in recommendations:
            try:
                rec_loc = Location(lat=rec.location.get('lat', 0), lng=rec.location.get('lng', 0))
                time_info = self.location_agent.get_time_to_location(user_location, rec_loc, TransportMode.WALKING)
                rec.travel_time_minutes = time_info.get('duration_minutes', 0) if time_info else None
                if rec.id.startswith('ChIJ'):
                    rec.is_open = self.location_agent.is_place_open(rec.id)
            except Exception as e:
                logger.debug(f"Could not enhance recommendation {rec.name}: {e}")

        return recommendations


    
    def get_contextual_recommendations(self,
                                      user_id: str,
                                      session_id: str,
                                      location: Dict[str, float],
                                      time_available_hours: float = 3.0,
                                      transport_mode: str = "walking",
                                      current_weather: Optional[Dict] = None) -> List[RecommendationItem]:
        """Get recommendations considering specific contextual constraints."""
        
        if not self.location_agent:
            # Fallback to standard recommendations
            return self.get_recommendations(user_id, session_id, location)
        
        from agents.location_agent import Location, TransportMode
        from agents.route_planner import RoutePlanner, POI, RouteConstraints
        
        # Get standard recommendations first
        recommendations = self.get_recommendations(
            user_id, session_id, location,
            context={'time_available': time_available_hours, 'weather': current_weather}
        )
        
        # Convert to POIs for route planning
        pois = []
        for rec in recommendations[:8]:  # Limit for optimization
            poi = POI(
                id=rec.id,
                name=rec.name,
                location=Location(
                    lat=rec.location.get('lat', 0),
                    lng=rec.location.get('lng', 0)
                ),
                visit_duration=60,  # Default 1 hour per place
                priority=rec.match_score,
                category=rec.category,
                cost=self._estimate_cost(rec.price_range),
                must_visit=(rec.match_score > 0.8)
            )
            pois.append(poi)
        
        # Create route planner and constraints
        route_planner = RoutePlanner(self.location_agent, self.memory)
        
        constraints = RouteConstraints(
            start_location=Location(lat=location.get('lat', 0), lng=location.get('lng', 0)),
            max_duration_hours=time_available_hours,
            transport_mode=TransportMode(transport_mode.lower()),
            start_time=datetime.now()
        )
        
        # Get optimized route
        optimized_route = route_planner.plan_route(pois, constraints)
        
        # Convert back to recommendations with route order
        ordered_recommendations = []
        for waypoint in optimized_route.waypoints:
            for rec in recommendations:
                if rec.id == waypoint.id:
                    rec.route_order = len(ordered_recommendations) + 1
                    rec.estimated_arrival_time = (
                        datetime.now() + 
                        timedelta(hours=sum(w.visit_duration/60 for w in optimized_route.waypoints[:rec.route_order]))
                    ).isoformat()
                    ordered_recommendations.append(rec)
                    break
        
        return ordered_recommendations
    
    def _estimate_cost(self, price_range: str) -> float:
        mapping = {
            'free': 0,
            '$': 15,      # ~$15 per person (inexpensive)
            '$$': 30,     # ~$30 per person (moderate)
            '$$$': 60,    # ~$60 per person (expensive)
            '$$$$': 100   # ~$100+ per person (very expensive)
        }
        return mapping.get(price_range, 30)  # Default to moderate ($30) if not found
    
    def _extract_context_features(self, context: Dict) -> np.ndarray:
        """Extract features from context."""
        features = []
        
        # Time features
        now = datetime.now()
        features.append(now.hour / 24)  # Normalized hour
        features.append(now.weekday() / 7)  # Normalized day of week
        
        # Weather features (if available)
        weather = context.get('weather', {})
        weather_map = {'clear': 1, 'clouds': 0.5, 'rain': 0, 'snow': -0.5}
        features.append(weather_map.get(weather.get('condition', 'clear'), 0.5))
        features.append(weather.get('temperature', 20) / 40)  # Normalized temp
        
        # User state features
        features.append(context.get('energy_level', 0.7))
        features.append(context.get('hunger_level', 0.5))
        
        # Pad to fixed size
        while len(features) < 20:
            features.append(0)
            
        return np.array(features[:20])
    
    def _create_state_vector(self,
                           preference_vector: np.ndarray,
                           context_features: np.ndarray,
                           location: Dict) -> np.ndarray:
        """Create state vector for RL."""
        # Combine all features
        location_features = np.array([
            location.get('lat', 0) / 90,  # Normalized latitude
            location.get('lng', 0) / 180  # Normalized longitude
        ])
        
        # Pad location features
        location_features = np.pad(location_features, (0, 28))
        
        # Combine all features
        state = np.concatenate([
            preference_vector,  # 50 dims
            context_features,   # 20 dims
            location_features   # 30 dims
        ])
        
        return state[:self.state_size]
    
    def _get_candidates(self,
                       preferences: Dict,
                       location: Dict,
                       num_candidates: int = 20) -> List[RecommendationItem]:
        """Get candidate items from knowledge base and location services."""
        
        candidates = []
        
        # 1. Get candidates from RAG system (knowledge base)
        search_terms = []
        for pref_type in ['strong_preferences', 'moderate_preferences']:
            for pref in preferences.get(pref_type, []):
                search_terms.append(pref.split(':')[1].strip())
        
        query = ' '.join(search_terms[:5])
        
        # Search with location filter
        filters = {
            'max_distance_km': 10
        }
        
        # Get results from RAG
        rag_results = self.rag.hybrid_search(query, top_k=num_candidates // 2, filters=filters)
        
        # Convert RAG results to RecommendationItems
        for doc in rag_results:
            metadata = doc.metadata
            
            # Calculate distance
            item_loc = metadata.get('location', {})
            if self.location_agent and item_loc:
                user_loc = Location(lat=location.get('lat', 0), lng=location.get('lng', 0))
                item_location = Location(lat=item_loc.get('lat', 0), lng=item_loc.get('lng', 0))
                distance = self.location_agent.calculate_distance(user_loc, item_location)
            else:
                # Fallback to simple calculation
                distance = self._calculate_distance(
                    location.get('lat', 0),
                    location.get('lng', 0),
                    item_loc.get('lat', 0),
                    item_loc.get('lng', 0)
                )
            
            item = RecommendationItem(
                id=doc.id,
                name=metadata.get('name', 'Unknown'),
                type=metadata.get('type', 'attraction'),
                location=item_loc,
                rating=metadata.get('rating', 3.5),
                price_range=metadata.get('price_range', '$'),
                category=metadata.get('category', 'general'),
                features=metadata.get('features', []),
                distance=distance,
                popularity=metadata.get('popularity', 0.5),
                match_score=0.0
            )
            candidates.append(item)
        
        # 2. Get real-time candidates from Google Places if location_agent available
        if self.location_agent and len(candidates) < num_candidates:
            from agents.location_agent import Location
            user_location = Location(lat=location.get('lat', 0), lng=location.get('lng', 0))
            
            # Determine place types based on preferences
            place_types = self._extract_place_types(preferences)
            
            for place_type in place_types[:3]:  # Limit API calls
                try:
                    places = self.location_agent.find_nearby_places(
                        location=user_location,
                        place_type=place_type,
                        radius=5000,  # 5km radius
                        min_rating=3.5,
                        max_results=5
                    )
                    
                    for place in places:
                        # Check if already in candidates
                        if any(c.id == place.place_id for c in candidates):
                            continue
                        
                        item = RecommendationItem(
                            id=place.place_id,
                            name=place.name,
                            type=self._map_google_type(place.types[0] if place.types else 'point_of_interest'),
                            location={'lat': place.location.lat, 'lng': place.location.lng},
                            rating=place.rating or 0,
                            price_range=self._map_price_level(place.price_level),
                            category=place.types[0] if place.types else 'general',
                            features=place.types[:5] if place.types else [],
                            distance=self.location_agent.calculate_distance(user_location, place.location),
                            popularity=min((place.user_ratings_total or 0) / 1000, 1.0),
                            match_score=0.0
                        )
                        candidates.append(item)
                        
                        if len(candidates) >= num_candidates:
                            break
                            
                except Exception as e:
                    logger.warning(f"Error fetching places from Google: {e}")
        
        return candidates[:num_candidates]
    
    def _extract_place_types(self, preferences: Dict) -> List[str]:
        """Extract Google Places types from preferences."""
        type_mapping = {
            'restaurant': 'restaurant',
            'food': 'restaurant',
            'cuisine': 'restaurant',
            'museum': 'museum',
            'art': 'art_gallery',
            'history': 'museum',
            'nature': 'park',
            'park': 'park',
            'shopping': 'shopping_mall',
            'nightlife': 'night_club',
            'bar': 'bar',
            'cafe': 'cafe',
            'attraction': 'tourist_attraction',
            'beach': 'natural_feature',
            'hotel': 'lodging'
        }
        
        place_types = []
        all_prefs = preferences.get('strong_preferences', []) + preferences.get('moderate_preferences', [])
        
        for pref in all_prefs:
            pref_lower = pref.lower()
            for keyword, place_type in type_mapping.items():
                if keyword in pref_lower and place_type not in place_types:
                    place_types.append(place_type)
        
        # Default types if none found
        if not place_types:
            place_types = ['tourist_attraction', 'restaurant', 'park']
        
        return place_types
    
    def _map_google_type(self, google_type: str) -> str:
        """Map Google place type to our type system."""
        mapping = {
            'restaurant': 'restaurant',
            'cafe': 'restaurant',
            'bar': 'restaurant',
            'museum': 'attraction',
            'park': 'attraction',
            'tourist_attraction': 'attraction',
            'lodging': 'accommodation',
            'shopping_mall': 'shopping',
            'store': 'shopping'
        }
        return mapping.get(google_type, 'attraction')
    
    def _map_price_level(self, price_level: Optional[int]) -> str:
        if price_level is None:
            return '$$'
        mapping = {
            0: 'free', 
            1: '$',
            2: '$$',
            3: '$$$',
            4: '$$$$'
        }
        return mapping.get(price_level, '$$')
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points in km (Haversine formula)."""
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in km
        
        return c * r
    
    def _score_candidates(self,
                         candidates: List[RecommendationItem],
                         state: np.ndarray,
                         preferences: Dict,
                         location: Dict) -> List[RecommendationItem]:
        """Score candidates using multi-objective optimization."""
        
        for candidate in candidates:
            scores = {}
            
            # Relevance score (how well it matches preferences)
            scores['relevance'] = self._calculate_relevance_score(
                candidate,
                preferences
            )
            
            # Distance score (closer is better)
            max_distance = 10.0  # km
            scores['distance'] = 1.0 - min(candidate.distance / max_distance, 1.0)
            
            # Popularity score
            scores['popularity'] = candidate.popularity
            
            # Novelty score (whether user has seen it before)
            scores['novelty'] = self._calculate_novelty_score(
                candidate,
                state
            )
            
            # Diversity score (how different from other recommendations)
            scores['diversity'] = 0.5  # Placeholder
            
            # Additional scores if location_agent available
            if self.location_agent:
                # Check if place is currently open
                try:
                    if hasattr(candidate, 'id') and candidate.id.startswith('ChIJ'):  # Google Place ID
                        is_open = self.location_agent.is_place_open(candidate.id)
                        if is_open is not None:
                            scores['availability'] = 1.0 if is_open else 0.3
                        else:
                            scores['availability'] = 0.7  # Unknown
                    else:
                        scores['availability'] = 0.7
                except:
                    scores['availability'] = 0.7
                
                # Weather suitability
                from agents.location_agent import Location
                place_location = Location(
                    lat=candidate.location.get('lat', 0),
                    lng=candidate.location.get('lng', 0)
                )
                weather = self.location_agent.get_weather(place_location)
                if weather:
                    # Outdoor places score lower in bad weather
                    if candidate.type in ['park', 'beach', 'outdoor'] and weather.get('condition') in ['rain', 'snow']:
                        scores['weather_suitable'] = 0.3
                    else:
                        scores['weather_suitable'] = 1.0
                else:
                    scores['weather_suitable'] = 0.8
            
            # Calculate weighted sum with updated weights
            if self.location_agent:
                weights = {
                    'relevance': 0.30,
                    'distance': 0.20,
                    'popularity': 0.15,
                    'novelty': 0.10,
                    'diversity': 0.10,
                    'availability': 0.10,
                    'weather_suitable': 0.05
                }
            else:
                weights = self.objective_weights
            
            total_score = 0
            for objective, weight in weights.items():
                total_score += scores.get(objective, 0) * weight
            
            candidate.match_score = total_score
        
        # Sort by match score
        candidates.sort(key=lambda x: x.match_score, reverse=True)
        
        return candidates
    
    def _calculate_relevance_score(self,
                                  item: RecommendationItem,
                                  preferences: Dict) -> float:
        """Calculate how relevant an item is to user preferences."""
        score = 0.5  # Base score
        
        # Check category matches
        strong_prefs = preferences.get('strong_preferences', [])
        for pref in strong_prefs:
            if item.category.lower() in pref.lower():
                score += 0.3
                
        # Check feature matches
        for feature in item.features:
            for pref in strong_prefs:
                if feature.lower() in pref.lower():
                    score += 0.1
                    
        # Price range match
        if 'budget' in str(preferences):
            # Simplified price matching
            score += 0.1
            
        return min(score, 1.0)
    
    def _calculate_novelty_score(self,
                                item: RecommendationItem,
                                state: np.ndarray) -> float:
        """Calculate novelty score (new/unseen items score higher)."""
        # Check if item was recommended before
        # Simplified - would check actual history
        return 0.8  # Assume mostly novel
    
    def _apply_rl_reranking(self,
                           candidates: List[RecommendationItem],
                           state: np.ndarray,
                           num_recommendations: int) -> List[RecommendationItem]:
        """Apply RL-based reranking."""
        
        if random.random() < self.epsilon:
            # Exploration: random selection
            selected = random.sample(
                candidates[:min(len(candidates), num_recommendations * 2)],
                min(num_recommendations, len(candidates))
            )
        else:
            # Exploitation: use DQN
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.dqn(state_tensor)
                
                # Map Q-values to candidates
                selected_indices = torch.topk(q_values, num_recommendations).indices[0]
                
                selected = []
                for idx in selected_indices:
                    if idx < len(candidates):
                        selected.append(candidates[idx])
                
                # Fill with top candidates if needed
                while len(selected) < num_recommendations and len(candidates) > len(selected):
                    for candidate in candidates:
                        if candidate not in selected:
                            selected.append(candidate)
                            break
        
        return selected[:num_recommendations]
    
    def update_from_feedback(self,
                            user_id: str,
                            session_id: str,
                            item_id: str,
                            feedback: float):
        """Update RL model based on user feedback."""
        
        # Get episode from memory
        episode_key = f"episode:{session_id}_{user_id}"
        
        # Create experience tuple
        # Simplified - would retrieve actual state and action
        state = np.random.randn(self.state_size)  # Placeholder
        action = 0  # Placeholder
        reward = (feedback - 3) / 2  # Normalize to [-1, 1]
        next_state = state + np.random.randn(self.state_size) * 0.1
        done = False
        
        # Store in replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))
        
        # Limit buffer size
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
        
        # Train if enough experiences
        if len(self.replay_buffer) >= self.batch_size:
            self._train_dqn()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update preference model
        self.preference_extractor.learn_from_feedback(
            user_id,
            item_id,
            feedback,
            {'item_id': item_id}
        )
        
        logger.info(f"Updated RL model with feedback: {feedback} for item {item_id}")
    
    def _train_dqn(self):
        """Train DQN on batch from replay buffer."""
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q = self.dqn(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        next_q = self.target_dqn(next_states).max(1)[0].detach()
        target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        # Loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        if random.random() < 0.01:  # 1% chance
            self.target_dqn.load_state_dict(self.dqn.state_dict())
    
    def get_explanation(self, recommendation: RecommendationItem, preferences: Dict) -> str:
        """Generate explanation for why item was recommended."""
        reasons = []
        
        # Check preference matches
        for pref in preferences.get('strong_preferences', []):
            if recommendation.category.lower() in pref.lower():
                reasons.append(f"Matches your interest in {pref}")
        
        # Distance
        if recommendation.distance < 1:
            reasons.append(f"Very close by ({recommendation.distance:.1f} km)")
        elif recommendation.distance < 3:
            reasons.append(f"Nearby ({recommendation.distance:.1f} km)")
        
        # Rating
        if recommendation.rating >= 4.5:
            reasons.append(f"Highly rated ({recommendation.rating}/5)")
        
        # Popularity
        if recommendation.popularity > 0.8:
            reasons.append("Very popular with visitors")
        
        # Match score
        if recommendation.match_score > 0.8:
            reasons.append("Excellent match for your preferences")
        
        if not reasons:
            reasons.append("Based on your overall preferences")
        
        return " • ".join(reasons)  # This should be the last line
