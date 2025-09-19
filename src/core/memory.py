# src/core/memory.py

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import deque
import redis
from loguru import logger

class ConversationMemory:
    """Manages conversation history and context."""
    
    def __init__(self, redis_client: redis.Redis, max_history: int = 50):
        self.redis_client = redis_client
        self.max_history = max_history
        
    def add_message(self, session_id: str, message: Dict[str, Any]):
        """Add a message to conversation history."""
        key = f"conversation:{session_id}"
        
        # Add timestamp
        message['timestamp'] = datetime.now().isoformat()
        
        # Store in Redis list
        self.redis_client.lpush(key, json.dumps(message))
        
        # Trim to max history
        self.redis_client.ltrim(key, 0, self.max_history - 1)
        
        # Set expiration (24 hours)
        self.redis_client.expire(key, 86400)
        
    def get_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve conversation history."""
        key = f"conversation:{session_id}"
        
        # Get messages from Redis
        messages = self.redis_client.lrange(key, 0, limit - 1)
        
        # Parse and return
        return [json.loads(msg) for msg in messages]
    
    def get_context_window(self, session_id: str, window_size: int = 5) -> str:
        """Get recent context for the LLM."""
        history = self.get_history(session_id, window_size)
        
        context = []
        for msg in reversed(history):
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            context.append(f"{role}: {content}")
            
        return "\n".join(context)
    
    def clear_session(self, session_id: str):
        """Clear conversation history for a session."""
        key = f"conversation:{session_id}"
        self.redis_client.delete(key)


class UserProfileMemory:
    """Manages long-term user preferences and history."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        
    def update_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Update user preferences."""
        key = f"user_profile:{user_id}"
        
        # Get existing profile
        existing = self.redis_client.get(key)
        if existing:
            profile = json.loads(existing)
        else:
            profile = {
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                'preferences': {},
                'history': [],
                'feedback': []
            }
        
        # Merge preferences
        profile['preferences'].update(preferences)
        profile['last_updated'] = datetime.now().isoformat()
        
        # Save back to Redis
        self.redis_client.set(key, json.dumps(profile))
        
    def get_profile(self, user_id: str) -> Optional[Dict]:
        """Retrieve user profile."""
        key = f"user_profile:{user_id}"
        data = self.redis_client.get(key)
        
        if data:
            return json.loads(data)
        return None
    
    def add_interaction(self, user_id: str, interaction: Dict[str, Any]):
        """Add an interaction to user history."""
        profile = self.get_profile(user_id) or {
            'user_id': user_id,
            'preferences': {},
            'history': [],
            'feedback': []
        }
        
        # Add timestamp
        interaction['timestamp'] = datetime.now().isoformat()
        
        # Add to history (keep last 100 interactions)
        profile['history'].append(interaction)
        profile['history'] = profile['history'][-100:]
        
        # Save
        key = f"user_profile:{user_id}"
        self.redis_client.set(key, json.dumps(profile))
        
    def add_feedback(self, user_id: str, item_id: str, rating: float, feedback_type: str):
        """Store user feedback for recommendations."""
        profile = self.get_profile(user_id) or {
            'user_id': user_id,
            'preferences': {},
            'history': [],
            'feedback': []
        }
        
        feedback = {
            'item_id': item_id,
            'rating': rating,
            'type': feedback_type,
            'timestamp': datetime.now().isoformat()
        }
        
        profile['feedback'].append(feedback)
        profile['feedback'] = profile['feedback'][-500:]  # Keep last 500 feedbacks
        
        # Save
        key = f"user_profile:{user_id}"
        self.redis_client.set(key, json.dumps(profile))


class EpisodicMemory:
    """Stores successful recommendation episodes for learning."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        
    def store_episode(self, episode: Dict[str, Any]):
        """Store a recommendation episode."""
        key = f"episode:{episode['episode_id']}"
        
        episode['timestamp'] = datetime.now().isoformat()
        
        # Store episode
        self.redis_client.set(key, json.dumps(episode))
        
        # Add to index by user
        user_key = f"user_episodes:{episode['user_id']}"
        self.redis_client.sadd(user_key, episode['episode_id'])
        
        # Add to successful episodes if rating is high
        if episode.get('rating', 0) >= 4:
            success_key = "successful_episodes"
            self.redis_client.zadd(
                success_key,
                {episode['episode_id']: episode.get('rating', 0)}
            )
            
    def get_similar_episodes(self, context: Dict[str, Any], limit: int = 5) -> List[Dict]:
        """Retrieve similar successful episodes."""
        # Get successful episodes
        success_key = "successful_episodes"
        episode_ids = self.redis_client.zrevrange(success_key, 0, 50)
        
        episodes = []
        for eid in episode_ids:
            key = f"episode:{eid.decode() if isinstance(eid, bytes) else eid}"
            data = self.redis_client.get(key)
            if data:
                episode = json.loads(data)
                
                # Simple similarity check (can be enhanced)
                if self._is_similar(episode.get('context', {}), context):
                    episodes.append(episode)
                    
                if len(episodes) >= limit:
                    break
                    
        return episodes
    
    def _is_similar(self, context1: Dict, context2: Dict) -> bool:
        """Check if two contexts are similar."""
        # Simple implementation - can be enhanced with embeddings
        similar_keys = ['location_type', 'time_of_day', 'weather', 'group_size']
        
        matches = 0
        for key in similar_keys:
            if key in context1 and key in context2:
                if context1[key] == context2[key]:
                    matches += 1
                    
        return matches >= 2  # At least 2 matching attributes


class WorkingMemory:
    """Temporary memory for current task processing."""
    
    def __init__(self):
        self.memory = {}
        
    def set(self, key: str, value: Any):
        """Store temporary data."""
        self.memory[key] = {
            'value': value,
            'timestamp': datetime.now()
        }
        
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve temporary data."""
        if key in self.memory:
            return self.memory[key]['value']
        return default
    
    def clear(self):
        """Clear all working memory."""
        self.memory.clear()
        
    def cleanup_old(self, max_age_seconds: int = 3600):
        """Remove old entries."""
        now = datetime.now()
        keys_to_remove = []
        
        for key, data in self.memory.items():
            age = (now - data['timestamp']).seconds
            if age > max_age_seconds:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.memory[key]


class MemoryManager:
    """Central memory management system."""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        
        self.conversation = ConversationMemory(self.redis_client)
        self.user_profile = UserProfileMemory(self.redis_client)
        self.episodic = EpisodicMemory(self.redis_client)
        self.working = WorkingMemory()
        
        logger.info("Memory Manager initialized")
        
    def get_full_context(self, session_id: str, user_id: Optional[str] = None) -> Dict:
        """Get complete context for decision making."""
        context = {
            'conversation': self.conversation.get_context_window(session_id),
            'working_memory': self.working.memory
        }
        
        if user_id:
            context['user_profile'] = self.user_profile.get_profile(user_id)
            context['similar_episodes'] = self.episodic.get_similar_episodes(
                context.get('current_context', {})
            )
            
        return context
    
    def store_interaction(self, session_id: str, user_id: str, 
                         message: Dict, response: Dict, 
                         recommendations: List[Dict] = None):
        """Store complete interaction."""
        # Store conversation
        self.conversation.add_message(session_id, message)
        self.conversation.add_message(session_id, response)
        
        # Store in user history
        if user_id:
            interaction = {
                'message': message,
                'response': response,
                'recommendations': recommendations
            }
            self.user_profile.add_interaction(user_id, interaction)
            
        logger.info(f"Stored interaction for session {session_id}, user {user_id}")
