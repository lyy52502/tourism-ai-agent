# src/main.py

import os
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis
from openai import OpenAI
import uvicorn
from loguru import logger
import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uuid
# Import our agents and systems
from core.memory import MemoryManager
from core.rag_system import RAGSystem
from agents.conversation_agent import ConversationAgent
from agents.preference_agent import PreferenceExtractor
from agents.recommendation_agent import RecommendationAgent
from agents.location_agent import LocationAgent, TransportMode
from agents.route_planner import RoutePlanner, POI, RouteConstraints
# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Pydantic models for API
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    location: Optional[Dict[str, float]] = None

class UserLocation(BaseModel):
    lat: float
    lng: float
    accuracy: Optional[float] = None

class FeedbackRequest(BaseModel):
    session_id: str
    user_id: str
    item_id: str
    rating: float = Field(ge=1, le=5)
    comment: Optional[str] = None

class RecommendationRequest(BaseModel):
    user_id: str
    location: UserLocation
    preferences: Optional[Dict[str, Any]] = None
    num_recommendations: int = Field(default=5, ge=1, le=20)

class RouteRequest(BaseModel):
    user_id: str
    location: UserLocation
    destinations: List[str]
    mode: str = Field(default="walking", pattern="^(walking|driving|transit)$")
    optimize: bool = True

# Global instances
location_agent = None
memory_manager = None
rag_system = None
conversation_agent = None
preference_extractor = None
recommendation_agent = None
openai_client = None

# Fix for main.py lifespan function - CORRECT INITIALIZATION ORDER

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global memory_manager, rag_system, conversation_agent, preference_extractor, recommendation_agent, openai_client, location_agent
    
    # Startup
    logger.info("Initializing Tourism AI Agent...")
    
    try:
        # ✅ Step 1: Initialize OpenAI client
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("✅ OpenAI client initialized")
        
        # ✅ Step 2: Initialize memory manager
        memory_manager = MemoryManager(
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", 6379))
        )
        logger.info("✅ Memory manager initialized")
        
        # ✅ Step 3: Initialize RAG system
        rag_system = RAGSystem(
            embedding_model=config['embedding']['model']
        )
        logger.info("✅ RAG system initialized")
        
        # Load initial data if available
        data_path = "data/knowledge_base/tourism_data.json"
        if os.path.exists(data_path):
            rag_system.load_tourism_data(data_path)
            logger.info(f"✅ Loaded tourism data from {data_path}")
        else:
            logger.warning(f"⚠️ Tourism data file not found: {data_path}")
        
        # ✅ Step 4: Initialize location agent
        location_agent = LocationAgent(
            google_maps_api_key=os.getenv("GOOGLE_MAPS_API_KEY"),
            openweather_api_key=os.getenv("OPENWEATHER_API_KEY")
        )
        logger.info("✅ Location agent initialized")

        # ✅ Step 5: Initialize preference extractor
        preference_extractor = PreferenceExtractor(
            openai_client=openai_client,
            memory_manager=memory_manager
        )
        logger.info("✅ Preference extractor initialized")
        
        # ✅ Step 6: Initialize recommendation agent (BEFORE conversation agent!)
        recommendation_agent = RecommendationAgent(
            rag_system=rag_system,
            preference_extractor=preference_extractor,
            memory_manager=memory_manager,
            location_agent=location_agent  # ← CRITICAL: Must pass location_agent
        )
        logger.info("✅ Recommendation agent initialized")
        
        # ✅ Step 7: Initialize conversation agent (LAST - needs recommendation_agent!)
        conversation_agent = ConversationAgent(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            memory_manager=memory_manager,
            preference_extractor=preference_extractor,
            location_agent=location_agent,
            recommendation_agent=recommendation_agent  # ← CRITICAL: Must pass this!
        )
        logger.info("✅ Conversation agent initialized")
        
        # ✅ Verify all components are initialized
        logger.info("=" * 50)
        logger.info("🎉 All systems initialized successfully!")
        logger.info(f"   • OpenAI: {openai_client is not None}")
        logger.info(f"   • Memory: {memory_manager is not None}")
        logger.info(f"   • RAG: {rag_system is not None}")
        logger.info(f"   • Location Agent: {location_agent is not None}")
        logger.info(f"   • Preference Extractor: {preference_extractor is not None}")
        logger.info(f"   • Recommendation Agent: {recommendation_agent is not None}")
        logger.info(f"   • Conversation Agent: {conversation_agent is not None}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize systems: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Tourism AI Agent...")
    # Add cleanup code here if needed

# Create FastAPI app
app = FastAPI(
    title="Tourism AI Agent",
    description="Intelligent tourism recommendation system with conversational AI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "redis": memory_manager is not None,
            "milvus": rag_system is not None,
            "openai": openai_client is not None
        }
    }

# Updated /chat endpoint in main.py
# Updated /chat endpoint in main.py

@app.post("/chat")
async def chat(request: Request):
    """Process chat message and return response with recommendations."""
    try:
        data = await request.json()
        message = data.get("message", "")
        session_id = data.get("session_id", str(uuid.uuid4()))
        user_id = data.get("user_id", "anonymous")
        location = data.get("location", None)
        
        ip_address = request.client.host
        
        logger.info(f"📨 Chat request")
        logger.info(f"   Session: {session_id}")
        logger.info(f"   User: {user_id}")
        logger.info(f"   Message: '{message}'")
        logger.info(f"   Location: {location}")
        logger.info(f"   IP: {ip_address}")

        # ✅ Process message with conversation agent - it handles recommendations internally now!
        response_data = conversation_agent.process_message(
            session_id=session_id,
            user_id=user_id,
            message=message,
            location=location,
            ip_address=ip_address
        )

        # ✅ Extract recommendations from conversation agent response
        recommendations = []
        raw_recommendations = response_data.get('recommendations', [])
        
        logger.info(f"📦 Got {len(raw_recommendations)} recommendations from conversation agent")
        
        # ✅ Format recommendations with all details
        for r in raw_recommendations:
            rec_dict = {
                "id": r.id,
                "name": r.name,
                "type": r.type,
                "rating": r.rating,
                "distance": round(r.distance, 2),
                "match_score": round(r.match_score, 2),
                "price_range": r.price_range,
                "category": r.category,
                "location": r.location,
                "features": r.features,
                "popularity": r.popularity
            }
            
            # Add photos if available
            if hasattr(r, 'photos') and r.photos:
                rec_dict["photos"] = r.photos
            elif hasattr(r, 'photo_reference') and r.photo_reference:
                api_key = os.getenv("GOOGLE_MAPS_API_KEY")
                if api_key:
                    rec_dict["photos"] = [
                        f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photo_reference={r.photo_reference}&key={api_key}"
                    ]
            
            # Add travel time if available
            if hasattr(r, 'travel_time_minutes') and r.travel_time_minutes:
                rec_dict["travel_time_minutes"] = round(r.travel_time_minutes, 1)
            
            # Add opening status if available
            if hasattr(r, 'is_open') and r.is_open is not None:
                rec_dict["is_open"] = r.is_open
            
            recommendations.append(rec_dict)
        
        logger.info(f"✅ Formatted {len(recommendations)} recommendations")
        
        # Get reply from conversation agent
        reply = response_data.get('message', "I'm here to help you explore Uppsala!")
        
        # Get location info
        response_location = response_data.get('location', location)

        # Store interaction in memory
        memory_manager.store_interaction(
            session_id=session_id,
            user_id=user_id,
            message={"role": "user", "content": message},
            response={"role": "assistant", "content": reply},
            recommendations=recommendations
        )
        
        logger.info(f"✅ Returning response with {len(recommendations)} recommendations")
        
        return {
            "reply": reply,
            "recommendations": recommendations,
            "session_id": session_id,
            "location": response_location,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return {
            "reply": "I apologize, but I encountered an error. Please try again.",
            "recommendations": [],
            "status": "error",
            "error": str(e)
        }
# WebSocket for real-time chat
@app.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    logger.info(f"WebSocket connection established for session {session_id}")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Process message
            response = conversation_agent.process_message(
                session_id=session_id,
                user_id=data.get('user_id', 'anonymous'),
                message=data.get('message', ''),
                location=data.get('location')
            )
            
            # Send response
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()
        
@app.post("/route/plan")
async def plan_route(
    user_id: str,
    place_ids: List[str],
    constraints: Dict[str, Any]
):
    """Plan optimized route through multiple places."""
    # Implementation here

@app.get("/places/nearby")
async def get_nearby_places(
    lat: float,
    lng: float,
    type: Optional[str] = None,
    radius: int = 5000
):
    """Get nearby places."""
    # Implementation here

@app.get("/weather")
async def get_weather(lat: float, lng: float):
    """Get current weather."""
    # Implementation here

# Get recommendations
@app.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Get personalized recommendations."""
    try:
        # Get recommendations
        recommendations = recommendation_agent.get_recommendations(
            user_id=request.user_id,
            session_id=str(uuid.uuid4()),  # Generate new session for this request
            location={'lat': request.location.lat, 'lng': request.location.lng},
            context=request.preferences,
            num_recommendations=request.num_recommendations
        )
        
        # Format response
        result = []
        for rec in recommendations:
            item = rec.to_dict()
            
            # Add explanation
            preferences = preference_extractor.get_preference_summary(request.user_id)
            item['explanation'] = recommendation_agent.get_explanation(rec, preferences)
            
            result.append(item)
        
        return {
            "recommendations": result,
            "total": len(result),
            "user_id": request.user_id
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Submit feedback
@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a recommendation."""
    try:
        # Update RL model with feedback
        recommendation_agent.update_from_feedback(
            user_id=request.user_id,
            session_id=request.session_id,
            item_id=request.item_id,
            feedback=request.rating
        )
        
        # Store feedback in memory
        memory_manager.user_profile.add_feedback(
            user_id=request.user_id,
            item_id=request.item_id,
            rating=request.rating,
            feedback_type='explicit'
        )
        
        # Store comment if provided
        if request.comment:
            memory_manager.conversation.add_message(
                session_id=request.session_id,
                message={
                    'role': 'feedback',
                    'content': request.comment,
                    'rating': request.rating,
                    'item_id': request.item_id
                }
            )
        
        return {
            "status": "success",
            "message": "Feedback recorded successfully",
            "item_id": request.item_id,
            "rating": request.rating
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get user preferences
@app.get("/preferences/{user_id}")
async def get_preferences(user_id: str):
    """Get user preferences."""
    try:
        preferences = preference_extractor.get_preference_summary(user_id)
        profile = memory_manager.user_profile.get_profile(user_id)
        
        return {
            "user_id": user_id,
            "preferences": preferences,
            "profile": profile
        }
        
    except Exception as e:
        logger.error(f"Error getting preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Update preferences
@app.put("/preferences/{user_id}")
async def update_preferences(user_id: str, preferences: Dict[str, Any] = Body(...)):
    """Update user preferences."""
    try:
        # Update preferences
        memory_manager.user_profile.update_preferences(user_id, preferences)
        
        # Update preference extractor
        preference_extractor.update_from_entities(user_id, preferences)
        
        return {
            "status": "success",
            "message": "Preferences updated successfully",
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get conversation history
@app.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = Query(default=10, ge=1, le=100)):
    """Get conversation history for a session."""
    try:
        history = memory_manager.conversation.get_history(session_id, limit)
        
        return {
            "session_id": session_id,
            "history": history,
            "count": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Search knowledge base
@app.get("/search")
async def search_knowledge(
    query: str = Query(..., min_length=1),
    filters: Optional[Dict[str, Any]] = None,
    limit: int = Query(default=10, ge=1, le=50)
):
    """Search the tourism knowledge base."""
    try:
        results = rag_system.hybrid_search(
            query=query,
            top_k=limit,
            filters=filters
        )
        
        # Format results
        formatted_results = []
        for doc in results:
            formatted_results.append({
                'id': doc.id,
                'content': doc.content,
                'metadata': doc.metadata
            })
        
        return {
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results)
        }
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add tourism data
@app.post("/data/add")
async def add_tourism_data(data: Dict[str, Any] = Body(...)):
    """Add new tourism data to knowledge base."""
    try:
        # Format content
        content = rag_system._format_tourism_content(data)
        
        # Extract metadata
        metadata = {
            'type': data.get('type', 'attraction'),
            'name': data.get('name'),
            'location': data.get('location'),
            'category': data.get('category'),
            'price_range': data.get('price_range'),
            'rating': data.get('rating'),
            'opening_hours': data.get('opening_hours')
        }
        
        # Add to RAG system
        rag_system.add_document(
            content=content,
            metadata=metadata,
            doc_id=data.get('id')
        )
        
        return {
            "status": "success",
            "message": "Data added successfully",
            "id": data.get('id')
        }
        
    except Exception as e:
        logger.error(f"Error adding tourism data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Reset session
@app.delete("/session/{session_id}")
async def reset_session(session_id: str):
    """Reset/clear a conversation session."""
    try:
        memory_manager.conversation.clear_session(session_id)
        
        return {
            "status": "success",
            "message": "Session cleared successfully",
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Error resetting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoint
@app.get("/analytics/summary")
async def get_analytics():
    """Get system analytics summary."""
    try:
        # This would connect to your analytics system
        # For now, return mock data
        return {
            "total_users": 150,
            "active_sessions": 12,
            "recommendations_today": 234,
            "average_rating": 4.2,
            "popular_categories": ["restaurants", "attractions", "activities"],
            "system_status": "operational"
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Main entry point
if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configure logging
    logger.add("logs/app_{time}.log", rotation="500 MB", level="INFO")
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("APP_PORT", 8000)),
        reload=os.getenv("APP_ENV") == "development"
    )
