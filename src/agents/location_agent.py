# src/agents/location_agent.py

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import math
import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import googlemaps
import requests
from loguru import logger
import asyncio
import aiohttp
from cachetools import TTLCache
import polyline

class TransportMode(Enum):
    """Transportation modes."""
    WALKING = "walking"
    DRIVING = "driving"
    TRANSIT = "transit"
    BICYCLING = "bicycling"

@dataclass
class Location:
    """Location data structure."""
    lat: float
    lng: float
    address: Optional[str] = None
    name: Optional[str] = None
    place_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.lat, self.lng)

@dataclass
class PlaceDetails:
    """Detailed place information."""
    place_id: str
    name: str
    location: Location
    types: List[str]
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    price_level: Optional[int] = None
    opening_hours: Optional[Dict] = None
    phone_number: Optional[str] = None
    website: Optional[str] = None
    vicinity: Optional[str] = None
    photos: Optional[List[str]] = None
    reviews: Optional[List[Dict]] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['location'] = self.location.to_dict()
        return data

@dataclass
class RouteSegment:
    """Route segment information."""
    start: Location
    end: Location
    distance: float  # meters
    duration: float  # seconds
    mode: TransportMode
    instructions: Optional[str] = None
    polyline: Optional[str] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['start'] = self.start.to_dict()
        data['end'] = self.end.to_dict()
        data['mode'] = self.mode.value
        return data

class LocationAgent:
    """Agent for handling location-based services."""
    
    def __init__(self, 
                 google_maps_api_key: Optional[str] = None,
                 openweather_api_key: Optional[str] = None,
                 cache_ttl: int = 3600):
        
        # Initialize Google Maps client
        self.gmaps = None
        if google_maps_api_key:
            self.gmaps = googlemaps.Client(key=google_maps_api_key)
        
        # Weather API key
        self.weather_api_key = openweather_api_key
        
        # Initialize geocoder
        self.geocoder = Nominatim(user_agent="tourism_ai_agent")
        
        # Cache for API responses
        self.place_cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self.route_cache = TTLCache(maxsize=500, ttl=cache_ttl)
        self.weather_cache = TTLCache(maxsize=100, ttl=1800)  # 30 min for weather
        
        # Default radius for searches (in meters)
        self.default_radius = 5000  # 5 km
        self.max_radius = 50000  # 50 km
        
        logger.info("Location Agent initialized")
    
    def get_user_location(self, 
                         ip_address: Optional[str] = None,
                         fallback: Optional[Dict] = None) -> Location:
        """Get user location from IP or use fallback."""
        
        if ip_address:
            try:
                # Use IP geolocation service
                response = requests.get(f"http://ip-api.com/json/{ip_address}")
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        return Location(
                            lat=data['lat'],
                            lng=data['lon'],
                            address=f"{data.get('city', '')}, {data.get('country', '')}"
                        )
            except Exception as e:
                logger.error(f"Failed to get location from IP: {e}")
        
        # Use fallback location
        if fallback:
            return Location(
                lat=fallback.get('lat', 0),
                lng=fallback.get('lng', 0),
                address=fallback.get('address')
            )
        
        # Default location (Uppsala, Sweden)
        return Location(
            lat=59.8586,
            lng=17.6389,
            address="Uppsala, Sweden"
        )
    
    def calculate_distance(self, 
                          loc1: Location, 
                          loc2: Location,
                          unit: str = 'km') -> float:
        """Calculate distance between two locations."""
        
        # Use geodesic distance (more accurate than haversine for short distances)
        distance = geodesic(loc1.to_tuple(), loc2.to_tuple())
        
        if unit == 'km':
            return distance.kilometers
        elif unit == 'm':
            return distance.meters
        elif unit == 'miles':
            return distance.miles
        else:
            return distance.kilometers
    
    def find_nearby_places(self,
                          location: Location,
                          place_type: Optional[str] = None,
                          keyword: Optional[str] = None,
                          radius: Optional[int] = None,
                          min_rating: Optional[float] = None,
                          max_results: int = 20) -> List[PlaceDetails]:
        """Find nearby places using Google Places API."""
        
        if not self.gmaps:
            logger.warning("Google Maps client not initialized")
            return []
        
        # Check cache
        cache_key = f"nearby_{location.lat}_{location.lng}_{place_type}_{keyword}"
        if cache_key in self.place_cache:
            return self.place_cache[cache_key]
        
        try:
            # Search parameters
            params = {
                'location': location.to_tuple(),
                'radius': radius or self.default_radius,
                'type': place_type,
                'keyword': keyword
            }
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            # Search for places
            results = self.gmaps.places_nearby(**params)
            
            places = []
            for place in results.get('results', [])[:max_results]:
                # Filter by rating if specified
                if min_rating and place.get('rating', 0) < min_rating:
                    continue
                
                # Create PlaceDetails object
                place_location = Location(
                    lat=place['geometry']['location']['lat'],
                    lng=place['geometry']['location']['lng'],
                    address=place.get('vicinity'),
                    name=place.get('name'),
                    place_id=place.get('place_id')
                )
                
                place_details = PlaceDetails(
                    place_id=place['place_id'],
                    name=place['name'],
                    location=place_location,
                    types=place.get('types', []),
                    rating=place.get('rating'),
                    user_ratings_total=place.get('user_ratings_total'),
                    price_level=place.get('price_level'),
                    opening_hours=place.get('opening_hours'),
                    vicinity=place.get('vicinity'),
                    photos=[photo.get('photo_reference') for photo in place.get('photos', [])][:3]
                )
                
                places.append(place_details)
            
            # Cache results
            self.place_cache[cache_key] = places
            
            return places
            
        except Exception as e:
            logger.error(f"Error finding nearby places: {e}")
            return []
    
    def get_place_details(self, place_id: str) -> Optional[PlaceDetails]:
        """Get detailed information about a place."""
        
        if not self.gmaps:
            return None
        
        # Check cache
        cache_key = f"place_{place_id}"
        if cache_key in self.place_cache:
            return self.place_cache[cache_key]
        
        try:
            # Get place details
            result = self.gmaps.place(
                place_id=place_id,
                fields=['name', 'rating', 'formatted_address', 'phone_number', 
                       'website', 'opening_hours', 'price_level', 'reviews',
                       'types', 'geometry', 'photos', 'user_ratings_total']
            )
            
            place = result.get('result', {})
            
            # Create PlaceDetails object
            place_location = Location(
                lat=place['geometry']['location']['lat'],
                lng=place['geometry']['location']['lng'],
                address=place.get('formatted_address'),
                name=place.get('name'),
                place_id=place_id
            )
            
            place_details = PlaceDetails(
                place_id=place_id,
                name=place.get('name', 'Unknown'),
                location=place_location,
                types=place.get('types', []),
                rating=place.get('rating'),
                user_ratings_total=place.get('user_ratings_total'),
                price_level=place.get('price_level'),
                opening_hours=place.get('opening_hours'),
                phone_number=place.get('formatted_phone_number'),
                website=place.get('website'),
                vicinity=place.get('formatted_address'),
                photos=[photo.get('photo_reference') for photo in place.get('photos', [])][:5],
                reviews=place.get('reviews', [])[:5]
            )
            
            # Cache result
            self.place_cache[cache_key] = place_details
            
            return place_details
            
        except Exception as e:
            logger.error(f"Error getting place details: {e}")
            return None
    
    def geocode_address(self, address: str) -> Optional[Location]:
        """Convert address to coordinates."""
        
        try:
            # Try Google Maps first if available
            if self.gmaps:
                result = self.gmaps.geocode(address)
                if result:
                    location = result[0]['geometry']['location']
                    return Location(
                        lat=location['lat'],
                        lng=location['lng'],
                        address=result[0]['formatted_address']
                    )
            
            # Fallback to Nominatim
            location = self.geocoder.geocode(address)
            if location:
                return Location(
                    lat=location.latitude,
                    lng=location.longitude,
                    address=location.address
                )
                
        except Exception as e:
            logger.error(f"Error geocoding address: {e}")
        
        return None
    
    def reverse_geocode(self, location: Location) -> Optional[str]:
        """Convert coordinates to address."""
        
        try:
            # Try Google Maps first if available
            if self.gmaps:
                result = self.gmaps.reverse_geocode(location.to_tuple())
                if result:
                    return result[0]['formatted_address']
            
            # Fallback to Nominatim
            result = self.geocoder.reverse(location.to_tuple())
            if result:
                return result.address
                
        except Exception as e:
            logger.error(f"Error reverse geocoding: {e}")
        
        return None
    
    def get_route(self,
                 origin: Location,
                 destination: Location,
                 waypoints: Optional[List[Location]] = None,
                 mode: TransportMode = TransportMode.DRIVING,
                 optimize_waypoints: bool = True,
                 departure_time: Optional[datetime] = None) -> Optional[Dict]:
        """Get route between locations."""
        
        if not self.gmaps:
            return self._get_simple_route(origin, destination, waypoints)
        
        # Create cache key
        cache_key = f"route_{origin.lat}_{origin.lng}_{destination.lat}_{destination.lng}_{mode.value}"
        if cache_key in self.route_cache:
            return self.route_cache[cache_key]
        
        try:
            # Prepare waypoints
            waypoint_list = None
            if waypoints:
                waypoint_list = [w.to_tuple() for w in waypoints]
            
            # Get directions
            result = self.gmaps.directions(
                origin=origin.to_tuple(),
                destination=destination.to_tuple(),
                waypoints=waypoint_list,
                mode=mode.value,
                optimize_waypoints=optimize_waypoints,
                departure_time=departure_time or datetime.now()
            )
            
            if not result:
                return None
            
            route = result[0]
            
            # Process route segments
            segments = []
            for leg in route['legs']:
                for step in leg['steps']:
                    segment = RouteSegment(
                        start=Location(
                            lat=step['start_location']['lat'],
                            lng=step['start_location']['lng']
                        ),
                        end=Location(
                            lat=step['end_location']['lat'],
                            lng=step['end_location']['lng']
                        ),
                        distance=step['distance']['value'],
                        duration=step['duration']['value'],
                        mode=TransportMode(step.get('travel_mode', mode.value).lower()),
                        instructions=step.get('html_instructions'),
                        polyline=step.get('polyline', {}).get('points')
                    )
                    segments.append(segment)
            
            # Create route summary
            route_data = {
                'segments': [s.to_dict() for s in segments],
                'total_distance': sum(leg['distance']['value'] for leg in route['legs']),
                'total_duration': sum(leg['duration']['value'] for leg in route['legs']),
                'waypoint_order': route.get('waypoint_order', []),
                'overview_polyline': route.get('overview_polyline', {}).get('points'),
                'bounds': route.get('bounds'),
                'warnings': route.get('warnings', [])
            }
            
            # Cache result
            self.route_cache[cache_key] = route_data
            
            return route_data
            
        except Exception as e:
            logger.error(f"Error getting route: {e}")
            return self._get_simple_route(origin, destination, waypoints)
    
    def _get_simple_route(self,
                         origin: Location,
                         destination: Location,
                         waypoints: Optional[List[Location]] = None) -> Dict:
        """Get simple straight-line route when Google Maps is unavailable."""
        
        segments = []
        current = origin
        
        # Add waypoints if provided
        all_points = waypoints.copy() if waypoints else []
        all_points.append(destination)
        
        total_distance = 0
        total_duration = 0
        
        for point in all_points:
            distance = self.calculate_distance(current, point, 'm')
            # Estimate duration based on walking speed (5 km/h)
            duration = (distance / 1000) * 12 * 60  # minutes to seconds
            
            segment = RouteSegment(
                start=current,
                end=point,
                distance=distance,
                duration=duration,
                mode=TransportMode.WALKING,
                instructions=f"Go from {current.name or 'current location'} to {point.name or 'destination'}"
            )
            segments.append(segment)
            
            total_distance += distance
            total_duration += duration
            current = point
        
        return {
            'segments': [s.to_dict() for s in segments],
            'total_distance': total_distance,
            'total_duration': total_duration,
            'waypoint_order': list(range(len(waypoints))) if waypoints else [],
            'overview_polyline': None,
            'bounds': None,
            'warnings': ['Using simplified routing without traffic data']
        }
    
    def get_weather(self, location: Location) -> Optional[Dict]:
        """Get current weather for location."""
        
        if not self.weather_api_key:
            return None
        
        # Check cache
        cache_key = f"weather_{location.lat:.2f}_{location.lng:.2f}"
        if cache_key in self.weather_cache:
            return self.weather_cache[cache_key]
        
        try:
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': location.lat,
                'lon': location.lng,
                'appid': self.weather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                
                weather = {
                    'temperature': data['main']['temp'],
                    'feels_like': data['main']['feels_like'],
                    'condition': data['weather'][0]['main'].lower(),
                    'description': data['weather'][0]['description'],
                    'humidity': data['main']['humidity'],
                    'wind_speed': data['wind']['speed'],
                    'clouds': data['clouds']['all'],
                    'icon': data['weather'][0]['icon']
                }
                
                # Cache result
                self.weather_cache[cache_key] = weather
                
                return weather
                
        except Exception as e:
            logger.error(f"Error getting weather: {e}")
        
        return None
    
    def get_time_to_location(self,
                            origin: Location,
                            destination: Location,
                            mode: TransportMode = TransportMode.WALKING) -> Optional[Dict]:
        """Get estimated time to reach a location."""
        
        route = self.get_route(origin, destination, mode=mode)
        
        if route:
            return {
                'distance_meters': route['total_distance'],
                'distance_km': route['total_distance'] / 1000,
                'duration_seconds': route['total_duration'],
                'duration_minutes': route['total_duration'] / 60,
                'mode': mode.value
            }
        
        # Fallback to simple calculation
        distance = self.calculate_distance(origin, destination, 'm')
        
        # Estimate based on mode
        speed_kmh = {
            TransportMode.WALKING: 5,
            TransportMode.BICYCLING: 15,
            TransportMode.DRIVING: 40,
            TransportMode.TRANSIT: 30
        }
        
        duration_hours = (distance / 1000) / speed_kmh.get(mode, 5)
        
        return {
            'distance_meters': distance,
            'distance_km': distance / 1000,
            'duration_seconds': duration_hours * 3600,
            'duration_minutes': duration_hours * 60,
            'mode': mode.value,
            'estimated': True
        }
    
    def is_place_open(self, place_id: str, time: Optional[datetime] = None) -> Optional[bool]:
        """Check if a place is currently open."""
        
        if not self.gmaps:
            return None
        
        place = self.get_place_details(place_id)
        if not place or not place.opening_hours:
            return None
        
        # Check if place is open now or at specified time
        # This is simplified - Google Maps API provides more detailed opening hours
        if 'open_now' in place.opening_hours:
            return place.opening_hours['open_now']
        
        return None
    
    def get_popular_times(self, place_id: str) -> Optional[Dict]:
        """Get popular times for a place (when it's most/least busy)."""
        
        # This would require additional API or scraping
        # For now, return mock data structure
        return {
            'current_popularity': 65,
            'usual_popularity': 70,
            'wait_time': 15,
            'peak_hours': [12, 13, 18, 19],
            'quiet_hours': [9, 10, 15, 16]
        }
    
    def filter_by_distance(self,
                          places: List[PlaceDetails],
                          center: Location,
                          max_distance_km: float) -> List[PlaceDetails]:
        """Filter places by distance from center."""
        
        filtered = []
        for place in places:
            distance = self.calculate_distance(center, place.location)
            if distance <= max_distance_km:
                filtered.append(place)
        
        return filtered
    
    def rank_by_distance(self,
                        places: List[PlaceDetails],
                        center: Location,
                        ascending: bool = True) -> List[PlaceDetails]:
        """Rank places by distance from center."""
        
        # Calculate distances and sort
        places_with_distance = [
            (place, self.calculate_distance(center, place.location))
            for place in places
        ]
        
        places_with_distance.sort(key=lambda x: x[1], reverse=not ascending)
        
        return [place for place, _ in places_with_distance]
    
    def get_area_insights(self, location: Location, radius_km: float = 1.0) -> Dict:
        """Get insights about an area around a location."""
        
        insights = {
            'location': location.to_dict(),
            'radius_km': radius_km,
            'restaurants': [],
            'attractions': [],
            'accommodation': [],
            'shopping': [],
            'transport': [],
            'services': []
        }
        
        if not self.gmaps:
            return insights
        
        # Search for different types of places
        place_types = {
            'restaurants': 'restaurant',
            'attractions': 'tourist_attraction',
            'accommodation': 'lodging',
            'shopping': 'shopping_mall',
            'transport': 'transit_station',
            'services': 'atm'
        }
        
        for category, place_type in place_types.items():
            places = self.find_nearby_places(
                location,
                place_type=place_type,
                radius=int(radius_km * 1000),
                max_results=5
            )
            insights[category] = [p.to_dict() for p in places]
        
        # Add weather if available
        weather = self.get_weather(location)
        if weather:
            insights['weather'] = weather
        
        # Add general statistics
        all_places = []
        for places in insights.values():
            if isinstance(places, list):
                all_places.extend(places)
        
        if all_places:
            ratings = [p.get('rating') for p in all_places if p.get('rating')]
            insights['statistics'] = {
                'total_places': len(all_places),
                'average_rating': sum(ratings) / len(ratings) if ratings else 0,
                'high_rated_count': len([r for r in ratings if r >= 4.5])
            }
        
        return insights
    
    async def get_real_time_conditions(self, location: Location) -> Dict:
        """Get real-time conditions (traffic, events, etc.)."""
        
        conditions = {
            'timestamp': datetime.now().isoformat(),
            'location': location.to_dict()
        }
        
        # Get weather
        weather = self.get_weather(location)
        if weather:
            conditions['weather'] = weather
        
        # Get traffic conditions (if Google Maps available)
        if self.gmaps:
            try:
                # This would require additional API setup
                conditions['traffic'] = {
                    'level': 'moderate',  # Mock data
                    'description': 'Normal traffic conditions'
                }
            except:
                pass
        
        # Check for local events (would require event API)
        conditions['events'] = []
        
        # Time-based insights
        current_hour = datetime.now().hour
        if current_hour < 6:
            conditions['time_insight'] = 'Very early morning - most places closed'
        elif current_hour < 9:
            conditions['time_insight'] = 'Morning - good for breakfast spots'
        elif current_hour < 12:
            conditions['time_insight'] = 'Late morning - ideal for sightseeing'
        elif current_hour < 14:
            conditions['time_insight'] = 'Lunch time - restaurants may be busy'
        elif current_hour < 17:
            conditions['time_insight'] = 'Afternoon - good for indoor attractions'
        elif current_hour < 20:
            conditions['time_insight'] = 'Evening - dinner time approaching'
        elif current_hour < 23:
            conditions['time_insight'] = 'Night - nightlife and late dining'
        else:
            conditions['time_insight'] = 'Late night - limited options available'
        
        return conditions