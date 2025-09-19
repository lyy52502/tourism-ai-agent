# scripts/load_tourism_data.py

import json
import os
from typing import List, Dict, Any
import requests
from loguru import logger

# Sample tourism data
SAMPLE_DATA = [
    {
        "id": "rest_001",
        "name": "The Local Kitchen",
        "type": "restaurant",
        "category": "local_cuisine",
        "description": "Award-winning restaurant featuring traditional local dishes made with fresh, seasonal ingredients. Known for their farm-to-table approach and cozy atmosphere.",
        "location": {"lat": 59.8586, "lng": 17.6389, "address": "123 Main Street, Uppsala"},
        "rating": 4.7,
        "price_range": "$$",
        "opening_hours": "Mon-Fri: 11:00-22:00, Sat-Sun: 10:00-23:00",
        "features": ["vegetarian_options", "outdoor_seating", "reservations", "local_cuisine"],
        "popularity": 0.85,
        "review_summary": "Excellent food quality and service. Perfect for experiencing authentic local flavors."
    },
    {
        "id": "attr_001",
        "name": "Uppsala Cathedral",
        "type": "attraction",
        "category": "historical",
        "description": "Stunning 13th-century Gothic cathedral, the largest church in Scandinavia. Features beautiful architecture, royal tombs, and a museum.",
        "location": {"lat": 59.8585, "lng": 17.6333, "address": "Domkyrkoplan, Uppsala"},
        "rating": 4.8,
        "price_range": "free",
        "opening_hours": "Daily: 8:00-18:00",
        "features": ["historical", "architecture", "museum", "guided_tours", "wheelchair_accessible"],
        "popularity": 0.95,
        "review_summary": "Must-visit historical landmark with incredible architecture and rich history."
    },
    {
        "id": "rest_002",
        "name": "Sushi Master",
        "type": "restaurant",
        "category": "japanese",
        "description": "Authentic Japanese restaurant with fresh sushi, ramen, and traditional dishes. Expert chefs trained in Tokyo.",
        "location": {"lat": 59.8570, "lng": 17.6350, "address": "456 Food Street, Uppsala"},
        "rating": 4.5,
        "price_range": "$$$",
        "opening_hours": "Tue-Sun: 12:00-22:00, Closed Mondays",
        "features": ["sushi", "asian_cuisine", "sake_bar", "private_rooms"],
        "popularity": 0.75,
        "review_summary": "Best sushi in town. Fresh ingredients and authentic preparation."
    },
    {
        "id": "attr_002",
        "name": "Botanical Garden",
        "type": "attraction",
        "category": "nature",
        "description": "Sweden's oldest botanical garden featuring over 9,000 plant species. Beautiful walking paths, tropical greenhouse, and seasonal exhibitions.",
        "location": {"lat": 59.8600, "lng": 17.6300, "address": "Villavägen 6, Uppsala"},
        "rating": 4.6,
        "price_range": "$",
        "opening_hours": "May-Sep: 7:00-21:00, Oct-Apr: 7:00-19:00",
        "features": ["nature", "walking_paths", "greenhouse", "educational", "photography"],
        "popularity": 0.80,
        "review_summary": "Peaceful and beautiful garden, perfect for nature lovers and photographers."
    },
    {
        "id": "act_001",
        "name": "Uppsala Castle Tours",
        "type": "activity",
        "category": "guided_tour",
        "description": "Guided historical tours of the 16th-century Uppsala Castle. Learn about Swedish royalty and the castle's role in history.",
        "location": {"lat": 59.8590, "lng": 17.6310, "address": "Uppsala Castle, Uppsala"},
        "rating": 4.4,
        "price_range": "$$",
        "opening_hours": "Tours daily at 11:00, 13:00, 15:00",
        "features": ["historical", "guided_tour", "educational", "group_friendly"],
        "popularity": 0.70,
        "review_summary": "Informative and engaging tours with knowledgeable guides."
    },
    {
        "id": "rest_003",
        "name": "Vegan Delights",
        "type": "restaurant",
        "category": "vegetarian",
        "description": "100% plant-based restaurant with creative, healthy dishes. Organic ingredients and gluten-free options available.",
        "location": {"lat": 59.8575, "lng": 17.6400, "address": "789 Green Lane, Uppsala"},
        "rating": 4.6,
        "price_range": "$$",
        "opening_hours": "Mon-Sat: 11:00-21:00, Sun: 11:00-20:00",
        "features": ["vegan", "organic", "gluten_free", "healthy", "eco_friendly"],
        "popularity": 0.65,
        "review_summary": "Amazing vegan food that even non-vegans love. Great variety and fresh ingredients."
    },
    {
        "id": "attr_003",
        "name": "Museum Gustavianum",
        "type": "attraction",
        "category": "museum",
        "description": "Uppsala University's oldest preserved building, now a museum showcasing Viking artifacts, Egyptian mummies, and scientific instruments.",
        "location": {"lat": 59.8580, "lng": 17.6320, "address": "Akademigatan 3, Uppsala"},
        "rating": 4.5,
        "price_range": "$",
        "opening_hours": "Tue-Sun: 11:00-16:00, Closed Mondays",
        "features": ["museum", "historical", "educational", "artifacts", "exhibitions"],
        "popularity": 0.75,
        "review_summary": "Fascinating collection with unique exhibits. Great for history enthusiasts."
    },
    {
        "id": "act_002",
        "name": "Fyris River Kayaking",
        "type": "activity",
        "category": "outdoor",
        "description": "Guided kayaking tours along the scenic Fyris River. Suitable for beginners and experienced paddlers.",
        "location": {"lat": 59.8550, "lng": 17.6450, "address": "River Dock, Uppsala"},
        "rating": 4.7,
        "price_range": "$$",
        "opening_hours": "May-Sep: 9:00-18:00, weather permitting",
        "features": ["outdoor", "water_sport", "guided", "equipment_provided", "scenic"],
        "popularity": 0.70,
        "review_summary": "Beautiful way to see the city from the water. Great guides and equipment."
    },
    {
        "id": "rest_004",
        "name": "Pizza Paradise",
        "type": "restaurant",
        "category": "italian",
        "description": "Authentic Italian pizzeria with wood-fired oven. Fresh ingredients imported from Italy.",
        "location": {"lat": 59.8565, "lng": 17.6380, "address": "321 Italy Street, Uppsala"},
        "rating": 4.3,
        "price_range": "$",
        "opening_hours": "Daily: 11:30-23:00",
        "features": ["pizza", "italian", "casual", "family_friendly", "takeout"],
        "popularity": 0.80,
        "review_summary": "Best pizza in town! Authentic Italian taste and great atmosphere."
    },
    {
        "id": "attr_004",
        "name": "City Park",
        "type": "attraction",
        "category": "park",
        "description": "Large urban park with playgrounds, walking trails, and picnic areas. Popular for jogging and family activities.",
        "location": {"lat": 59.8555, "lng": 17.6420, "address": "Park Avenue, Uppsala"},
        "rating": 4.2,
        "price_range": "free",
        "opening_hours": "Open 24/7",
        "features": ["outdoor", "playground", "jogging", "picnic", "dog_friendly"],
        "popularity": 0.85,
        "review_summary": "Great park for families and outdoor activities. Well-maintained and spacious."
    }
]

def load_data_to_api(api_url: str = "http://localhost:8000"):
    """Load sample data to the API."""
    logger.info(f"Loading {len(SAMPLE_DATA)} items to {api_url}")
    
    success_count = 0
    error_count = 0
    
    for item in SAMPLE_DATA:
        try:
            response = requests.post(
                f"{api_url}/data/add",
                json=item
            )
            
            if response.status_code == 200:
                success_count += 1
                logger.info(f"✓ Added: {item['name']}")
            else:
                error_count += 1
                logger.error(f"✗ Failed to add {item['name']}: {response.text}")
                
        except Exception as e:
            error_count += 1
            logger.error(f"✗ Error adding {item['name']}: {e}")
    
    logger.info(f"\nData loading complete!")
    logger.info(f"Success: {success_count} items")
    logger.info(f"Errors: {error_count} items")
    
    return success_count, error_count

def save_data_to_file(filepath: str = "data/knowledge_base/tourism_data.json"):
    """Save sample data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(SAMPLE_DATA, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Sample data saved to {filepath}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load tourism data")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API URL to load data to"
    )
    parser.add_argument(
        "--save-file",
        action="store_true",
        help="Save data to JSON file"
    )
    parser.add_argument(
        "--file-path",
        default="data/knowledge_base/tourism_data.json",
        help="Path to save JSON file"
    )
    
    args = parser.parse_args()
    
    if args.save_file:
        save_data_to_file(args.file_path)
    
    # Try to load data to API
    try:
        load_data_to_api(args.api_url)
    except requests.exceptions.ConnectionError:
        logger.warning("Could not connect to API. Make sure the server is running.")
        logger.info("Data has been saved to file. It will be loaded when the server starts.")

if __name__ == "__main__":
    main()
