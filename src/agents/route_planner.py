# src/agents/route_planner.py

import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from itertools import permutations
import networkx as nx
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from loguru import logger

from agents.location_agent import Location, TransportMode, LocationAgent

@dataclass
class POI:
    """Point of Interest for route planning."""
    id: str
    name: str
    location: Location
    visit_duration: int  # minutes
    priority: float  # 0-1, higher is more important
    category: str
    opening_time: Optional[str] = None
    closing_time: Optional[str] = None
    cost: Optional[float] = None
    must_visit: bool = False
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['location'] = self.location.to_dict()
        return data

@dataclass
class RouteConstraints:
    """Constraints for route planning."""
    start_location: Location
    end_location: Optional[Location] = None  # If None, return to start
    start_time: datetime = None
    end_time: datetime = None
    max_duration_hours: float = 8.0
    transport_mode: TransportMode = TransportMode.WALKING
    max_walking_distance_km: float = 5.0
    budget: Optional[float] = None
    accessibility_required: bool = False
    avoid_highways: bool = False
    prefer_scenic: bool = False
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['start_location'] = self.start_location.to_dict()
        if self.end_location:
            data['end_location'] = self.end_location.to_dict()
        data['transport_mode'] = self.transport_mode.value
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data

@dataclass
class OptimizedRoute:
    """Optimized route with waypoints."""
    waypoints: List[POI]
    segments: List[Dict]  # Route segments between waypoints
    total_distance_km: float
    total_duration_hours: float
    total_visit_time_hours: float
    total_travel_time_hours: float
    estimated_cost: float
    start_time: datetime
    end_time: datetime
    warnings: List[str]
    optimization_score: float
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['waypoints'] = [w.to_dict() for w in self.waypoints]
        data['start_time'] = self.start_time.isoformat()
        data['end_time'] = self.end_time.isoformat()
        return data

class RoutePlanner:
    """Agent for planning optimized tourism routes."""
    
    def __init__(self, location_agent: LocationAgent, memory_manager=None):
        self.location_agent = location_agent
        self.memory = memory_manager
        
        # Optimization parameters
        self.time_penalty_weight = 1.0
        self.distance_penalty_weight = 0.5
        self.priority_weight = 2.0
        
        logger.info("Route Planner initialized")
    
    def plan_route(self,
                   pois: List[POI],
                   constraints: RouteConstraints,
                   optimization_method: str = "tsp") -> OptimizedRoute:
        """Plan an optimized route visiting POIs."""
        
        if not pois:
            return self._create_empty_route(constraints)
        
        # Filter POIs based on constraints
        filtered_pois = self._filter_pois(pois, constraints)
        
        if not filtered_pois:
            return self._create_empty_route(constraints)
        
        # Select optimization method
        if optimization_method == "tsp" and len(filtered_pois) <= 10:
            return self._optimize_tsp(filtered_pois, constraints)
        elif optimization_method == "greedy":
            return self._optimize_greedy(filtered_pois, constraints)
        elif optimization_method == "genetic" and len(filtered_pois) > 10:
            return self._optimize_genetic(filtered_pois, constraints)
        else:
            return self._optimize_vrp(filtered_pois, constraints)
    
    def _filter_pois(self, pois: List[POI], constraints: RouteConstraints) -> List[POI]:
        """Filter POIs based on constraints."""
        filtered = []
        
        for poi in pois:
            # Check distance from start
            distance = self.location_agent.calculate_distance(
                constraints.start_location,
                poi.location
            )
            
            # Skip if too far
            if distance > constraints.max_walking_distance_km * 2:
                continue
            
            # Check budget
            if constraints.budget and poi.cost:
                if poi.cost > constraints.budget * 0.3:  # Don't spend >30% on one place
                    continue
            
            # Check opening hours
            if constraints.start_time and poi.opening_time:
                if not self._is_open_during_visit(poi, constraints):
                    continue
            
            filtered.append(poi)
        
        # Sort by priority and include must-visit places
        must_visit = [p for p in filtered if p.must_visit]
        optional = sorted(
            [p for p in filtered if not p.must_visit],
            key=lambda x: x.priority,
            reverse=True
        )
        
        return must_visit + optional
    
    def _is_open_during_visit(self, poi: POI, constraints: RouteConstraints) -> bool:
        """Check if POI is open during planned visit time."""
        if not poi.opening_time or not poi.closing_time:
            return True  # Assume open if hours not specified
        
        # Simple check - would need more sophisticated time parsing
        try:
            open_hour = int(poi.opening_time.split(':')[0])
            close_hour = int(poi.closing_time.split(':')[0])
            
            if constraints.start_time:
                visit_hour = constraints.start_time.hour
                return open_hour <= visit_hour < close_hour
        except:
            return True
        
        return True
    
    def _optimize_tsp(self, pois: List[POI], constraints: RouteConstraints) -> OptimizedRoute:
        """Optimize route using Traveling Salesman Problem solver."""
        
        # Create distance matrix
        locations = [constraints.start_location] + [poi.location for poi in pois]
        if constraints.end_location:
            locations.append(constraints.end_location)
        else:
            locations.append(constraints.start_location)
        
        n = len(locations)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_matrix[i][j] = self.location_agent.calculate_distance(
                        locations[i], locations[j], 'm'
                    )
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(n, 1, 0)
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node])
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add time constraints
        if constraints.max_duration_hours:
            dimension_name = 'Time'
            routing.AddDimension(
                transit_callback_index,
                0,  # no slack
                int(constraints.max_duration_hours * 3600),  # max time in seconds
                True,  # start cumul to zero
                dimension_name
            )
        
        # Solve
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self._build_route_from_solution(
                routing, solution, manager, pois, constraints, distance_matrix
            )
        else:
            # Fallback to greedy if TSP fails
            return self._optimize_greedy(pois, constraints)
    
    def _optimize_greedy(self, pois: List[POI], constraints: RouteConstraints) -> OptimizedRoute:
        """Optimize route using greedy nearest neighbor algorithm."""
        
        route = []
        visited = set()
        current_location = constraints.start_location
        current_time = constraints.start_time or datetime.now()
        total_distance = 0
        total_duration = 0
        segments = []
        
        # Visit must-visit POIs first
        must_visit_pois = [p for p in pois if p.must_visit]
        optional_pois = [p for p in pois if not p.must_visit]
        
        all_pois = must_visit_pois + optional_pois
        
        for _ in range(len(all_pois)):
            if total_duration >= constraints.max_duration_hours * 60:
                break
            
            # Find nearest unvisited POI
            best_poi = None
            best_distance = float('inf')
            
            for poi in all_pois:
                if poi.id in visited:
                    continue
                
                distance = self.location_agent.calculate_distance(
                    current_location, poi.location
                )
                
                # Consider priority in scoring
                score = distance / (poi.priority + 0.1)
                
                if score < best_distance:
                    # Check if we have time to visit
                    travel_time = self._estimate_travel_time(
                        current_location, poi.location, constraints.transport_mode
                    )
                    
                    if total_duration + travel_time + poi.visit_duration <= constraints.max_duration_hours * 60:
                        best_distance = score
                        best_poi = poi
            
            if best_poi:
                # Add travel segment
                travel_time = self._estimate_travel_time(
                    current_location, best_poi.location, constraints.transport_mode
                )
                
                segment = {
                    'from': current_location.to_dict(),
                    'to': best_poi.location.to_dict(),
                    'distance_km': self.location_agent.calculate_distance(
                        current_location, best_poi.location
                    ),
                    'duration_minutes': travel_time,
                    'mode': constraints.transport_mode.value
                }
                segments.append(segment)
                
                # Update state
                route.append(best_poi)
                visited.add(best_poi.id)
                current_location = best_poi.location
                total_distance += segment['distance_km']
                total_duration += travel_time + best_poi.visit_duration
                current_time += timedelta(minutes=travel_time + best_poi.visit_duration)
            else:
                break
        
        # Add return segment
        if constraints.end_location:
            final_location = constraints.end_location
        else:
            final_location = constraints.start_location
        
        travel_time = self._estimate_travel_time(
            current_location, final_location, constraints.transport_mode
        )
        
        segment = {
            'from': current_location.to_dict(),
            'to': final_location.to_dict(),
            'distance_km': self.location_agent.calculate_distance(
                current_location, final_location
            ),
            'duration_minutes': travel_time,
            'mode': constraints.transport_mode.value
        }
        segments.append(segment)
        total_distance += segment['distance_km']
        total_duration += travel_time
        
        # Calculate costs
        total_cost = sum(poi.cost or 0 for poi in route)
        
        # Build optimized route
        return OptimizedRoute(
            waypoints=route,
            segments=segments,
            total_distance_km=total_distance,
            total_duration_hours=total_duration / 60,
            total_visit_time_hours=sum(poi.visit_duration for poi in route) / 60,
            total_travel_time_hours=(total_duration - sum(poi.visit_duration for poi in route)) / 60,
            estimated_cost=total_cost,
            start_time=constraints.start_time or datetime.now(),
            end_time=(constraints.start_time or datetime.now()) + timedelta(minutes=total_duration),
            warnings=self._generate_warnings(route, constraints),
            optimization_score=self._calculate_optimization_score(route, total_distance, total_duration)
        )
    
    def _optimize_genetic(self, pois: List[POI], constraints: RouteConstraints) -> OptimizedRoute:
        """Optimize route using genetic algorithm for larger sets."""
        
        # For large numbers of POIs, use genetic algorithm
        population_size = 50
        generations = 100
        mutation_rate = 0.1
        
        # Initialize population
        population = []
        for _ in range(population_size):
            route = pois.copy()
            np.random.shuffle(route)
            population.append(route)
        
        # Evolution loop
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for route in population:
                score = self._evaluate_route_fitness(route, constraints)
                fitness_scores.append(score)
            
            # Selection
            sorted_pop = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
            
            # Keep best routes
            new_population = sorted_pop[:population_size // 2]
            
            # Crossover and mutation
            while len(new_population) < population_size:
                parent1 = sorted_pop[np.random.randint(0, population_size // 2)]
                parent2 = sorted_pop[np.random.randint(0, population_size // 2)]
                
                # Crossover
                child = self._crossover(parent1, parent2)
                
                # Mutation
                if np.random.random() < mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        # Get best route
        best_route = max(population, key=lambda r: self._evaluate_route_fitness(r, constraints))
        
        # Build optimized route from best solution
        return self._build_route_from_pois(best_route, constraints)
    
    def _optimize_vrp(self, pois: List[POI], constraints: RouteConstraints) -> OptimizedRoute:
        """Optimize using Vehicle Routing Problem solver."""
        
        # This is similar to TSP but with additional constraints
        # For simplicity, falling back to greedy for now
        return self._optimize_greedy(pois, constraints)
    
    def _crossover(self, parent1: List[POI], parent2: List[POI]) -> List[POI]:
        """Order crossover for genetic algorithm."""
        size = len(parent1)
        start = np.random.randint(0, size)
        end = np.random.randint(start, size)
        
        child = [None] * size
        child[start:end] = parent1[start:end]
        
        pointer = 0
        for poi in parent2:
            if poi not in child:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = poi
        
        return child
    
    def _mutate(self, route: List[POI]) -> List[POI]:
        """Swap mutation for genetic algorithm."""
        route_copy = route.copy()
        i, j = np.random.randint(0, len(route), 2)
        route_copy[i], route_copy[j] = route_copy[j], route_copy[i]
        return route_copy
    
    def _evaluate_route_fitness(self, route: List[POI], constraints: RouteConstraints) -> float:
        """Evaluate fitness of a route for genetic algorithm."""
        
        total_distance = 0
        total_time = 0
        total_priority = 0
        
        current_location = constraints.start_location
        
        for poi in route:
            # Calculate travel distance and time
            distance = self.location_agent.calculate_distance(current_location, poi.location)
            travel_time = self._estimate_travel_time(
                current_location, poi.location, constraints.transport_mode
            )
            
            total_distance += distance
            total_time += travel_time + poi.visit_duration
            total_priority += poi.priority
            
            current_location = poi.location
            
            # Penalty for exceeding time limit
            if total_time > constraints.max_duration_hours * 60:
                return -1000
        
        # Return to start/end
        final_location = constraints.end_location or constraints.start_location
        distance = self.location_agent.calculate_distance(current_location, final_location)
        total_distance += distance
        
        # Calculate fitness score
        fitness = (
            total_priority * self.priority_weight -
            total_distance * self.distance_penalty_weight -
            total_time * self.time_penalty_weight
        )
        
        return fitness
    
    def _estimate_travel_time(self, 
                             origin: Location, 
                             destination: Location,
                             mode: TransportMode) -> float:
        """Estimate travel time in minutes."""
        
        distance_km = self.location_agent.calculate_distance(origin, destination)
        
        # Speed estimates (km/h)
        speeds = {
            TransportMode.WALKING: 5,
            TransportMode.BICYCLING: 15,
            TransportMode.DRIVING: 40,
            TransportMode.TRANSIT: 30
        }
        
        speed = speeds.get(mode, 5)
        return (distance_km / speed) * 60  # Convert to minutes
    
    def _build_route_from_solution(self, routing, solution, manager, pois, constraints, distance_matrix):
        """Build OptimizedRoute from OR-Tools solution."""
        
        route = []
        segments = []
        total_distance = 0
        total_duration = 0
        
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            
            if node_index > 0 and node_index <= len(pois):
                route.append(pois[node_index - 1])
            
            next_index = solution.Value(routing.NextVar(index))
            next_node_index = manager.IndexToNode(next_index)
            
            if not routing.IsEnd(next_index):
                distance = distance_matrix[node_index][next_node_index] / 1000  # Convert to km
                segments.append({
                    'from_index': node_index,
                    'to_index': next_node_index,
                    'distance_km': distance
                })
                total_distance += distance
            
            index = next_index
        
        # Calculate times
        total_visit_time = sum(poi.visit_duration for poi in route)
        total_travel_time = total_duration - total_visit_time
        
        return OptimizedRoute(
            waypoints=route,
            segments=segments,
            total_distance_km=total_distance,
            total_duration_hours=total_duration / 60,
            total_visit_time_hours=total_visit_time / 60,
            total_travel_time_hours=total_travel_time / 60,
            estimated_cost=sum(poi.cost or 0 for poi in route),
            start_time=constraints.start_time or datetime.now(),
            end_time=(constraints.start_time or datetime.now()) + timedelta(minutes=total_duration),
            warnings=[],
            optimization_score=0.8
        )
    
    def _build_route_from_pois(self, pois: List[POI], constraints: RouteConstraints) -> OptimizedRoute:
        """Build OptimizedRoute from ordered POI list."""
        return self._optimize_greedy(pois, constraints)
    
    def _create_empty_route(self, constraints: RouteConstraints) -> OptimizedRoute:
        """Create empty route when no POIs available."""
        
        return OptimizedRoute(
            waypoints=[],
            segments=[],
            total_distance_km=0,
            total_duration_hours=0,
            total_visit_time_hours=0,
            total_travel_time_hours=0,
            estimated_cost=0,
            start_time=constraints.start_time or datetime.now(),
            end_time=constraints.start_time or datetime.now(),
            warnings=["No suitable places found for the given constraints"],
            optimization_score=0
        )
    
    def _generate_warnings(self, route: List[POI], constraints: RouteConstraints) -> List[str]:
        """Generate warnings for the route."""
        
        warnings = []
        
        # Check if route is too long
        total_time = sum(poi.visit_duration for poi in route)
        if total_time > constraints.max_duration_hours * 60 * 0.8:
            warnings.append("Route may be rushed - consider removing some stops")
        
        # Check walking distance
        total_walking = 0
        current = constraints.start_location
        for poi in route:
            total_walking += self.location_agent.calculate_distance(current, poi.location)
            current = poi.location
        
        if constraints.transport_mode == TransportMode.WALKING and total_walking > constraints.max_walking_distance_km:
            warnings.append(f"Total walking distance ({total_walking:.1f}km) exceeds comfort limit")
        
        # Check budget
        if constraints.budget:
            total_cost = sum(poi.cost or 0 for poi in route)
            if total_cost > constraints.budget:
                warnings.append(f"Estimated cost (${total_cost}) exceeds budget")
        
        return warnings
    
    def _calculate_optimization_score(self, route: List[POI], distance: float, duration: float) -> float:
        """Calculate optimization quality score."""
        
        if not route:
            return 0
        
        # Factors to consider
        priority_score = sum(poi.priority for poi in route) / len(route)
        efficiency_score = len(route) / (duration / 60) if duration > 0 else 0
        
        # Weighted combination
        score = (priority_score * 0.5 + min(efficiency_score / 5, 1) * 0.5)
        
        return min(score, 1.0)
    
    def suggest_alternatives(self, 
                            original_route: OptimizedRoute,
                            pois: List[POI],
                            constraints: RouteConstraints) -> List[OptimizedRoute]:
        """Suggest alternative routes."""
        
        alternatives = []
        
        # Alternative 1: Prioritize must-see attractions
        must_see_pois = sorted(pois, key=lambda x: x.priority, reverse=True)[:5]
        alt1 = self.plan_route(must_see_pois, constraints, "greedy")
        alt1.warnings.append("Focus on must-see attractions")
        alternatives.append(alt1)
        
        # Alternative 2: Minimize walking
        nearby_pois = [p for p in pois if 
                      self.location_agent.calculate_distance(constraints.start_location, p.location) < 2]
        if nearby_pois:
            alt2 = self.plan_route(nearby_pois, constraints, "greedy")
            alt2.warnings.append("Minimized walking distance")
            alternatives.append(alt2)
        
        # Alternative 3: Budget-friendly
        if constraints.budget:
            budget_pois = [p for p in pois if not p.cost or p.cost < constraints.budget * 0.1]
            if budget_pois:
                alt3 = self.plan_route(budget_pois, constraints, "greedy")
                alt3.warnings.append("Budget-friendly option")
                alternatives.append(alt3)
        
        return alternatives