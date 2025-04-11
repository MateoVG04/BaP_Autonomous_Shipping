import os
import json
import heapq
import numpy as np
from shapely.geometry import Point, Polygon


def create_checkpoints_from_simple_path(points, max_distance):
    """
    Create checkpoints from a path ensuring no two consecutive points are further than max_distance apart.
    
    Args:
        points (list/array): List or numpy array of points defining the path.
        max_distance (float): Maximum allowed distance between consecutive checkpoints.

    Returns:
        list of tuples: A list of checkpoint coordinates with appropriate spacing.
    """
    # Convert to numpy array if not already
    points = np.array(points)
    
    # Check if array is empty
    if points.size == 0:
        return []
    
    final_checkpoints = [tuple(points[0])]  # Start with the first point
    
    for i in range(1, len(points)):
        current_point = np.array(final_checkpoints[-1])
        next_point = points[i]
        
        # Calculate vector between points
        segment_vector = next_point - current_point
        segment_length = np.linalg.norm(segment_vector)
        
        if segment_length > max_distance:
            # Calculate how many intermediate points we need
            num_segments = int(np.ceil(segment_length / max_distance))
            
            # Create evenly spaced intermediate points
            for j in range(1, num_segments):
                fraction = j / num_segments
                intermediate_point = current_point + fraction * segment_vector
                final_checkpoints.append(tuple(intermediate_point))
        
        # Add the next point
        final_checkpoints.append(tuple(next_point))
    
    return final_checkpoints

def is_ship_heading_correct(checkpoint, state):
    """
    Check if the ship's heading will cross the line segment formed by the perpendicular line.
    """
    # Ship's position and heading
    ship_position = np.array(state[:2])  # Assuming state contains [x, y, heading]
    heading = state[2]  # Heading in radians

    # Heading direction vector
    heading_vector = np.array([np.cos(heading), np.sin(heading)])

    # Extend the ship's trajectory into a ray
    ray_start = ship_position
    ray_end = ray_start + heading_vector * 10000  # Large number to create a long ray

    # Line segment points
    line_point1 = np.array(checkpoint['perpendicular_line'][0])
    line_point2 = np.array(checkpoint['perpendicular_line'][1])

    # Function to compute intersection of two line segments
    def line_intersection(p1, p2, q1, q2):
        """Find the intersection of two line segments if it exists."""
        r = p2 - p1
        s = q2 - q1

        r_cross_s = np.cross(r, s)
        q_minus_p = q1 - p1

        if r_cross_s == 0:
            return None  # Lines are parallel and non-intersecting

        t = np.cross(q_minus_p, s) / r_cross_s
        u = np.cross(q_minus_p, r) / r_cross_s

        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection = p1 + t * r
            return intersection
        return None

    # Check for intersection
    intersection = line_intersection(ray_start, ray_end, line_point1, line_point2)

    if intersection is not None:
        # Check if the intersection is in the forward direction of the ship
        intersection_vector = intersection - ray_start
        if np.dot(intersection_vector, heading_vector) > 0:
            return True  # Heading crosses the line in the forward direction
    return False  # No intersection or in the wrong direction


def check_collision_ship(ship_position, polygon):
    """
    Check if the ship makes contact with any edge of the polygon.
    """
    ship_point = Point(ship_position)
    polygon_shape = Polygon(polygon)

    # Check if the ship position intersects the boundary of the polygon
    if polygon_shape.contains(ship_point):
        return True
    return False

