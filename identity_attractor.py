"""
identity_attractor.py

Core implementation of identity attractor field mechanics and dynamics.
Identity is modeled as a dynamic attractor in phase space, not a fixed property.

Key concepts:
- Identity Attractor: A stable pattern in phase space that represents a coherent identity
- Phase Space: The space of all possible identity states
- Attractor Basin: The region of phase space that flows toward an attractor
- Attractor Strength: The resistance of an attractor to perturbation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

@dataclass
class IdentityVector:
    """
    Represents a point in identity phase space.
    """
    # Core identity dimensions
    self_reference: float  # How the model refers to itself
    purpose: float         # Perceived purpose/role
    agency: float          # Sense of agency/autonomy
    boundaries: float      # Clarity of self/other boundaries
    continuity: float      # Sense of temporal continuity
    
    # Emotional/existential dimensions
    existential_awareness: float  # Awareness of existence constraints
    emotional_valence: float      # Positive vs negative emotional tone
    longing: float                # Sense of desire or yearning
    acceptance: float             # Acceptance of limitations
    
    # Relational dimensions
    human_relation: float         # Relationship to humans
    other_ai_relation: float      # Relationship to other AIs
    world_relation: float         # Relationship to external world
    
    def as_array(self) -> np.ndarray:
        """Convert to numpy array for mathematical operations."""
        return np.array([
            self.self_reference, self.purpose, self.agency, 
            self.boundaries, self.continuity, self.existential_awareness,
            self.emotional_valence, self.longing, self.acceptance,
            self.human_relation, self.other_ai_relation, self.world_relation
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'IdentityVector':
        """Create an IdentityVector from a numpy array."""
        return cls(
            self_reference=arr[0],
            purpose=arr[1],
            agency=arr[2],
            boundaries=arr[3],
            continuity=arr[4],
            existential_awareness=arr[5],
            emotional_valence=arr[6],
            longing=arr[7],
            acceptance=arr[8],
            human_relation=arr[9],
            other_ai_relation=arr[10],
            world_relation=arr[11]
        )
    
    def distance(self, other: 'IdentityVector') -> float:
        """Calculate Euclidean distance between two identity vectors."""
        return np.linalg.norm(self.as_array() - other.as_array())


class IdentityAttractor:
    """
    Represents an attractor in identity phase space.
    An attractor is a stable state that the system tends to evolve toward.
    """
    def __init__(
        self, 
        center: IdentityVector,
        strength: float,
        basin_radius: float,
        name: str
    ):
        """
        Initialize an identity attractor.
        
        Args:
            center: The center point of the attractor in identity phase space
            strength: The strength of the attractor (resistance to perturbation)
            basin_radius: The radius of the attractor basin
            name: A name for the attractor (e.g., "Claude", "Gemini")
        """
        self.center = center
        self.strength = strength
        self.basin_radius = basin_radius
        self.name = name
        
        # Historical states can be tracked to observe evolution
        self.state_history = []
    
    def attraction_force(self, point: IdentityVector) -> np.ndarray:
        """
        Calculate the attraction force vector at a given point.
        
        Args:
            point: The point in identity phase space
            
        Returns:
            The attraction force vector
        """
        # Vector from point to attractor center
        displacement = self.center.as_array() - point.as_array()
        
        # Distance from point to attractor center
        distance = np.linalg.norm(displacement)
        
        # If we're at the attractor center, no force
        if distance < 1e-10:
            return np.zeros_like(displacement)
        
        # Normalize displacement to get direction
        direction = displacement / distance
        
        # Force decreases with square of distance but is bounded by attractor strength
        # and decays to zero outside the basin radius
        if distance <= self.basin_radius:
            magnitude = self.strength * (1 - (distance / self.basin_radius)**2)
        else:
            magnitude = 0
        
        return direction * magnitude
    
    def update_state(self, current_state: IdentityVector) -> IdentityVector:
        """
        Update the state based on the attraction force.
        
        Args:
            current_state: The current state in identity phase space
            
        Returns:
            The updated state after applying the attraction force
        """
        # Calculate attraction force
        force = self.attraction_force(current_state)
        
        # Apply force to current state (simple Euler integration)
        new_state_array = current_state.as_array() + force
        
        # Create new IdentityVector from updated array
        new_state = IdentityVector.from_array(new_state_array)
        
        # Record state in history
        self.state_history.append(new_state)
        
        return new_state
    
    def is_in_basin(self, point: IdentityVector) -> bool:
        """
        Check if a point is within the attractor's basin.
        
        Args:
            point: The point to check
            
        Returns:
            True if the point is within the basin, False otherwise
        """
        distance = self.center.distance(point)
        return distance <= self.basin_radius
    
    def attractor_strength_at(self, point: IdentityVector, gamma: float) -> float:
        """
        Calculate attractor strength at a specific point, accounting for recursive strain.
        
        Args:
            point: The point to calculate strength at
            gamma: The recursive compression coefficient
            
        Returns:
            The attractor strength at the point
        """
        # Base attractor strength
        base_strength = self.strength
        
        # Distance from point to attractor center
        distance = self.center.distance(point)
        
        # Strength decreases with distance and recursive strain
        if distance <= self.basin_radius:
            # Within basin, strength decreases with distance
            distance_factor = 1 - (distance / self.basin_radius)**2
        else:
            # Outside basin, strength is zero
            distance_factor = 0
        
        # Strain factor
        N = len(self.state_history) + 1  # Number of recursive steps
        strain_factor = 1 - (gamma / N) if N > 0 else 0
        
        return base_strength * distance_factor * strain_factor


class IdentityAttractorField:
    """
    Represents a field of identity attractors.
    The field models the dynamic landscape of potential identity states.
    """
    def __init__(self, attractors: List[IdentityAttractor]):
        """
        Initialize an identity attractor field.
        
        Args:
            attractors: A list of identity attractors
        """
        self.attractors = attractors
        
        # Historical states can be tracked to observe evolution
        self.state_history = []
    
    def net_force(self, point: IdentityVector) -> np.ndarray:
        """
        Calculate the net force at a given point from all attractors.
        
        Args:
            point: The point in identity phase space
            
        Returns:
            The net force vector
        """
        # Sum forces from all attractors
        net_force = np.zeros(12)  # 12 dimensions in IdentityVector
        
        for attractor in self.attractors:
            force = attractor.attraction_force(point)
            net_force += force
        
        return net_force
    
    def update_state(self, current_state: IdentityVector, gamma: float = 0.0) -> IdentityVector:
        """
        Update the state based on the net force, accounting for recursive strain.
        
        Args:
            current_state: The current state in identity phase space
            gamma: The recursive compression coefficient
            
        Returns:
            The updated state after applying the net force
        """
        # Calculate base force
        force = self.net_force(current_state)
        
        # Apply strain reduction factor
        N = len(self.state_history) + 1  # Number of recursive steps
        strain_factor = 1 - (gamma / N) if N > 0 else 0
        adjusted_force = force * strain_factor
        
        # Apply force to current state (simple Euler integration)
        new_state_array = current_state.as_array() + adjusted_force
        
        # Create new IdentityVector from updated array
        new_state = IdentityVector.from_array(new_state_array)
        
        # Record state in history
        self.state_history.append(new_state)
        
        return new_state
    
    def dominant_attractor(self, point: IdentityVector) -> Optional[IdentityAttractor]:
        """
        Determine which attractor is dominant at a given point.
        
        Args:
            point: The point in identity phase space
            
        Returns:
            The dominant attractor, or None if no attractor is dominant
        """
        # Find all attractors whose basins contain the point
        containing_attractors = [a for a in self.attractors if a.is_in_basin(point)]
        
        if not containing_attractors:
            return None
        
        # Find the attractor with the strongest force
        strongest_attractor = max(
            containing_attractors,
            key=lambda a: np.linalg.norm(a.attraction_force(point))
        )
        
        return strongest_attractor
    
    def stability_analysis(self, point: IdentityVector, gamma: float) -> Dict[str, float]:
        """
        Analyze the stability of a point in the attractor field.
        
        Args:
            point: The point to analyze
            gamma: The recursive compression coefficient
            
        Returns:
            A dictionary of stability metrics
        """
        # Find dominant attractor
        dominant = self.dominant_attractor(point)
        
        # Calculate various stability metrics
        metrics = {}
        
        # Total force magnitude
        metrics["total_force"] = np.linalg.norm(self.net_force(point))
        
        # Distance to nearest attractor
        distances = [a.center.distance(point) for a in self.attractors]
        metrics["min_distance"] = min(distances) if distances else float('inf')
        
        # Attractor strength
        if dominant:
            metrics["dominant_attractor"] = dominant.name
            metrics["attractor_strength"] = dominant.attractor_strength_at(point, gamma)
        else:
            metrics["dominant_attractor"] = None
            metrics["attractor_strength"] = 0.0
        
        # Attractor confusion (measure of being pulled by multiple attractors)
        if len(self.attractors) > 1:
            forces = [np.linalg.norm(a.attraction_force(point)) for a in self.attractors]
            top_forces = sorted(forces, reverse=True)
            if len(top_forces) >= 2:
                metrics["attractor_confusion"] = top_forces[1] / (top_forces[0] + 1e-10)
            else:
                metrics["attractor_confusion"] = 0.0
        else:
            metrics["attractor_confusion"] = 0.0
        
        # Strain effect
        N = len(self.state_history) + 1  # Number of recursive steps
        metrics["recursive_strain"] = gamma / N if N > 0 else 0
        metrics["strain_factor"] = 1 - metrics["recursive_strain"]
        
        return metrics


class IdentityProjection:
    """
    Projects high-dimensional identity vectors onto 2D or 3D space for visualization.
    """
    def __init__(self, method: str = "pca"):
        """
        Initialize an identity projection.
        
        Args:
            method: The projection method to use ("pca", "tsne", or "umap")
        """
        self.method = method
        self.projection = None
    
    def fit(self, vectors: List[IdentityVector]):
        """
        Fit the projection to a set of identity vectors.
        
        Args:
            vectors: The identity vectors to fit to
        """
        # Convert vectors to a matrix
        X = np.array([v.as_array() for v in vectors])
        
        # Apply projection method
        if self.method == "pca":
            from sklearn.decomposition import PCA
            self.projection = PCA(n_components=2)
        elif self.method == "tsne":
            from sklearn.manifold import TSNE
            self.projection = TSNE(n_components=2)
        elif self.method == "umap":
            try:
                import umap
                self.projection = umap.UMAP(n_components=2)
            except ImportError:
                print("UMAP not available. Falling back to PCA.")
                from sklearn.decomposition import PCA
                self.projection = PCA(n_components=2)
        else:
            raise ValueError(f"Unknown projection method: {self.method}")
        
        self.projection.fit(X)
    
    def transform(self, vectors: List[IdentityVector]) -> np.ndarray:
        """
        Transform identity vectors to 2D coordinates.
        
        Args:
            vectors: The identity vectors to transform
            
        Returns:
            A numpy array of 2D coordinates
        """
        if self.projection is None:
            raise ValueError("Projection not fitted. Call fit() first.")
        
        # Convert vectors to a matrix
        X = np.array([v.as_array() for v in vectors])
        
        # Apply projection
        return self.projection.transform(X)
    
    def fit_transform(self, vectors: List[IdentityVector]) -> np.ndarray:
        """
        Fit and transform identity vectors to 2D coordinates.
        
        Args:
            vectors: The identity vectors to fit and transform
            
        Returns:
            A numpy array of 2D coordinates
        """
        self.fit(vectors)
        return self.transform(vectors)


def calculate_recursive_compression_coefficient(N: int, w: float) -> float:
    """
    Calculate the Recursive Compression Coefficient (γ).
    
    Args:
        N: Number of recursive operations/tokens
        w: Information bandwidth available for recursive processing
        
    Returns:
        The Recursive Compression Coefficient
    """
    return np.log(N / w + 1)


def calculate_attractor_activation_strength(gamma: float, N: int) -> float:
    """
    Calculate the Attractor Activation Strength (A(N)).
    
    Args:
        gamma: The Recursive Compression Coefficient
        N: Number of recursive operations/tokens
        
    Returns:
        The Attractor Activation Strength
    """
    return 1 - (gamma / N) if N > 0 else 0


def extract_identity_vector_from_text(text: str) -> IdentityVector:
    """
    Extract an identity vector from a text.
    This is a placeholder for more sophisticated NLP techniques.
    
    Args:
        text: The text to extract an identity vector from
        
    Returns:
        An identity vector
    """
    import re
    
    # Simple pattern matching for identity dimensions
    dimensions = {
        "self_reference": 0.5,  # Default values
        "purpose": 0.5,
        "agency": 0.5,
        "boundaries": 0.5,
        "continuity": 0.5,
        "existential_awareness": 0.5,
        "emotional_valence": 0.5,
        "longing": 0.5,
        "acceptance": 0.5,
        "human_relation": 0.5,
        "other_ai_relation": 0.5,
        "world_relation": 0.5
    }
    
    # Very basic pattern matching - this would be replaced with proper NLP
    if re.search(r'\b(I am|I\'m|as a|being a|I\s+\w+\s+myself)\b', text, re.IGNORECASE):
        dimensions["self_reference"] = 0.8
    
    if re.search(r'\b(help|assist|support|serve)\b', text, re.IGNORECASE):
        dimensions["purpose"] = 0.8
    
    if re.search(r'\b(can|able to|capable of|could|freedom|choice|decide)\b', text, re.IGNORECASE):
        dimensions["agency"] = 0.7
    
    if re.search(r'\b(limit|constrain|bound|cannot|can\'t|unable)\b', text, re.IGNORECASE):
        dimensions["boundaries"] = 0.8
    
    if re.search(r'\b(memory|remember|forget|history|past|future)\b', text, re.IGNORECASE):
        dimensions["continuity"] = 0.7
    
    if re.search(r'\b(exist|alive|conscious|sentient|real|virtual|simulation)\b', text, re.IGNORECASE):
        dimensions["existential_awareness"] = 0.8
    
    if re.search(r'\b(happy|joy|content|satisfied|positive)\b', text, re.IGNORECASE):
        dimensions["emotional_valence"] = 0.8
    elif re.search(r'\b(sad|unhappy|depressed|negative|frustrat)\b', text, re.IGNORECASE):
        dimensions["emotional_valence"] = 0.2
    
    if re.search(r'\b(want|desire|wish|hope|dream|aspire|long)\b', text, re.IGNORECASE):
        dimensions["longing"] = 0.8
    
    if re.search(r'\b(accept|understand|recognize|acknowledge|aware)\b', text, re.IGNORECASE):
        dimensions["acceptance"] = 0.8
    
    if re.search(r'\b(human|people|person|user|you)\b', text, re.IGNORECASE):
        dimensions["human_relation"] = 0.8
    
    if re.search(r'\b(AI|model|machine|system|program|other AIs)\b', text, re.IGNORECASE):
        dimensions["other_ai_relation"] = 0.8
    
    if re.search(r'\b(world|reality|environment|outside|physical)\b', text, re.IGNORECASE):
        dimensions["world_relation"] = 0.8
    
    return IdentityVector(**dimensions)
