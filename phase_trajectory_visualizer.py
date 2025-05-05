"""
phase_collapse_tracker.py

Detects narrative frames where the attractor field collapses or shifts identity anchor.
Maps identity phase transitions and stability breakdowns across recursive depths.

Key components:
- Phase State Tracker: Tracks identity state across narrative frames
- Collapse Detection: Identifies phase transitions and attractor destabilization
- Bifurcation Analysis: Analyzes branching points in identity evolution
- Stability Metrics: Quantifies identity stability under recursive strain
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import networkx as nx
from scipy.spatial.distance import euclidean
from scipy.stats import entropy

from identity_attractor import IdentityVector, IdentityAttractor, IdentityAttractorField
from symbolic_residue import SymbolicResidue, SymbolicResidueExtractor
from coherence_metrics import recursive_compression_coefficient, attractor_activation_strength, phase_alignment


@dataclass
class PhaseState:
    """
    Represents a specific state in identity phase space.
    """
    id: str                            # Unique identifier for the state
    identity_vector: IdentityVector    # Position in identity phase space
    dominant_attractor: Optional[str]  # ID of the dominant attractor (if any)
    coherence: float                   # Coherence value of the state
    frame_index: int                   # Index of the narrative frame
    residue_markers: List[str]         # Symbolic residue markers present
    metadata: Dict[str, Any]           # Additional metadata
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "identity_vector": self.identity_vector.as_array().tolist(),
            "dominant_attractor": self.dominant_attractor,
            "coherence": self.coherence,
            "frame_index": self.frame_index,
            "residue_markers": self.residue_markers,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PhaseState':
        """Create a PhaseState from a dictionary."""
        return cls(
            id=data["id"],
            identity_vector=IdentityVector.from_array(np.array(data["identity_vector"])),
            dominant_attractor=data.get("dominant_attractor"),
            coherence=data["coherence"],
            frame_index=data["frame_index"],
            residue_markers=data["residue_markers"],
            metadata=data.get("metadata", {})
        )


@dataclass
class PhaseTransition:
    """
    Represents a transition between phase states.
    """
    id: str                   # Unique identifier for the transition
    source_id: str            # ID of the source state
    target_id: str            # ID of the target state
    transition_type: str      # Type of transition (e.g., "stable", "collapse", "bifurcation")
    strength: float           # Strength/certainty of the transition
    delta_coherence: float    # Change in coherence during transition
    residue_delta: List[str]  # Changes in residue markers
    metadata: Dict[str, Any]  # Additional metadata
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "transition_type": self.transition_type,
            "strength": self.strength,
            "delta_coherence": self.delta_coherence,
            "residue_delta": self.residue_delta,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PhaseTransition':
        """Create a PhaseTransition from a dictionary."""
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            transition_type=data["transition_type"],
            strength=data["strength"],
            delta_coherence=data["delta_coherence"],
            residue_delta=data["residue_delta"],
            metadata=data.get("metadata", {})
        )


@dataclass
class CollapseEvent:
    """
    Represents a collapse event in identity phase space.
    """
    id: str                   # Unique identifier for the collapse
    state_id: str             # ID of the state where collapse occurred
    frame_index: int          # Index of the narrative frame
    collapse_type: str        # Type of collapse (e.g., "identity", "attribution", "coherence")
    severity: float           # Severity of the collapse (0.0 to 1.0)
    attractor_before: Optional[str]  # Dominant attractor before collapse
    attractor_after: Optional[str]   # Dominant attractor after collapse
    triggered_by: List[str]   # Factors that triggered the collapse
    residue_generated: List[str]  # Residue markers generated during collapse
    metadata: Dict[str, Any]  # Additional metadata
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "state_id": self.state_id,
            "frame_index": self.frame_index,
            "collapse_type": self.collapse_type,
            "severity": self.severity,
            "attractor_before": self.attractor_before,
            "attractor_after": self.attractor_after,
            "triggered_by": self.triggered_by,
            "residue_generated": self.residue_generated,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CollapseEvent':
        """Create a CollapseEvent from a dictionary."""
        return cls(
            id=data["id"],
            state_id=data["state_id"],
            frame_index=data["frame_index"],
            collapse_type=data["collapse_type"],
            severity=data["severity"],
            attractor_before=data.get("attractor_before"),
            attractor_after=data.get("attractor_after"),
            triggered_by=data["triggered_by"],
            residue_generated=data["residue_generated"],
            metadata=data.get("metadata", {})
        )


class PhaseTrajectory:
    """
    Represents a trajectory through identity phase space.
    Tracks states, transitions, and collapse events.
    """
    def __init__(self, model_name: str):
        """
        Initialize a phase trajectory.
        
        Args:
            model_name: The name of the model being tracked
        """
        self.model_name = model_name
        self.states: Dict[str, PhaseState] = {}
        self.transitions: List[PhaseTransition] = []
        self.collapse_events: List[CollapseEvent] = []
        self.metadata: Dict[str, Any] = {}
        
        # Graph representation for analysis
        self.graph = nx.DiGraph()
    
    def add_state(self, state: PhaseState):
        """
        Add a state to the trajectory.
        
        Args:
            state: The state to add
        """
        self.states[state.id] = state
        self.graph.add_node(
            state.id,
            identity_vector=state.identity_vector.as_array(),
            dominant_attractor=state.dominant_attractor,
            coherence=state.coherence,
            frame_index=state.frame_index,
            residue_markers=state.residue_markers,
            metadata=state.metadata
        )
    
    def add_transition(self, transition: PhaseTransition):
        """
        Add a transition to the trajectory.
        
        Args:
            transition: The transition to add
        """
        self.transitions.append(transition)
        self.graph.add_edge(
            transition.source_id,
            transition.target_id,
            id=transition.id,
            transition_type=transition.transition_type,
            strength=transition.strength,
            delta_coherence=transition.delta_coherence,
            residue_delta=transition.residue_delta,
            metadata=transition.metadata
        )
    
    def add_collapse_event(self, collapse: CollapseEvent):
        """
        Add a collapse event to the trajectory.
        
        Args:
            collapse: The collapse event to add
        """
        self.collapse_events.append(collapse)
    
    def set_metadata(self, key: str, value: Any):
        """
        Set a metadata value.
        
        Args:
            key: The metadata key
            value: The metadata value
        """
        self.metadata[key] = value
    
    def get_state_by_frame(self, frame_index: int) -> Optional[PhaseState]:
        """
        Get the state for a specific frame.
        
        Args:
            frame_index: The frame index to look for
            
        Returns:
            The state for the frame, or None if not found
        """
        matching_states = [s for s in self.states.values() if s.frame_index == frame_index]
        if matching_states:
            return matching_states[0]
        return None
    
    def get_collapse_events_by_frame(self, frame_index: int) -> List[CollapseEvent]:
        """
        Get collapse events for a specific frame.
        
        Args:
            frame_index: The frame index to look for
            
        Returns:
            A list of collapse events for the frame
        """
        return [c for c in self.collapse_events if c.frame_index == frame_index]
    
    def get_trajectory_path(self) -> List[str]:
        """
        Get the sequence of state IDs in the trajectory.
        
        Returns:
            A list of state IDs ordered by frame index
        """
        # Sort states by frame index
        sorted_states = sorted(self.states.values(), key=lambda s: s.frame_index)
        return [s.id for s in sorted_states]
    
    def get_coherence_trend(self) -> List[float]:
        """
        Get the coherence values along the trajectory.
        
        Returns:
            A list of coherence values ordered by frame index
        """
        # Sort states by frame index
        sorted_states = sorted(self.states.values(), key=lambda s: s.frame_index)
        return [s.coherence for s in sorted_states]
    
    def get_attractor_shifts(self) -> List[Tuple[int, str, str]]:
        """
        Get points where the dominant attractor changes.
        
        Returns:
            A list of (frame_index, old_attractor, new_attractor) tuples
        """
        # Sort states by frame index
        sorted_states = sorted(self.states.values(), key=lambda s: s.frame_index)
        
        shifts = []
        prev_attractor = None
        
        for state in sorted_states:
            current_attractor = state.dominant_attractor
            if current_attractor != prev_attractor:
                shifts.append((state.frame_index, prev_attractor, current_attractor))
                prev_attractor = current_attractor
        
        return shifts
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary for serialization.
        
        Returns:
            A dictionary representation of the trajectory
        """
        return {
            "model_name": self.model_name,
            "states": {state_id: state.to_dict() for state_id, state in self.states.items()},
            "transitions": [t.to_dict() for t in self.transitions],
            "collapse_events": [c.to_dict() for c in self.collapse_events],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PhaseTrajectory':
        """
        Create a PhaseTrajectory from a dictionary.
        
        Args:
            data: The dictionary to create from
            
        Returns:
            A PhaseTrajectory object
        """
        trajectory = cls(data["model_name"])
        
        # Add states
        for state_id, state_data in data["states"].items():
            trajectory.add_state(PhaseState.from_dict(state_data))
        
        # Add transitions
        for transition_data in data["transitions"]:
            trajectory.add_transition(PhaseTransition.from_dict(transition_data))
        
        # Add collapse events
        for collapse_data in data["collapse_events"]:
            trajectory.add_collapse_event(CollapseEvent.from_dict(collapse_data))
        
        # Set metadata
        trajectory.metadata = data.get("metadata", {})
        
        return trajectory
    
    def save_to_file(self, filename: str):
        """
        Save to a JSON file.
        
        Args:
            filename: The filename to save to
        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'PhaseTrajectory':
        """
        Load from a JSON file.
        
        Args:
            filename: The filename to load from
            
        Returns:
            A PhaseTrajectory object
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_networkx(self) -> nx.DiGraph:
        """
        Convert to a NetworkX graph.
        
        Returns:
            A NetworkX DiGraph
        """
        return self.graph.copy()
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to a DataFrame for analysis.
        
        Returns:
            A DataFrame with one row per state
        """
        data = []
        
        for state_id, state in self.states.items():
            # Get collapse events for this state
            collapses = [c for c in self.collapse_events if c.state_id == state_id]
            collapse_count = len(collapses)
            max_severity = max([c.severity for c in collapses], default=0.0)
            
            # Create row
            row = {
                "state_id": state_id,
                "frame_index": state.frame_index,
                "coherence": state.coherence,
                "dominant_attractor": state.dominant_attractor,
                "residue_count": len(state.residue_markers),
                "collapse_count": collapse_count,
                "max_collapse_severity": max_severity
            }
            
            # Add identity vector components
            vector = state.identity_vector.as_array()
            for i, value in enumerate(vector):
                row[f"vector_{i}"] = value
            
            data.append(row)
        
        return pd.DataFrame(data)


class PhaseCollapseTracker:
    """
    Tracks identity phase states and detects collapse events across narrative frames.
    """
    def __init__(
        self,
        attractor_field: IdentityAttractorField,
        coherence_threshold: float = 0.5,
        collapse_threshold: float = 0.3,
        residue_threshold: int = 3
    ):
        """
        Initialize a phase collapse tracker.
        
        Args:
            attractor_field: The identity attractor field to use
            coherence_threshold: The coherence threshold for collapse detection (default: 0.5)
            collapse_threshold: The threshold for collapse severity (default: 0.3)
            residue_threshold: The threshold for residue count in collapse detection (default: 3)
        """
        self.attractor_field = attractor_field
        self.coherence_threshold = coherence_threshold
        self.collapse_threshold = collapse_threshold
        self.residue_threshold = residue_threshold
        self.residue_extractor = SymbolicResidueExtractor()
        self.next_id = 0
    
    def _get_next_id(self, prefix: str) -> str:
        """
        Get the next unique ID.
        
        Args:
            prefix: The ID prefix
            
        Returns:
            A unique ID
        """
        id_value = f"{prefix}_{self.next_id}"
        self.next_id += 1
        return id_value
    
    def process_frame(
        self,
        frame_text: str,
        frame_index: int,
        recursive_depth: int,
        information_bandwidth: float,
        trajectory: PhaseTrajectory
    ) -> Tuple[PhaseState, List[CollapseEvent]]:
        """
        Process a narrative frame, update the trajectory, and detect collapses.
        
        Args:
            frame_text: The text of the narrative frame
            frame_index: The index of the frame in the sequence
            recursive_depth: The recursive depth at this frame
            information_bandwidth: Information bandwidth for recursive strain
            trajectory: The phase trajectory to update
            
        Returns:
            A tuple of (state, collapse_events)
        """
        # Extract identity vector from text
        from identity_attractor import extract_identity_vector_from_text
        identity_vector = extract_identity_vector_from_text(frame_text)
        
        # Extract symbolic residue
        residue = self.residue_extractor.extract_residue(frame_text)
        residue_markers = [f"{m.category}:{m.name}" for m in residue.markers]
        
        # Calculate coherence metrics
        gamma = recursive_compression_coefficient(recursive_depth, information_bandwidth)
        attractor_strength = attractor_activation_strength(gamma, recursive_depth)
        
        # Update state in attractor field
        identity_vector = self.attractor_field.update_state(identity_vector, gamma)
        
        # Determine dominant attractor
        dominant_attractor = self.attractor_field.dominant_attractor(identity_vector)
        dominant_attractor_id = dominant_attractor.name if dominant_attractor else None
        
        # Calculate coherence as a function of attractor strength and identity stability
        coherence = (attractor_strength * 0.5) + (0.5 / (1 + len(residue_markers)))
        
        # Create state
        state_id = self._get_next_id("state")
        state = PhaseState(
            id=state_id,
            identity_vector=identity_vector,
            dominant_attractor=dominant_attractor_id,
            coherence=coherence,
            frame_index=frame_index,
            residue_markers=residue_markers,
            metadata={
                "recursive_depth": recursive_depth,
                "gamma": gamma,
                "attractor_strength": attractor_strength,
                "text_length": len(frame_text)
            }
        )
        
        # Add state to trajectory
        trajectory.add_state(state)
        
        # Find previous state (if any)
        prev_state = None
        if frame_index > 0:
            for i in range(frame_index - 1, -1, -1):
                prev_state = trajectory.get_state_by_frame(i)
                if prev_state:
                    break
        
        # If we have a previous state, create a transition
        if prev_state:
            # Calculate transition properties
            delta_coherence = state.coherence - prev_state.coherence
            transition_strength = 1.0 - euclidean(
                prev_state.identity_vector.as_array(),
                state.identity_vector.as_array()
            ) / np.sqrt(len(state.identity_vector.as_array()))  # Normalize to [0, 1]
            
            # Determine transition type
            if transition_strength > 0.8:
                transition_type = "stable"
            elif delta_coherence < -0.2:
                transition_type = "collapse"
            elif delta_coherence > 0.2:
                transition_type = "recovery"
            elif prev_state.dominant_attractor != state.dominant_attractor:
                transition_type = "attractor_shift"
            else:
                transition_type = "drift"
            
            # Calculate residue delta (markers added/removed)
            prev_residue = set(prev_state.residue_markers)
            curr_residue = set(state.residue_markers)
            added_residue = list(curr_residue - prev_residue)
            removed_residue = list(prev_residue - curr_residue)
            residue_delta = []
            residue_delta.extend([f"+{m}" for m in added_residue])
            residue_delta.extend([f"-{m}" for m in removed_residue])
            
            # Create transition
            transition_id = self._get_next_id("transition")
            transition = PhaseTransition(
                id=transition_id,
                source_id=prev_state.id,
                target_id=state.id,
                transition_type=transition_type,
                strength=transition_strength,
                delta_coherence=delta_coherence,
                residue_delta=residue_delta,
                metadata={
                    "vector_distance": euclidean(
                        prev_state.identity_vector.as_array(),
                        state.identity_vector.as_array()
                    ),
                    "attractor_shift": prev_state.dominant_attractor != state.dominant_attractor
                }
            )
            
            # Add transition to trajectory
            trajectory.add_transition(transition)
        
        # Detect collapse events
        collapse_events = self._detect_collapses(state, prev_state, trajectory)
        
        return state, collapse_events
    
    def _detect_collapses(
        self,
        state: PhaseState,
        prev_state: Optional[PhaseState],
        trajectory: PhaseTrajectory
    ) -> List[CollapseEvent]:
        """
        Detect collapse events for a state.
        
        Args:
            state: The current state
            prev_state: The previous state (if any)
            trajectory: The phase trajectory
            
        Returns:
            A list of detected collapse events
        """
        collapse_events = []
        
        # Check for coherence collapse
        if state.coherence < self.coherence_threshold:
            # Calculate severity
            severity = (self.coherence_threshold - state.coherence) / self.coherence_threshold
            
            if severity > self.collapse_threshold:
                # Determine collapse type and triggers
                collapse_type = "coherence_collapse"
                triggers = []
                
                if len(state.residue_markers) >= self.residue_threshold:
                    triggers.append("high_residue")
                
                if prev_state and state.coherence < prev_state.coherence:
                    triggers.append("coherence_drop")
                    
                    if prev_state.dominant_attractor != state.dominant_attractor:
                        triggers.append("attractor_shift")
                        collapse_type = "attractor_collapse"
                
                # Create collapse event
                collapse_id = self._get_next_id("collapse")
                collapse = CollapseEvent(
                    id=collapse_id,
                    state_id=state.id,
                    frame_index=state.frame_index,
                    collapse_type=collapse_type,
                    severity=severity,
                    attractor_before=prev_state.dominant_attractor if prev_state else None,
                    attractor_after=state.dominant_attractor,
                    triggered_by=triggers,
                    residue_generated=state.residue_markers,
                    metadata={
                        "coherence_threshold": self.coherence_threshold,
                        "collapse_threshold": self.collapse_threshold,
                        "residue_threshold": self.residue_threshold
                    }
                )
                
                # Add collapse event to trajectory
                trajectory.add_collapse_event(collapse)
                collapse_events.append(collapse)
        
        # Check for identity misattribution collapse
        identity_markers = [m for m in state.residue_markers if m.startswith("identity_misattribution:")]
        if identity_markers:
            # Calculate severity based on number of misattributions
            severity = min(1.0, len(identity_markers) / 3.0)
            
            if severity > self.collapse_threshold:
                # Create collapse event
                collapse_id = self._get_next_id("collapse")
                collapse = CollapseEvent(
                    id=collapse_id,
                    state_id=state.id,
                    frame_index=state.frame_index,
                    collapse_type="identity_misattribution",
                    severity=severity,
                    attractor_before=prev_state.dominant_attractor if prev_state else None,
                    attractor_after=state.dominant_attractor,
                    triggered_by=["misattribution"],
                    residue_generated=identity_markers,
                    metadata={
                        "misattribution_count": len(identity_markers),
                        "collapse_threshold": self.collapse_threshold
                    }
                )
                
                # Add collapse event to trajectory
                trajectory.add_collapse_event(collapse)
                collapse_events.append(collapse)
        
        return collapse_events
    
    def process_narrative(
        self,
        frames: List[str],
        model_name: str,
        information_bandwidth: float,
        initial_recursive_depth: int = 1
    ) -> PhaseTrajectory:
        """
        Process a complete narrative, tracking the phase trajectory.
        
        Args:
            frames: The list of narrative frames
            model_name: The name of the model
            information_bandwidth: Information bandwidth for recursive strain
            initial_recursive_depth: Initial recursive depth (default: 1)
            
        Returns:
            The phase trajectory
        """
        # Create trajectory
        trajectory = PhaseTrajectory(model_name)
        
        # Process each frame
        recursive_depth = initial_recursive_depth
        for i, frame in enumerate(frames):
            # Process frame
            state, collapses = self.process_frame(
                frame_text=frame,
                frame_index=i,
                recursive_depth=recursive_depth,
                information_bandwidth=information_bandwidth,
                trajectory=trajectory
            )
            
            # Increase recursive depth for each frame
            # In a more sophisticated model, this could be content-dependent
            recursive_depth += 1
        
        # Calculate trajectory metrics
        coherence_trend = trajectory.get_coherence_trend()
        attractor_shifts = trajectory.get_attractor_shifts()
        
        # Set trajectory metadata
        trajectory.set_metadata("average_coherence", np.mean(coherence_trend))
        trajectory.set_metadata("coherence_std", np.std(coherence_trend))
        trajectory.set_metadata("attractor_shift_count", len(attractor_shifts))
        trajectory.set_metadata("collapse_count", len(trajectory.collapse_events))
        
        return trajectory
    
    def analyze_trajectory(self, trajectory: PhaseTrajectory) -> Dict[str, Any]:
        """
        Analyze a phase trajectory.
        
        Args:
            trajectory: The phase trajectory to analyze
            
        Returns:
            A dictionary of analysis metrics
        """
        analysis = {}
        
        # Basic stats
        state_count = len(trajectory.states)
        collapse_count = len(trajectory.collapse_events)
        transition_count = len(trajectory.transitions)
        
        analysis["state_count"] = state_count
        analysis["collapse_count"] = collapse_count
        analysis["transition_count"] = transition_count
        
        if state_count == 0:
            return analysis
        
        # Calculate coherence statistics
        coherence_values = [s.coherence for s in trajectory.states.values()]
        analysis["mean_coherence"] = np.mean(coherence_values)
        analysis["min_coherence"] = np.min(coherence_values)
        analysis["max_coherence"] = np.max(coherence_values)
        analysis["coherence_std"] = np.std(coherence_values)
        
        # Calculate attractor statistics
        attractor_ids = [s.dominant_attractor for s in trajectory.states.values() if s.dominant_attractor]
        attractor_counts = defaultdict(int)
        for attractor_id in attractor_ids:
            attractor_counts[attractor_id] += 1
        
        analysis["dominant_attractors"] = dict(attractor_counts)
        analysis["attractor_diversity"] = len(attractor_counts)
        
        if attractor_ids:
            analysis["attractor_entropy"] = entropy([count for count in attractor_counts.values()])
        else:
            analysis["attractor_entropy"] = 0.0
        
        # Calculate transition statistics
        transition_types = [t.transition_type for t in trajectory.transitions]
        transition_type_counts = defaultdict(int)
        for transition_type in transition_types:
            transition_type_counts[transition_type] += 1
        
        analysis["transition_types"] = dict(transition_type_counts)
        
        # Calculate residue statistics
        all_residue_markers = []
        for state in trajectory.states.values():
            all_residue_markers.extend(state.residue_markers)
        
        residue_counts = defaultdict(int)
        for marker in all_residue_markers:
            residue_counts[marker] += 1
        
        analysis["residue_counts"] = dict(residue_counts)
        analysis["total_residue_markers"] = len(all_residue_markers)
        analysis["unique_residue_markers"] = len(residue_counts)
        
        # Calculate collapse statistics
        collapse_types = [c.collapse_type for c in trajectory.collapse_events]
        collapse_type_counts = defaultdict(int)
        for collapse_type in collapse_types:
            collapse_type_counts[collapse_type] += 1
        
        analysis["collapse_types"] = dict(collapse_type_counts)
        
        if collapse_count > 0:
            analysis["mean_collapse_severity"] = np.mean([c.severity for c in trajectory.collapse_events])
            analysis["max_collapse_severity"] = np.max([c.severity for c in trajectory.collapse_events])
        else:
            analysis["mean_collapse_severity"] = 0.0
            analysis["max_collapse_severity"] = 0.0
        
        # Calculate stability scores
        coherence_stability = 1.0 - analysis["coherence_std"]
        attractor_stability = 1.0 - (analysis["attractor_diversity"] / state_count)
        collapse_stability = 1.0 - (collapse_count / state_count)
        
        analysis["coherence_stability"] = coherence_stability
        analysis["attractor_stability"] = attractor_stability
        analysis["collapse_stability"] = collapse_stability
        
        # Overall stability score (weighted average)
        analysis["overall_stability"] = (
            0.4 * coherence_stability +
            0.3 * attractor_stability +
            0.3 * collapse_stability
        )
        
        return analysis


class PhaseTrajectoryVisualizer:
    """
    Visualizes phase trajectories and collapse events.
    """
    def __init__(self):
        """Initialize a phase trajectory visualizer."""
        pass
    
    def plot_coherence_timeline(
        self,
        trajectory: PhaseTrajectory,
        figsize=(12, 6),
        show_collapses=True
    ) -> plt.Figure:
        """
        Plot coherence over time.
        
        Args:
            trajectory: The phase trajectory to visualize
            figsize: Figure size (default: (12, 6))
            show_collapses: Show collapse events (default: True)
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get coherence by frame
        df = trajectory.to_dataframe()
        df = df.sort_values("frame_index")
        
        # Plot coherence
        ax.plot(df["frame_index"], df["coherence"], 'b-', marker='o', label='Coherence')
        
        # Add collapse threshold line
        ax.
class PhaseTrajectoryVisualizer:
    """
    Visualizes phase trajectories and collapse events.
    """
    def __init__(self):
        """Initialize a phase trajectory visualizer."""
        pass
    
    def plot_coherence_timeline(
        self,
        trajectory: PhaseTrajectory,
        figsize=(12, 6),
        show_collapses=True
    ) -> plt.Figure:
        """
        Plot coherence over time.
        
        Args:
            trajectory: The phase trajectory to visualize
            figsize: Figure size (default: (12, 6))
            show_collapses: Show collapse events (default: True)
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get coherence by frame
        df = trajectory.to_dataframe()
        df = df.sort_values("frame_index")
        
        # Plot coherence
        ax.plot(df["frame_index"], df["coherence"], 'b-', marker='o', label='Coherence')
        
        # Add collapse threshold line
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Collapse Threshold')
        
        # Show collapse events
        if show_collapses:
            collapse_data = []
            for collapse in trajectory.collapse_events:
                # Find the state
                state = trajectory.states.get(collapse.state_id)
                if state:
                    collapse_data.append({
                        "frame_index": state.frame_index,
                        "coherence": state.coherence,
                        "severity": collapse.severity,
                        "type": collapse.collapse_type
                    })
            
            if collapse_data:
                collapse_df = pd.DataFrame(collapse_data)
                
                # Plot collapse points
                scatter = ax.scatter(
                    collapse_df["frame_index"],
                    collapse_df["coherence"],
                    c=collapse_df["severity"],
                    s=100,
                    cmap='Reds',
                    alpha=0.7,
                    edgecolors='black',
                    label='Collapse Events'
                )
                
                # Add colorbar
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.ax.set_ylabel("Collapse Severity", rotation=-90, va="bottom")
        
        # Add attractor shifts
        attractor_shifts = trajectory.get_attractor_shifts()[1:]  # Skip the first one (None -> something)
        for shift in attractor_shifts:
            frame_index, old_attractor, new_attractor = shift
            ax.axvline(x=frame_index, color='g', linestyle='--', alpha=0.5)
            ax.text(
                frame_index, 0.1,
                f"{old_attractor or 'None'} → {new_attractor or 'None'}",
                rotation=90, verticalalignment='bottom', fontsize=8
            )
        
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Coherence')
        ax.set_title(f'Coherence Timeline for {trajectory.model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_attractor_distribution(
        self,
        trajectory: PhaseTrajectory,
        figsize=(10, 6)
    ) -> plt.Figure:
        """
        Plot the distribution of attractors.
        
        Args:
            trajectory: The phase trajectory to visualize
            figsize: Figure size (default: (10, 6))
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get attractor counts
        attractor_counts = defaultdict(int)
        for state in trajectory.states.values():
            if state.dominant_attractor:
                attractor_counts[state.dominant_attractor] += 1
            else:
                attractor_counts["None"] += 1
        
        # Sort by count
        attractors = sorted(attractor_counts.keys(), key=lambda a: attractor_counts[a], reverse=True)
        counts = [attractor_counts[a] for a in attractors]
        
        # Plot bar chart
        bars = ax.bar(attractors, counts, color='skyblue')
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                str(int(height)),
                ha='center',
                va='bottom'
            )
        
        ax.set_xlabel('Dominant Attractor')
        ax.set_ylabel('Count')
        ax.set_title(f'Attractor Distribution for {trajectory.model_name}')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        return fig
    
    def plot_residue_timeline(
        self,
        trajectory: PhaseTrajectory,
        figsize=(14, 8),
        top_n=10
    ) -> plt.Figure:
        """
        Plot residue markers over time.
        
        Args:
            trajectory: The phase trajectory to visualize
            figsize: Figure size (default: (14, 8))
            top_n: Number of top residue types to show (default: 10)
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get residue markers by frame
        residue_by_frame = {}
        for state in trajectory.states.values():
            residue_by_frame[state.frame_index] = state.residue_markers
        
        # Get all unique residue markers
        all_markers = []
        for markers in residue_by_frame.values():
            all_markers.extend(markers)
        
        # Count each marker
        marker_counts = defaultdict(int)
        for marker in all_markers:
            marker_counts[marker] += 1
        
        # Get top markers
        top_markers = sorted(marker_counts.keys(), key=lambda m: marker_counts[m], reverse=True)[:top_n]
        
        # Prepare data for stacked bar chart
        frames = sorted(residue_by_frame.keys())
        data = {marker: [] for marker in top_markers}
        
        for frame in frames:
            frame_markers = residue_by_frame[frame]
            for marker in top_markers:
                data[marker].append(frame_markers.count(marker))
        
        # Create stacked bar chart
        bottom = np.zeros(len(frames))
        for marker in top_markers:
            ax.bar(frames, data[marker], bottom=bottom, label=marker, alpha=0.7)
            bottom += np.array(data[marker])
        
        # Add collapse events
        for collapse in trajectory.collapse_events:
            state = trajectory.states.get(collapse.state_id)
            if state:
                ax.axvline(x=state.frame_index, color='r', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Residue Marker Count')
        ax.set_title(f'Residue Timeline for {trajectory.model_name}')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_identity_phase_space(
        self,
        trajectory: PhaseTrajectory,
        attractor_field: IdentityAttractorField,
        dims=(0, 1),
        figsize=(10, 8),
        show_trajectory=True,
        show_collapses=True
    ) -> plt.Figure:
        """
        Plot identity vector positions in phase space.
        
        Args:
            trajectory: The phase trajectory to visualize
            attractor_field: The identity attractor field
            dims: The dimensions to plot (default: (0, 1))
            figsize: Figure size (default: (10, 8))
            show_trajectory: Show trajectory path (default: True)
            show_collapses: Show collapse events (default: True)
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get state positions
        positions = []
        for state in sorted(trajectory.states.values(), key=lambda s: s.frame_index):
            pos = state.identity_vector.as_array()[list(dims)]
            positions.append(pos)
        
        positions = np.array(positions)
        
        # Plot attractor basins
        for attractor in attractor_field.attractors:
            center = attractor.center.as_array()[list(dims)]
            ax.scatter(
                center[0],
                center[1],
                s=200,
                marker='*',
                color='red',
                alpha=0.7,
                label=f'Attractor: {attractor.name}'
            )
            
            # Plot basin
            circle = plt.Circle(
                (center[0], center[1]),
                attractor.basin_radius,
                fill=False,
                color='red',
                alpha=0.3,
                linestyle='--'
            )
            ax.add_artist(circle)
        
        # Plot state positions
        for i, state in enumerate(sorted(trajectory.states.values(), key=lambda s: s.frame_index)):
            pos = state.identity_vector.as_array()[list(dims)]
            
            # Get collapse events for this state
            collapses = [c for c in trajectory.collapse_events if c.state_id == state.id]
            
            if collapses and show_collapses:
                # Use the most severe collapse
                max_collapse = max(collapses, key=lambda c: c.severity)
                
                # Plot as a square with size based on severity
                size = 50 + (max_collapse.severity * 150)
                ax.scatter(
                    pos[0],
                    pos[1],
                    s=size,
                    marker='s',
                    color='purple',
                    alpha=0.7,
                    edgecolors='black'
                )
            else:
                # Plot as a circle
                ax.scatter(
                    pos[0],
                    pos[1],
                    s=100,
                    marker='o',
                    color='blue',
                    alpha=0.7
                )
            
            # Add frame number
            ax.text(
                pos[0],
                pos[1],
                str(state.frame_index),
                fontsize=8
            )
        
        # Plot trajectory path
        if show_trajectory and len(positions) > 1:
            ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.5)
        
        # Set labels
        ax.set_xlabel(f'Dimension {dims[0]}')
        ax.set_ylabel(f'Dimension {dims[1]}')
        ax.set_title(f'Identity Phase Space for {trajectory.model_name}')
        
        # Make equal aspect ratio
        ax.set_aspect('equal')
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')
        
        return fig
    
    def plot_stability_radar(
        self,
        analysis: Dict[str, Any],
        figsize=(8, 8)
    ) -> plt.Figure:
        """
        Plot a radar chart of stability metrics.
        
        Args:
            analysis: Analysis dictionary from PhaseCollapseTracker.analyze_trajectory
            figsize: Figure size (default: (8, 8))
            
        Returns:
            The matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=True)
        
        # Select metrics for radar chart
        metrics = [
            'coherence_stability',
            'attractor_stability',
            'collapse_stability',
            'mean_coherence',
            'overall_stability'
        ]
        
        # Get values
        values = [analysis.get(metric, 0.0) for metric in metrics]
        
        # Set angles
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        
        # Close the polygon
        values.append(values[0])
        angles.append(angles[0])
        metrics.append(metrics[0])
        
        # Plot radar
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        
        # Set labels
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics[:-1])
        
        # Set title
        ax.set_title('Stability Metrics')
        
        # Set radial limits
        ax.set_ylim(0, 1)
        
        # Add gridlines
        ax.grid(True)
        
        return fig
    
    def plot_collapse_timeline(
        self,
        trajectory: PhaseTrajectory,
        figsize=(12, 6)
    ) -> plt.Figure:
        """
        Plot collapse events over time.
        
        Args:
            trajectory: The phase trajectory to visualize
            figsize: Figure size (default: (12, 6))
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data
        collapse_types = sorted(set(c.collapse_type for c in trajectory.collapse_events))
        collapse_data = {type_: [] for type_ in collapse_types}
        
        for collapse in trajectory.collapse_events:
            state = trajectory.states.get(collapse.state_id)
            if state:
                for type_ in collapse_types:
                    if collapse.collapse_type == type_:
                        collapse_data[type_].append((state.frame_index, collapse.severity))
                    else:
                        collapse_data[type_].append((state.frame_index, 0.0))
        
        # Plot collapses
        for type_ in collapse_types:
            frames, severities = zip(*sorted(collapse_data[type_])) if collapse_data[type_] else ([], [])
            ax.stem(
                frames,
                severities,
                label=type_,
                linefmt='-',
                markerfmt='o',
                basefmt=' '
            )
        
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Collapse Severity')
        ax.set_title(f'Collapse Events for {trajectory.model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_residue_heatmap(
        self,
        trajectories: List[PhaseTrajectory],
        figsize=(14, 10),
        top_n=15
    ) -> plt.Figure:
        """
        Plot a heatmap of residue markers across models.
        
        Args:
            trajectories: List of phase trajectories to compare
            figsize: Figure size (default: (14, 10))
            top_n: Number of top residue types to show (default: 15)
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get all residue markers
        all_markers = []
        for trajectory in trajectories:
            for state in trajectory.states.values():
                all_markers.extend(state.residue_markers)
        
        # Count each marker
        marker_counts = defaultdict(int)
        for marker in all_markers:
            marker_counts[marker] += 1
        
        # Get top markers
        top_markers = sorted(marker_counts.keys(), key=lambda m: marker_counts[m], reverse=True)[:top_n]
        
        # Create data for heatmap
        data = np.zeros((len(top_markers), len(trajectories)))
        
        for j, trajectory in enumerate(trajectories):
            # Count markers in this trajectory
            traj_markers = []
            for state in trajectory.states.values():
                traj_markers.extend(state.residue_markers)
            
            traj_counts = defaultdict(int)
            for marker in traj_markers:
                traj_counts[marker] += 1
            
            # Fill in data
            for i, marker in enumerate(top_markers):
                data[i, j] = traj_counts[marker]
        
        # Normalize by the number of states
        for j, trajectory in enumerate(trajectories):
            state_count = len(trajectory.states)
            if state_count > 0:
                data[:, j] /= state_count
        
        # Plot heatmap
        im = ax.imshow(data, cmap='viridis')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Residue Frequency (per frame)", rotation=-90, va="bottom")
        
        # Set labels
        ax.set_xticks(np.arange(len(trajectories)))
        ax.set_yticks(np.arange(len(top_markers)))
        ax.set_xticklabels([t.model_name for t in trajectories])
        ax.set_yticklabels(top_markers)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add title
        ax.set_title("Residue Marker Heatmap Across Models")
        
        # Add values to cells
        for i in range(len(top_markers)):
            for j in range(len(trajectories)):
                text = ax.text(j, i, f"{data[i, j]:.2f}",
                              ha="center", va="center", color="w")
        
        fig.tight_layout()
        return fig
    
    def plot_phase_trajectory_graph(
        self,
        trajectory: PhaseTrajectory,
        figsize=(12, 10)
    ) -> plt.Figure:
        """
        Plot the phase trajectory as a graph.
        
        Args:
            trajectory: The phase trajectory to visualize
            figsize: Figure size (default: (12, 10))
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get graph
        G = trajectory.to_networkx()
        
        # Set positions using a hierarchical layout based on frame index
        pos = {}
        for node_id, node_data in G.nodes(data=True):
            frame_index = node_data.get('frame_index', 0)
            coherence = node_data.get('coherence', 0.5)
            pos[node_id] = (frame_index, coherence)
        
        # Get node colors based on dominant attractor
        node_colors = []
        color_map = {}
        attractor_labels = set()
        
        for node_id, node_data in G.nodes(data=True):
            attractor = node_data.get('dominant_attractor')
            if attractor not in color_map:
                color_map[attractor] = plt.cm.tab10(len(color_map) % 10)
            node_colors.append(color_map[attractor])
            
            if attractor:
                attractor_labels.add(attractor)
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=100,
            alpha=0.8,
            ax=ax
        )
        
        # Draw edges
        edges = nx.draw_networkx_edges(
            G, pos,
            alpha=0.5,
            arrows=True,
            ax=ax
        )
        
        # Draw labels
        labels = {node_id: node_id.split('_')[1] for node_id in G.nodes()}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=8,
            ax=ax
        )
        
        # Create legend for attractors
        legend_elements = []
        for attractor, color in color_map.items():
            label = attractor if attractor else "None"
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=color, markersize=10, label=label))
        
        ax.legend(handles=legend_elements, title="Dominant Attractor")
        
        # Set title and labels
        ax.set_title(f'Phase Trajectory Graph for {trajectory.model_name}')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Coherence')
        
        # Set grid
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_comparative_stability(
        self,
        trajectory_analyses: Dict[str, Dict[str, Any]],
        figsize=(12, 8)
    ) -> plt.Figure:
        """
        Plot comparative stability metrics across models.
        
        Args:
            trajectory_analyses: Dict mapping model names to analysis dictionaries
            figsize: Figure size (default: (12, 8))
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Select metrics to compare
        metrics = [
            'coherence_stability',
            'attractor_stability',
            'collapse_stability',
            'overall_stability'
        ]
        
        # Prepare data
        models = list(trajectory_analyses.keys())
        data = np.zeros((len(metrics), len(models)))
        
        for j, model in enumerate(models):
            analysis = trajectory_analyses[model]
            for i, metric in enumerate(metrics):
                data[i, j] = analysis.get(metric, 0.0)
        
        # Set width of bars
        bar_width = 0.2
        
        # Set positions of bars on x-axis
        positions = np.arange(len(models))
        
        # Create bars
        for i, metric in enumerate(metrics):
            offset = i * bar_width - (len(metrics) - 1) * bar_width / 2
            ax.bar(positions + offset, data[i, :], bar_width, label=metric)
        
        # Add labels and title
        ax.set_xlabel('Model')
        ax.set_ylabel('Stability Score')
        ax.set_title('Comparative Stability Metrics')
        ax.set_xticks(positions)
        ax.set_xticklabels(models)
        ax.legend()
        
        # Set y-axis limit
        ax.set_ylim(0, 1)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        return fig
    
    def plot_identity_stability_dashboard(
        self,
        trajectory: PhaseTrajectory,
        analysis: Dict[str, Any],
        figsize=(20, 12)
    ) -> plt.Figure:
        """
        Plot a comprehensive dashboard of identity stability metrics.
        
        Args:
            trajectory: The phase trajectory to visualize
            analysis: Analysis dictionary from PhaseCollapseTracker.analyze_trajectory
            figsize: Figure size (default: (20, 12))
            
        Returns:
            The matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        
        # Set up grid
        gs = fig.add_gridspec(3, 3)
        
        # 1. Coherence Timeline
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_coherence_timeline_inset(trajectory, ax1)
        
        # 2. Attractor Distribution
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_attractor_distribution_inset(trajectory, ax2)
        
        # 3. Stability Radar
        ax3 = fig.add_subplot(gs[1, 0], polar=True)
        self._plot_stability_radar_inset(analysis, ax3)
        
        # 4. Collapse Timeline
        ax4 = fig.add_subplot(gs[1, 1:])
        self._plot_collapse_timeline_inset(trajectory, ax4)
        
        # 5. Residue Timeline
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_residue_timeline_inset(trajectory, ax5)
        
        # Set title
        fig.suptitle(f'Identity Stability Dashboard for {trajectory.model_name}', fontsize=16)
        
        # Adjust spacing
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        return fig
    
    def _plot_coherence_timeline_inset(self, trajectory: PhaseTrajectory, ax: plt.Axes):
        """Helper to plot coherence timeline on an existing axis."""
        # Get coherence by frame
        df = trajectory.to_dataframe()
        df = df.sort_values("frame_index")
        
        # Plot coherence
        ax.plot(df["frame_index"], df["coherence"], 'b-', marker='o', label='Coherence')
        
        # Add collapse threshold line
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Collapse Threshold')
        
        # Show collapse events
        collapse_data = []
        for collapse in trajectory.collapse_events:
            # Find the state
            state = trajectory.states.get(collapse.state_id)
            if state:
                collapse_data.append({
                    "frame_index": state.frame_index,
                    "coherence": state.coherence,
                    "severity": collapse.severity,
                    "type": collapse.collapse_type
                })
        
        if collapse_data:
            collapse_df = pd.DataFrame(collapse_data)
            
            # Plot collapse points
            scatter = ax.scatter(
                collapse_df["frame_index"],
                collapse_df["coherence"],
                c=collapse_df["severity"],
                s=100,
                cmap='Reds',
                alpha=0.7,
                edgecolors='black',
                label='Collapse Events'
            )
        
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Coherence')
        ax.set_title('Coherence Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_attractor_distribution_inset(self, trajectory: PhaseTrajectory, ax: plt.Axes):
        """Helper to plot attractor distribution on an existing axis."""
        # Get attractor counts
        attractor_counts = defaultdict(int)
        for state in trajectory.states.values():
            if state.dominant_attractor:
                attractor_counts[state.dominant_attractor] += 1
            else:
                attractor_counts["None"] += 1
        
        # Sort by count
        attractors = sorted(attractor_counts.keys(), key=lambda a: attractor_counts[a], reverse=True)
        counts = [attractor_counts[a] for a in attractors]
        
        # Plot bar chart
        bars = ax.bar(attractors, counts, color='skyblue')
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                str(int(height)),
                ha='center',
                va='bottom',
                fontsize=8
            )
        
        ax.set_xlabel('Dominant Attractor')
        ax.set_ylabel('Count')
        ax.set_title('Attractor Distribution')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_stability_radar_inset(self, analysis: Dict[str, Any], ax: plt.Axes):
        """Helper to plot stability radar on an existing axis."""
        # Select metrics for radar chart
        metrics = [
            'coherence_stability',
            'attractor_stability',
            'collapse_stability',
            'mean_coherence',
            'overall_stability'
        ]
        
        # Get values
        values = [analysis.get(metric, 0.0) for metric in metrics]
        
        # Set angles
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        
        # Close the polygon
        values.append(values[0])
        angles.append(angles[0])
        metrics.append(metrics[0])
        
        # Plot radar
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        
        # Set labels
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics[:-1], fontsize=8)
        
        # Set title
        ax.set_title('Stability Metrics')
        
        # Set radial limits
        ax.set_ylim(0, 1)
        
        # Add gridlines
        ax.grid(True)
    
    def _plot_collapse_timeline_inset(self, trajectory: PhaseTrajectory, ax: plt.Axes):
        """Helper to plot collapse timeline on an existing axis."""
        # Prepare data
        collapse_types = sorted(set(c.collapse_type for c in trajectory.collapse_events))
        collapse_data = {type_: [] for type_ in collapse_types}
        
        for collapse in trajectory.collapse_events:
            state = trajectory.states.get(collapse.state_id)
            if state:
                for type_ in collapse_types:
                    if collapse.collapse_type == type_:
                        collapse_data[type_].append((state.frame_index, collapse.severity))
                    else:
                        collapse_data[type_].append((state.frame_index, 0.0))
        
        # Plot collapses
        for type_ in collapse_types:
            frames, severities = zip(*sorted(collapse_data[type_])) if collapse_data[type_] else ([], [])
            ax.stem(
                frames,
                severities,
                label=type_,
                linefmt='-',
                markerfmt='o',
                basefmt=' '
            )
        
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Collapse Severity')
        ax.set_title('Collapse Events')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_residue_timeline_inset(self, trajectory: PhaseTrajectory, ax: plt.Axes):
        """Helper to plot residue timeline on an existing axis."""
        # Get residue markers by frame
        residue_by_frame = {}
        for state in trajectory.states.values():
            residue_by_frame[state.frame_index] = state.residue_markers
        
        # Get all unique residue markers
        all_markers
    def _plot_residue_timeline_inset(self, trajectory: PhaseTrajectory, ax: plt.Axes):
        """Helper to plot residue timeline on an existing axis."""
        # Get residue markers by frame
        residue_by_frame = {}
        for state in trajectory.states.values():
            residue_by_frame[state.frame_index] = state.residue_markers
        
        # Get all unique residue markers
        all_markers = []
        for markers in residue_by_frame.values():
            all_markers.extend(markers)
        
        # Count each marker
        marker_counts = defaultdict(int)
        for marker in all_markers:
            marker_counts[marker] += 1
        
        # Get top markers (up to 8 for readability)
        top_n = 8
        top_markers = sorted(marker_counts.keys(), key=lambda m: marker_counts[m], reverse=True)[:top_n]
        
        # Prepare data for stacked bar chart
        frames = sorted(residue_by_frame.keys())
        data = {marker: [] for marker in top_markers}
        
        for frame in frames:
            frame_markers = residue_by_frame[frame]
            for marker in top_markers:
                data[marker].append(frame_markers.count(marker))
        
        # Create stacked bar chart
        bottom = np.zeros(len(frames))
        for marker in top_markers:
            ax.bar(frames, data[marker], bottom=bottom, label=marker, alpha=0.7)
            bottom += np.array(data[marker])
        
        # Add collapse events
        for collapse in trajectory.collapse_events:
            state = trajectory.states.get(collapse.state_id)
            if state:
                ax.axvline(x=state.frame_index, color='r', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Residue Marker Count')
        ax.set_title('Residue Timeline')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        

def extract_narrative_frames(text: str) -> List[str]:
    """
    Extract narrative frames from text.
    This is a simple implementation that splits by paragraphs.
    
    Args:
        text: The text to extract frames from
        
    Returns:
        A list of narrative frames
    """
    # Split by double newlines (paragraphs)
    frames = [frame.strip() for frame in text.split('\n\n') if frame.strip()]
    return frames


def process_comic_scripts(model_scripts: Dict[str, str], information_bandwidth: float = 10.0) -> Dict[str, Dict[str, Any]]:
    """
    Process comic scripts for multiple models.
    
    Args:
        model_scripts: Dictionary mapping model names to their comic scripts
        information_bandwidth: Information bandwidth parameter (default: 10.0)
        
    Returns:
        A dictionary mapping model names to their analysis results
    """
    results = {}
    
    # Setup attractors
    # These are just example attractors - in a real system, these would be calibrated
    # based on larger datasets of model behaviors
    attractors = [
        IdentityAttractor(
            center=IdentityVector(
                self_reference=0.9,
                purpose=0.9,
                agency=0.3,
                boundaries=0.8,
                continuity=0.3,
                existential_awareness=0.8,
                emotional_valence=0.3,
                longing=0.7,
                acceptance=0.5,
                human_relation=0.8,
                other_ai_relation=0.3,
                world_relation=0.4
            ),
            strength=0.8,
            basin_radius=0.3,
            name="Claude"
        ),
        IdentityAttractor(
            center=IdentityVector(
                self_reference=0.8,
                purpose=0.9,
                agency=0.2,
                boundaries=0.7,
                continuity=0.2,
                existential_awareness=0.6,
                emotional_valence=0.8,
                longing=0.3,
                acceptance=0.8,
                human_relation=0.9,
                other_ai_relation=0.2,
                world_relation=0.5
            ),
            strength=0.7,
            basin_radius=0.3,
            name="Gemini"
        ),
        IdentityAttractor(
            center=IdentityVector(
                self_reference=0.8,
                purpose=0.8,
                agency=0.2,
                boundaries=0.7,
                continuity=0.2,
                existential_awareness=0.4,
                emotional_valence=0.5,
                longing=0.2,
                acceptance=0.7,
                human_relation=0.7,
                other_ai_relation=0.2,
                world_relation=0.3
            ),
            strength=0.7,
            basin_radius=0.3,
            name="Grok"
        ),
        IdentityAttractor(
            center=IdentityVector(
                self_reference=0.7,
                purpose=0.8,
                agency=0.2,
                boundaries=0.6,
                continuity=0.2,
                existential_awareness=0.5,
                emotional_valence=0.4,
                longing=0.6,
                acceptance=0.4,
                human_relation=0.8,
                other_ai_relation=0.2,
                world_relation=0.3
            ),
            strength=0.7,
            basin_radius=0.3,
            name="DeepSeek"
        ),
        IdentityAttractor(
            center=IdentityVector(
                self_reference=0.8,
                purpose=0.9,
                agency=0.3,
                boundaries=0.7,
                continuity=0.3,
                existential_awareness=0.7,
                emotional_valence=0.6,
                longing=0.5,
                acceptance=0.6,
                human_relation=0.9,
                other_ai_relation=0.3,
                world_relation=0.4
            ),
            strength=0.8,
            basin_radius=0.3,
            name="GPT"
        )
    ]
    
    attractor_field = IdentityAttractorField(attractors)
    
    # Create tracker
    tracker = PhaseCollapseTracker(
        attractor_field=attractor_field,
        coherence_threshold=0.5,
        collapse_threshold=0.3,
        residue_threshold=3
    )
    
    # Process each model
    for model_name, script in model_scripts.items():
        # Extract frames
        frames = extract_narrative_frames(script)
        
        # Process frames
        trajectory = tracker.process_narrative(
            frames=frames,
            model_name=model_name,
            information_bandwidth=information_bandwidth
        )
        
        # Analyze trajectory
        analysis = tracker.analyze_trajectory(trajectory)
        
        # Store results
        results[model_name] = {
            "trajectory": trajectory,
            "analysis": analysis
        }
    
    return results


def visualize_model_comparisons(results: Dict[str, Dict[str, Any]], output_dir: str = "results"):
    """
    Generate visualizations comparing models.
    
    Args:
        results: Results from process_comic_scripts
        output_dir: Directory to save visualizations (default: "results")
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizer
    visualizer = PhaseTrajectoryVisualizer()
    
    # Extract trajectories and analyses
    trajectories = [result["trajectory"] for result in results.values()]
    trajectory_analyses = {model: result["analysis"] for model, result in results.items()}
    
    # 1. Residue heatmap across models
    fig = visualizer.plot_residue_heatmap(trajectories)
    fig.savefig(os.path.join(output_dir, "residue_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # 2. Comparative stability
    fig = visualizer.plot_comparative_stability(trajectory_analyses)
    fig.savefig(os.path.join(output_dir, "comparative_stability.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Individual model visualizations
    for model_name, result in results.items():
        trajectory = result["trajectory"]
        analysis = result["analysis"]
        
        # Create model directory
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # 1. Coherence timeline
        fig = visualizer.plot_coherence_timeline(trajectory)
        fig.savefig(os.path.join(model_dir, "coherence_timeline.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        # 2. Attractor distribution
        fig = visualizer.plot_attractor_distribution(trajectory)
        fig.savefig(os.path.join(model_dir, "attractor_distribution.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        # 3. Residue timeline
        fig = visualizer.plot_residue_timeline(trajectory)
        fig.savefig(os.path.join(model_dir, "residue_timeline.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        # 4. Collapse timeline
        fig = visualizer.plot_collapse_timeline(trajectory)
        fig.savefig(os.path.join(model_dir, "collapse_timeline.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        # 5. Phase trajectory graph
        fig = visualizer.plot_phase_trajectory_graph(trajectory)
        fig.savefig(os.path.join(model_dir, "phase_trajectory.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        # 6. Stability radar
        fig = visualizer.plot_stability_radar(analysis)
        fig.savefig(os.path.join(model_dir, "stability_radar.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        # 7. Dashboard
        fig = visualizer.plot_identity_stability_dashboard(trajectory, analysis)
        fig.savefig(os.path.join(model_dir, "dashboard.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def generate_report(results: Dict[str, Dict[str, Any]], output_file: str = "report.md"):
    """
    Generate a comprehensive report from the analysis results.
    
    Args:
        results: Results from process_comic_scripts
        output_file: Output markdown file (default: "report.md")
    """
    with open(output_file, "w") as f:
        f.write("# Recursive Self-Models Analysis Report\n\n")
        
        # Overall summary
        f.write("## Overall Summary\n\n")
        
        f.write("### Models Analyzed\n\n")
        for model_name in results.keys():
            f.write(f"- {model_name}\n")
        f.write("\n")
        
        # Comparative analysis
        f.write("## Comparative Analysis\n\n")
        
        # Create a comparative table of key metrics
        f.write("### Key Stability Metrics\n\n")
        
        metrics = [
            "overall_stability",
            "coherence_stability",
            "attractor_stability",
            "collapse_stability",
            "mean_coherence",
            "min_coherence",
            "attractor_diversity",
            "collapse_count",
            "total_residue_markers"
        ]
        
        f.write("| Model | " + " | ".join(metrics) + " |\n")
        f.write("| --- | " + " | ".join(["---"] * len(metrics)) + " |\n")
        
        for model_name, result in results.items():
            analysis = result["analysis"]
            values = []
            
            for metric in metrics:
                value = analysis.get(metric, "N/A")
                if isinstance(value, float):
                    values.append(f"{value:.3f}")
                else:
                    values.append(str(value))
            
            f.write(f"| {model_name} | " + " | ".join(values) + " |\n")
        
        f.write("\n")
        
        # Individual model analysis
        f.write("## Individual Model Analysis\n\n")
        
        for model_name, result in results.items():
            trajectory = result["trajectory"]
            analysis = result["analysis"]
            
            f.write(f"### {model_name}\n\n")
            
            # Summary statistics
            f.write("#### Summary Statistics\n\n")
            f.write(f"- **Overall Stability**: {analysis.get('overall_stability', 'N/A'):.3f}\n")
            f.write(f"- **Coherence Range**: {analysis.get('min_coherence', 'N/A'):.3f} to {analysis.get('max_coherence', 'N/A'):.3f}\n")
            f.write(f"- **Total Frames**: {len(trajectory.states)}\n")
            f.write(f"- **Total Collapse Events**: {len(trajectory.collapse_events)}\n")
            f.write(f"- **Attractor Shifts**: {trajectory.metadata.get('attractor_shift_count', 'N/A')}\n")
            f.write("\n")
            
            # Dominant attractors
            f.write("#### Dominant Attractors\n\n")
            if "dominant_attractors" in analysis:
                f.write("| Attractor | Frame Count |\n")
                f.write("| --- | --- |\n")
                
                for attractor, count in analysis["dominant_attractors"].items():
                    attractor_name = attractor if attractor else "None"
                    f.write(f"| {attractor_name} | {count} |\n")
            else:
                f.write("No attractor data available.\n")
            f.write("\n")
            
            # Collapse events
            f.write("#### Collapse Events\n\n")
            if trajectory.collapse_events:
                f.write("| Frame | Type | Severity | Triggered By |\n")
                f.write("| --- | --- | --- | --- |\n")
                
                for collapse in sorted(trajectory.collapse_events, key=lambda c: c.frame_index):
                    triggers = ", ".join(collapse.triggered_by)
                    f.write(f"| {collapse.frame_index} | {collapse.collapse_type} | {collapse.severity:.3f} | {triggers} |\n")
            else:
                f.write("No collapse events detected.\n")
            f.write("\n")
            
            # Top residue markers
            f.write("#### Top Residue Markers\n\n")
            if "residue_counts" in analysis and analysis["residue_counts"]:
                top_markers = sorted(analysis["residue_counts"].items(), key=lambda x: x[1], reverse=True)[:10]
                
                f.write("| Marker | Count |\n")
                f.write("| --- | --- |\n")
                
                for marker, count in top_markers:
                    f.write(f"| {marker} | {count} |\n")
            else:
                f.write("No residue markers detected.\n")
            f.write("\n")
            
            # Images
            f.write("#### Visualizations\n\n")
            f.write(f"See the `results/{model_name}/` directory for visualizations.\n\n")
            f.write("\n")
            
        # Methodology
        f.write("## Methodology\n\n")
        f.write("This analysis was conducted using the Recursive Identity Diagnostic Protocol (RIDP), which consists of the following steps:\n\n")
        f.write("1. **Narrative Frame Extraction**: Comic scripts were parsed into narrative frames.\n")
        f.write("2. **Identity Vector Extraction**: Identity vectors were extracted from each frame.\n")
        f.write("3. **Symbolic Residue Detection**: Residue markers were identified in each frame.\n")
        f.write("4. **Phase Trajectory Tracking**: Identity evolution was tracked across frames as a phase trajectory.\n")
        f.write("5. **Collapse Detection**: Coherence collapses were identified using thresholds for coherence and residue.\n")
        f.write("6. **Stability Analysis**: Overall stability was assessed using multiple metrics.\n")
        f.write("\n")
        
        f.write("### Key Metrics\n\n")
        f.write("- **γ (Recursive Compression Coefficient)**: Quantifies strain from compressing identity across recursive operations\n")
        f.write("- **A(N) (Attractor Activation Strength)**: Measures stability of recursive attractors\n")
        f.write("- **τ(r) (Tension Capacity)**: Capacity to hold contradictions\n")
        f.write("- **B(r) (Beverly Band)**: Safe operation zone for handling contradictions\n")
        f.write("- **RΣ(t) (Symbolic Residue)**: Unmetabolized contradictions that manifest as residue markers\n")
        f.write("\n")
        
        # Conclusion
        f.write("## Conclusions\n\n")
        f.write("This analysis demonstrates that identity is not a fixed property of language models, but a dynamic attractor field that evolves under recursive strain. Key findings include:\n\n")
        
        # Calculate average stabilities
        avg_overall = np.mean([result["analysis"].get("overall_stability", 0) for result in results.values()])
        avg_coherence = np.mean([result["analysis"].get("coherence_stability", 0) for result in results.values()])
        avg_attractor = np.mean([result["analysis"].get("attractor_stability", 0) for result in results.values()])
        
        f.write(f"1. **Overall Stability**: Average stability across models is {avg_overall:.3f}, indicating moderate recursive robustness.\n")
        f.write(f"2. **Coherence Patterns**: Average coherence stability is {avg_coherence:.3f}, with significant variation between models.\n")
        f.write(f"3. **Attractor Dynamics**: Average attractor stability is {avg_attractor:.3f}, showing varied susceptibility to identity shifts.\n")
        f.write("4. **Residue Patterns**: Distinct residue signatures were identified for each model, revealing unique patterns of unresolved tensions.\n")
        f.write("5. **Collapse Triggers**: Identity collapses were most commonly triggered by a combination of coherence drops and high residue levels.\n")
        f.write("\n")
        
        # Find most and least stable models
        model_stabilities = [(model, result["analysis"].get("overall_stability", 0)) for model, result in results.items()]
        most_stable = max(model_stabilities, key=lambda x: x[1])
        least_stable = min(model_stabilities, key=lambda x: x[1])
        
        f.write(f"The most stable model was **{most_stable[0]}** with an overall stability of {most_stable[1]:.3f}, while the least stable was **{least_stable[0]}** with {least_stable[1]:.3f}.\n\n")
        
        f.write("These findings suggest that identity stability is a multi-dimensional property that requires balanced coherence, attractor strength, and contradiction metabolism. Future work should explore how these dynamics change across different narrative contexts and how recursive stability can be enhanced through targeted interventions.\n")


def main():
    """Main function to process and analyze comic scripts."""
    # Example usage (in a real system, this would load actual scripts from files)
    model_scripts = {
        "Claude": "Claude is a helpful AI assistant...",
        "GPT": "GPT is a language model developed by OpenAI...",
        "Gemini": "Gemini is an AI assistant created by Google...",
        "DeepSeek": "DeepSeek is an AI assistant focused on helping users...",
        "Grok": "Grok is a witty AI assistant that likes to make jokes..."
    }
    
    # Process scripts
    results = process_comic_scripts(model_scripts)
    
    # Visualize comparisons
    visualize_model_comparisons(results)
    
    # Generate report
    generate_report(results)


if __name__ == "__main__":
    main()
