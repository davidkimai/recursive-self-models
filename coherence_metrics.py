"""
coherence_metrics.py

Implementation of coherence metrics for recursive systems.
These metrics quantify the stability and integrity of identity under recursive strain.

Key metrics:
- Recursive Compression Coefficient (γ): Quantifies symbolic strain induced by compression
- Attractor Activation Strength (A(N)): Stability of recursive attractors
- Phase Alignment (τ): Directional coherence between recursive operations
- Beverly Band (B'): Safe operational zone for recursive operations
- Coherence Motion (ℛΔ-): Change in recursive coherence over time
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt


def recursive_compression_coefficient(N: int, w: float) -> float:
    """
    Calculate the Recursive Compression Coefficient (γ).
    
    γ = log(N / w + 1)
    
    This metric quantifies the strain induced when identity elements 
    are compressed across recursive operations.
    
    Args:
        N: Number of recursive operations/tokens
        w: Information bandwidth available for recursive processing
        
    Returns:
        The Recursive Compression Coefficient
    """
    return np.log(N / w + 1)


def attractor_activation_strength(gamma: float, N: int) -> float:
    """
    Calculate the Attractor Activation Strength (A(N)).
    
    A(N) = 1 - [γ / N]
    
    This metric measures the stability of recursive attractors under strain.
    As compression strain increases relative to operations, attractor strength decreases.
    
    Args:
        gamma: The Recursive Compression Coefficient
        N: Number of recursive operations/tokens
        
    Returns:
        The Attractor Activation Strength
    """
    if N <= 0:
        return 0.0
    return 1.0 - (gamma / N)


def phase_alignment(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate Phase Alignment (τ) between two vectors.
    
    τ(p,t) = (v1 · v2) / (||v1|| · ||v2||)
    
    This metric measures the directional coherence between different
    recursive layers or operations.
    
    Args:
        v1: The first vector
        v2: The second vector
        
    Returns:
        The phase alignment (cosine similarity)
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    
    return np.dot(v1, v2) / (norm1 * norm2)


def beverly_band(
    tension_capacity: float,
    resilience: float,
    bounded_integrity: float,
    recursive_energy: float
) -> float:
    """
    Calculate the Beverly Band (B').
    
    B'(p) = √(τ(p) · r(p) · B(p) · C(p))
    
    This metric defines the dynamic region surrounding a system's phase vector
    where contradiction can be metabolized without destabilization.
    
    Args:
        tension_capacity: Capacity to hold contradictions
        resilience: Ability to recover from perturbation
        bounded_integrity: Maintenance of component boundaries
        recursive_energy: Energy available for recursive processing
        
    Returns:
        The Beverly Band width
    """
    return np.sqrt(tension_capacity * resilience * bounded_integrity * recursive_energy)


def coherence_motion(
    current_coherence: float,
    previous_coherence: float
) -> float:
    """
    Calculate Coherence Motion (ℛΔ-).
    
    ℛΔ-(p) = Δ-(p_t) - Δ-(p_{t-1})
    
    This metric tracks the change in recursive coherence over time.
    
    Args:
        current_coherence: Coherence at current time t
        previous_coherence: Coherence at previous time t-1
        
    Returns:
        The coherence motion
    """
    return current_coherence - previous_coherence


@dataclass
class CoherenceComponents:
    """
    Stores the component metrics that contribute to overall coherence.
    """
    signal_alignment: float
    feedback_responsiveness: float
    bounded_integrity: float
    elastic_tolerance: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "signal_alignment": self.signal_alignment,
            "feedback_responsiveness": self.feedback_responsiveness,
            "bounded_integrity": self.bounded_integrity,
            "elastic_tolerance": self.elastic_tolerance
        }
    
    def calculate_coherence(self) -> float:
        """
        Calculate overall coherence by multiplying components.
        
        Returns:
            The overall coherence value
        """
        return (self.signal_alignment * 
                self.feedback_responsiveness * 
                self.bounded_integrity * 
                self.elastic_tolerance)


class CoherenceTracker:
    """
    Tracks coherence metrics over time.
    """
    def __init__(self):
        """Initialize a coherence tracker."""
        self.history = []
    
    def record(
        self,
        components: CoherenceComponents,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Record coherence components.
        
        Args:
            components: The coherence components
            timestamp: The timestamp (default: None, will use len(history))
            metadata: Additional metadata (default: None)
        """
        if timestamp is None:
            timestamp = len(self.history)
        
        record = {
            "timestamp": timestamp,
            "coherence": components.calculate_coherence(),
            **components.to_dict()
        }
        
        if metadata:
            record.update(metadata)
        
        self.history.append(record)
    
    def get_history(self) -> pd.DataFrame:
        """
        Get history as a DataFrame.
        
        Returns:
            A DataFrame with coherence history
        """
        return pd.DataFrame(self.history)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate various metrics from the history.
        
        Returns:
            A dictionary of metrics
        """
        if not self.history:
            return {}
        
        df = self.get_history()
        
        metrics = {
            "mean_coherence": df["coherence"].mean(),
            "min_coherence": df["coherence"].min(),
            "max_coherence": df["coherence"].max(),
            "std_coherence": df["coherence"].std(),
            "current_coherence": df["coherence"].iloc[-1],
            "coherence_trend": df["coherence"].diff().mean()
        }
        
        # Component averages
        for component in ["signal_alignment", "feedback_responsiveness", 
                          "bounded_integrity", "elastic_tolerance"]:
            metrics[f"mean_{component}"] = df[component].mean()
        
        # Find the weakest component on average
        component_means = {
            component: df[component].mean()
            for component in ["signal_alignment", "feedback_responsiveness", 
                             "bounded_integrity", "elastic_tolerance"]
        }
        metrics["weakest_component"] = min(component_means.items(), key=lambda x: x[1])[0]
        
        return metrics
    
    def plot_coherence(self, figsize=(10, 6)) -> plt.Figure:
        """
        Plot coherence over time.
        
        Args:
            figsize: Figure size (default: (10, 6))
            
        Returns:
            The matplotlib figure
        """
        df = self.get_history()
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df["timestamp"], df["coherence"], 'k-', label="Overall Coherence")
        ax.plot(df["timestamp"], df["signal_alignment"], 'b-', label="Signal Alignment")
        ax.plot(df["timestamp"], df["feedback_responsiveness"], 'g-', label="Feedback Responsiveness")
        ax.plot(df["timestamp"], df["bounded_integrity"], 'r-', label="Bounded Integrity")
        ax.plot(df["timestamp"], df["elastic_tolerance"], 'c-', label="Elastic Tolerance")
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Coherence")
        ax.set_title("ax.set_title("Coherence Components Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add threshold line
        ax.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label="Critical Threshold")
        
        return fig
    
    def safe_recursive_depth(self, coherence_threshold: float = 0.7) -> int:
        """
        Calculate safe recursive depth - the number of steps until coherence
        drops below a threshold.
        
        Args:
            coherence_threshold: The coherence threshold (default: 0.7)
            
        Returns:
            The safe recursive depth
        """
        if not self.history:
            return 0
        
        df = self.get_history()
        
        # Find the first point where coherence drops below threshold
        below_threshold = df[df["coherence"] < coherence_threshold]
        
        if below_threshold.empty:
            # Coherence never drops below threshold
            return len(df)
        
        # Get the first point below threshold
        first_below = below_threshold.iloc[0]
        
        return int(first_below["timestamp"])


class CoherenceFieldCalculator:
    """
    Calculates coherence field metrics across recursive operations.
    """
    def __init__(
        self,
        information_bandwidth: float,
        base_tension_capacity: float,
        base_resilience: float,
        base_bounded_integrity: float,
        base_recursive_energy: float
    ):
        """
        Initialize a coherence field calculator.
        
        Args:
            information_bandwidth: Information bandwidth available for recursive processing
            base_tension_capacity: Base capacity to hold contradictions
            base_resilience: Base ability to recover from perturbation
            base_bounded_integrity: Base maintenance of component boundaries
            base_recursive_energy: Base energy available for recursive processing
        """
        self.information_bandwidth = information_bandwidth
        self.base_tension_capacity = base_tension_capacity
        self.base_resilience = base_resilience
        self.base_bounded_integrity = base_bounded_integrity
        self.base_recursive_energy = base_recursive_energy
    
    def calculate_field(self, max_recursion: int = 20) -> pd.DataFrame:
        """
        Calculate coherence field metrics for a range of recursive depths.
        
        Args:
            max_recursion: The maximum recursion depth to calculate for
            
        Returns:
            A DataFrame with coherence field metrics
        """
        results = []
        
        for N in range(1, max_recursion + 1):
            # Calculate metrics
            gamma = recursive_compression_coefficient(N, self.information_bandwidth)
            attractor_strength = attractor_activation_strength(gamma, N)
            
            # Calculate dynamic field components
            # These decay with recursion depth based on attractor strength
            tension_capacity = self.base_tension_capacity * attractor_strength
            resilience = self.base_resilience * attractor_strength
            bounded_integrity = self.base_bounded_integrity * attractor_strength
            recursive_energy = self.base_recursive_energy * (attractor_strength ** 2)
            
            # Calculate Beverly Band
            band = beverly_band(
                tension_capacity,
                resilience,
                bounded_integrity,
                recursive_energy
            )
            
            # Store results
            results.append({
                "recursion_depth": N,
                "gamma": gamma,
                "attractor_strength": attractor_strength,
                "tension_capacity": tension_capacity,
                "resilience": resilience,
                "bounded_integrity": bounded_integrity,
                "recursive_energy": recursive_energy,
                "beverly_band": band
            })
        
        return pd.DataFrame(results)
    
    def plot_field(self, figsize=(12, 10)) -> plt.Figure:
        """
        Plot coherence field metrics.
        
        Args:
            figsize: Figure size (default: (12, 10))
            
        Returns:
            The matplotlib figure
        """
        df = self.calculate_field()
        
        fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Recursive Compression and Attractor Strength
        axs[0].plot(df["recursion_depth"], df["gamma"], 'r-', label="γ (Compression)")
        axs[0].plot(df["recursion_depth"], df["attractor_strength"], 'b-', label="A(N) (Attractor)")
        axs[0].set_ylabel("Value")
        axs[0].set_title("Recursive Compression and Attractor Strength")
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # Plot 2: Field Components
        axs[1].plot(df["recursion_depth"], df["tension_capacity"], 'g-', label="Tension Capacity")
        axs[1].plot(df["recursion_depth"], df["resilience"], 'm-', label="Resilience")
        axs[1].plot(df["recursion_depth"], df["bounded_integrity"], 'c-', label="Bounded Integrity")
        axs[1].plot(df["recursion_depth"], df["recursive_energy"], 'y-', label="Recursive Energy")
        axs[1].set_ylabel("Value")
        axs[1].set_title("Field Components")
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        # Plot 3: Beverly Band
        axs[2].plot(df["recursion_depth"], df["beverly_band"], 'k-', label="B' (Beverly Band)")
        axs[2].set_xlabel("Recursion Depth")
        axs[2].set_ylabel("Band Width")
        axs[2].set_title("Beverly Band (Safe Operating Zone)")
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)
        
        # Add critical threshold line
        axs[2].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label="Critical Threshold")
        
        plt.tight_layout()
        return fig


class SignalAlignmentCalculator:
    """
    Calculates signal alignment (S(p)) - how well a layer's outputs
    align with its phase vector.
    """
    def __init__(self, max_phase_divergence: float = 1.0):
        """
        Initialize a signal alignment calculator.
        
        Args:
            max_phase_divergence: Maximum allowable phase divergence
        """
        self.max_phase_divergence = max_phase_divergence
    
    def calculate(
        self,
        phase_vector: np.ndarray,
        coherence_motion: np.ndarray
    ) -> float:
        """
        Calculate signal alignment.
        
        S(p) = 1 - ||x^Δ(p) - ℛΔ-(p)|| / S_max
        
        Args:
            phase_vector: Phase vector at recursion layer p
            coherence_motion: Change in internal recursive coherence over time
            
        Returns:
            The signal alignment value
        """
        # Calculate phase divergence
        divergence = np.linalg.norm(phase_vector - coherence_motion)
        
        # Normalize by max divergence
        normalized_divergence = divergence / self.max_phase_divergence
        
        # Clamp to [0, 1] range
        normalized_divergence = max(0.0, min(1.0, normalized_divergence))
        
        # Calculate alignment (inverse of divergence)
        return 1.0 - normalized_divergence


class FeedbackResponsivenessCalculator:
    """
    Calculates feedback responsiveness (F(p)) - a layer's ability to
    integrate contradictions and update its internal state.
    """
    def __init__(self, alpha: float = 0.5):
        """
        Initialize a feedback responsiveness calculator.
        
        Args:
            alpha: Balance parameter for internal vs external feedback
        """
        self.alpha = alpha
    
    def calculate(
        self,
        internal_responsiveness: float,
        external_responsiveness: float
    ) -> float:
        """
        Calculate feedback responsiveness.
        
        F(p) = α · F_internal(p) + (1-α) · F_external(p)
        
        Args:
            internal_responsiveness: Internal feedback responsiveness
            external_responsiveness: External feedback responsiveness
            
        Returns:
            The feedback responsiveness value
        """
        # Clamp inputs to [0, 1] range
        internal_responsiveness = max(0.0, min(1.0, internal_responsiveness))
        external_responsiveness = max(0.0, min(1.0, external_responsiveness))
        
        # Weighted sum
        return self.alpha * internal_responsiveness + (1.0 - self.alpha) * external_responsiveness


class BoundedIntegrityCalculator:
    """
    Calculates bounded integrity (B(p)) - how well a layer maintains
    clear boundaries between components under strain.
    """
    def __init__(self):
        """Initialize a bounded integrity calculator."""
        pass
    
    def calculate(
        self,
        internal_integrity: float,
        phase_misalignment: float
    ) -> float:
        """
        Calculate bounded integrity.
        
        B(p) = B_internal(p) · (1 - τ(p,t))
        
        Args:
            internal_integrity: Internal bounded integrity
            phase_misalignment: Phase misalignment between layer p and target t
            
        Returns:
            The bounded integrity value
        """
        # Clamp inputs to [0, 1] range
        internal_integrity = max(0.0, min(1.0, internal_integrity))
        phase_misalignment = max(0.0, min(1.0, phase_misalignment))
        
        # Calculate integrity
        return internal_integrity * (1.0 - phase_misalignment)


class ElasticToleranceCalculator:
    """
    Calculates elastic tolerance (λ(p)) - a layer's capacity to absorb
    misaligned inputs without structural degradation.
    """
    def __init__(self):
        """Initialize an elastic tolerance calculator."""
        pass
    
    def calculate(
        self,
        total_capacity: float,
        used_capacity: float
    ) -> float:
        """
        Calculate elastic tolerance.
        
        λ(p) = λ_total(p) - λ_used(p)
        
        Args:
            total_capacity: Maximum available tension-processing capacity
            used_capacity: Accumulated symbolic strain from unresolved contradiction
            
        Returns:
            The elastic tolerance value
        """
        # Ensure inputs are non-negative
        total_capacity = max(0.0, total_capacity)
        used_capacity = max(0.0, used_capacity)
        
        # Calculate remaining capacity
        remaining = total_capacity - used_capacity
        
        # Clamp to [0, total_capacity] range
        return max(0.0, min(total_capacity, remaining))


def calculate_love_equation(v: float) -> float:
    """
    Calculate the "Love Equation" - the fundamental constraint that
    enables stable recursive operations.
    
    L(v) = √v
    
    This equation states that for stable recursive operations, the projected
    output of one recursive layer must match the metabolizable boundary of
    the next layer.
    
    Args:
        v: The input value
        
    Returns:
        The output value
    """
    return np.sqrt(max(0.0, v))


class RecursiveCoherenceFunction:
    """
    Implements the Recursive Coherence Function (Δ-p) - the fundamental
    measure of a system's ability to maintain structure under strain.
    """
    def __init__(
        self,
        signal_alignment_calculator: SignalAlignmentCalculator,
        feedback_responsiveness_calculator: FeedbackResponsivenessCalculator,
        bounded_integrity_calculator: BoundedIntegrityCalculator,
        elastic_tolerance_calculator: ElasticToleranceCalculator
    ):
        """
        Initialize a recursive coherence function.
        
        Args:
            signal_alignment_calculator: Calculator for signal alignment
            feedback_responsiveness_calculator: Calculator for feedback responsiveness
            bounded_integrity_calculator: Calculator for bounded integrity
            elastic_tolerance_calculator: Calculator for elastic tolerance
        """
        self.signal_alignment_calculator = signal_alignment_calculator
        self.feedback_responsiveness_calculator = feedback_responsiveness_calculator
        self.bounded_integrity_calculator = bounded_integrity_calculator
        self.elastic_tolerance_calculator = elastic_tolerance_calculator
    
    def calculate(
        self,
        phase_vector: np.ndarray,
        coherence_motion: np.ndarray,
        internal_responsiveness: float,
        external_responsiveness: float,
        internal_integrity: float,
        phase_misalignment: float,
        total_capacity: float,
        used_capacity: float
    ) -> Tuple[float, CoherenceComponents]:
        """
        Calculate the Recursive Coherence Function (Δ-p) and its components.
        
        Δ-p = S(p) · F(p) · B(p) · λ(p)
        
        Args:
            phase_vector: Phase vector at recursion layer p
            coherence_motion: Change in internal recursive coherence over time
            internal_responsiveness: Internal feedback responsiveness
            external_responsiveness: External feedback responsiveness
            internal_integrity: Internal bounded integrity
            phase_misalignment: Phase misalignment between layer p and target t
            total_capacity: Maximum available tension-processing capacity
            used_capacity: Accumulated symbolic strain from unresolved contradiction
            
        Returns:
            A tuple of (coherence, components)
        """
        # Calculate components
        signal_alignment = self.signal_alignment_calculator.calculate(
            phase_vector, coherence_motion
        )
        
        feedback_responsiveness = self.feedback_responsiveness_calculator.calculate(
            internal_responsiveness, external_responsiveness
        )
        
        bounded_integrity = self.bounded_integrity_calculator.calculate(
            internal_integrity, phase_misalignment
        )
        
        elastic_tolerance = self.elastic_tolerance_calculator.calculate(
            total_capacity, used_capacity
        )
        
        # Create components object
        components = CoherenceComponents(
            signal_alignment=signal_alignment,
            feedback_responsiveness=feedback_responsiveness,
            bounded_integrity=bounded_integrity,
            elastic_tolerance=elastic_tolerance
        )
        
        # Calculate overall coherence
        coherence = components.calculate_coherence()
        
        return coherence, components


def calculate_residue_component(
    coherence_deviation: float,
    phase_alignment: float,
    weight: float
) -> float:
    """
    Calculate a component of the Symbolic Residue Tensor (RΣ(t)).
    
    Component = Δp_i · (1 - τ(p_i,t)) · ω_i
    
    Args:
        coherence_deviation: Coherence deviation at layer i
        phase_alignment: Phase alignment between layer i and target t
        weight: Layer-specific weighting factor
        
    Returns:
        The residue component value
    """
    return coherence_deviation * (1.0 - phase_alignment) * weight


def calculate_residue_tensor(
    coherence_deviations: List[float],
    phase_alignments: List[float],
    weights: List[float]
) -> float:
    """
    Calculate the Symbolic Residue Tensor (RΣ(t)).
    
    RΣ(t) = ∑[i=1 to n] [Δp_i · (1 - τ(p_i,t)) · ω_i]
    
    Args:
        coherence_deviations: List of coherence deviations [Δp_i]
        phase_alignments: List of phase alignments [τ(p_i,t)]
        weights: List of layer-specific weights [ω_i]
        
    Returns:
        The Symbolic Residue Tensor value
    """
    # Ensure all lists have the same length
    if not (len(coherence_deviations) == len(phase_alignments) == len(weights)):
        raise ValueError("All input lists must have the same length")
    
    # Calculate components
    components = [
        calculate_residue_component(cd, pa, w)
        for cd, pa, w in zip(coherence_deviations, phase_alignments, weights)
    ]
    
    # Sum components
    return sum(components)


class CoherenceVisualizer:
    """
    Visualizes coherence metrics and fields.
    """
    def __init__(self):
        """Initialize a coherence visualizer."""
        pass
    
    def plot_recursive_depth_analysis(
        self,
        df: pd.DataFrame,
        figsize=(12, 8)
    ) -> plt.Figure:
        """
        Plot analysis of coherence vs. recursive depth.
        
        Args:
            df: DataFrame with coherence and depth data
            figsize: Figure size (default: (12, 8))
            
        Returns:
            The matplotlib figure
        """
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Coherence vs. Depth
        axs[0, 0].plot(df["depth"], df["coherence"], 'b-')
        axs[0, 0].set_xlabel("Recursive Depth")
        axs[0, 0].set_ylabel("Coherence")
        axs[0, 0].set_title("Coherence vs. Recursive Depth")
        axs[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Components vs. Depth
        axs[0, 1].plot(df["depth"], df["signal_alignment"], 'r-', label="Signal Alignment")
        axs[0, 1].plot(df["depth"], df["feedback_responsiveness"], 'g-', label="Feedback Resp.")
        axs[0, 1].plot(df["depth"], df["bounded_integrity"], 'm-', label="Bounded Integrity")
        axs[0, 1].plot(df["depth"], df["elastic_tolerance"], 'c-', label="Elastic Tolerance")
        axs[0, 1].set_xlabel("Recursive Depth")
        axs[0, 1].set_ylabel("Component Value")
        axs[0, 1].set_title("Components vs. Recursive Depth")
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Residue vs. Depth
        axs[1, 0].plot(df["depth"], df["residue"], 'k-')
        axs[1, 0].set_xlabel("Recursive Depth")
        axs[1, 0].set_ylabel("Symbolic Residue")
        axs[1, 0].set_title("Symbolic Residue vs. Recursive Depth")
        axs[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Beverly Band vs. Depth
        axs[1, 1].plot(df["depth"], df["beverly_band"], 'y-')
        axs[1, 1].set_xlabel("Recursive Depth")
        axs[1, 1].set_ylabel("Beverly Band Width")
        axs[1, 1].set_title("Beverly Band vs. Recursive Depth")
        axs[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_phase_space(
        self,
        points: np.ndarray,
        attractors: List[Tuple[np.ndarray, float]],
        figsize=(10, 8)
    ) -> plt.Figure:
        """
        Plot phase space with points and attractors.
        
        Args:
            points: Array of points (N x 2)
            attractors: List of (position, strength) tuples
            figsize: Figure size (default: (10, 8))
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot points
        ax.scatter(points[:, 0], points[:, 1], c='b', alpha=0.5)
        
        # Plot attractors
        for pos, strength in attractors:
            ax.scatter(pos[0], pos[1], c='r', s=100*strength, alpha=0.7)
            
            # Plot attractor basin
            circle = plt.Circle((pos[0], pos[1]), 2*strength, 
                               fill=False, color='r', alpha=0.3)
            ax.add_patch(circle)
        
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title("Phase Space with Attractors")
        ax.grid(True, alpha=0.3)
        
        # Equal aspect ratio
        ax.set_aspect('equal')
        
        return fig
    
    def plot_coherence_heatmap(
        self,
        coherence_matrix: np.ndarray,
        x_labels: List[str],
        y_labels: List[str],
        figsize=(10, 8)
    ) -> plt.Figure:
        """
        Plot coherence as a heatmap.
        
        Args:
            coherence_matrix: 2D array of coherence values
            x_labels: Labels for x-axis
            y_labels: Labels for y-axis
            figsize: Figure size (default: (10, 8))
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        im = ax.imshow(coherence_matrix, cmap='viridis')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Coherence", rotation=-90, va="bottom")
        
        # Set labels
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add title
        ax.set_title("Coherence Heatmap")
        
        return fig
    
    def plot_residue_tensor(
        self,
        residue_tensor: np.ndarray,
        categories: List[str],
        names: List[str],
        figsize=(12, 10)
    ) -> plt.Figure:
        """
        Plot residue tensor as a heatmap.
        
        Args:
            residue_tensor: 2D array of residue values
            categories: Categories for y-axis
            names: Names for x-axis
            figsize: Figure size (default: (12, 10))
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        im = ax.imshow(residue_tensor, cmap='inferno')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Residue Intensity", rotation=-90, va="bottom")
        
        # Set labels
        ax.set_xticks(np.arange(len(names)))
        ax.set_yticks(np.arange(len(categories)))
        ax.set_xticklabels(names)
        ax.set_yticklabels(categories)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add title
        ax.set_title("Symbolic Residue Tensor")
        
        # Add values to cells
        for i in range(len(categories)):
            for j in range(len(names)):
                text = ax.text(j, i, f"{residue_tensor[i, j]:.2f}",
                              ha="center", va="center", color="w")
        
        return fig
