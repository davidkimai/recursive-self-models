"""
recursive_trace.py

Core implementation of recursive tracing and collapse detection.
Recursive tracing maps the flow of identity concepts and attributions
through symbolic structures, and detects points of collapse or instability.

Key concepts:
- Recursive Trace: Path of concept/attribution flow through symbolic structures
- Collapse Detection: Identification of points where coherence breaks down
- Trace Visualization: Visual representation of recursive traces
- Collapse Analysis: Analysis of collapse patterns and causes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Set, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import json

@dataclass
class TraceNode:
    """
    Represents a node in a recursive trace.
    """
    id: str                     # Unique identifier for the node
    type: str                   # Type of node (e.g., "identity", "attribution", "concept")
    content: str                # Content of the node
    depth: int                  # Recursive depth of the node
    coherence: float            # Coherence value of the node
    position: Optional[int]     # Position in source text (if applicable)
    metadata: Dict[str, Any]    # Additional metadata
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "depth": self.depth,
            "coherence": self.coherence,
            "position": self.position,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TraceNode':
        """Create a TraceNode from a dictionary."""
        return cls(
            id=data["id"],
            type=data["type"],
            content=data["content"],
            depth=data["depth"],
            coherence=data["coherence"],
            position=data.get("position"),
            metadata=data.get("metadata", {})
        )


@dataclass
class TraceEdge:
    """
    Represents an edge in a recursive trace.
    """
    source_id: str              # ID of source node
    target_id: str              # ID of target node
    type: str                   # Type of edge (e.g., "attribution", "reference", "contradiction")
    weight: float               # Weight/strength of the edge
    metadata: Dict[str, Any]    # Additional metadata
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "weight": self.weight,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TraceEdge':
        """Create a TraceEdge from a dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            type=data["type"],
            weight=data["weight"],
            metadata=data.get("metadata", {})
        )


@dataclass
class CollapsePoint:
    """
    Represents a point of coherence collapse in a recursive trace.
    """
    node_id: str                # ID of the node where collapse occurred
    type: str                   # Type of collapse (e.g., "identity", "attribution", "contradiction")
    severity: float             # Severity of the collapse (0.0 to 1.0)
    cause: str                  # Cause of the collapse
    depth: int                  # Recursive depth of the collapse
    metadata: Dict[str, Any]    # Additional metadata
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "type": self.type,
            "severity": self.severity,
            "cause": self.cause,
            "depth": self.depth,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CollapsePoint':
        """Create a CollapsePoint from a dictionary."""
        return cls(
            node_id=data["node_id"],
            type=data["type"],
            severity=data["severity"],
            cause=data["cause"],
            depth=data["depth"],
            metadata=data.get("metadata", {})
        )


class RecursiveTrace:
    """
    Represents a recursive trace through symbolic structures.
    A recursive trace maps the flow of identity concepts and attributions.
    """
    def __init__(self):
        """Initialize an empty recursive trace."""
        self.nodes: Dict[str, TraceNode] = {}
        self.edges: List[TraceEdge] = []
        self.collapse_points: List[CollapsePoint] = []
        self.metadata: Dict[str, Any] = {}
        
        # Graph representation for analysis
        self.graph = nx.DiGraph()
    
    def add_node(self, node: TraceNode):
        """
        Add a node to the trace.
        
        Args:
            node: The node to add
        """
        self.nodes[node.id] = node
        self.graph.add_node(
            node.id,
            type=node.type,
            content=node.content,
            depth=node.depth,
            coherence=node.coherence,
            position=node.position,
            metadata=node.metadata
        )
    
    def add_edge(self, edge: TraceEdge):
        """
        Add an edge to the trace.
        
        Args:
            edge: The edge to add
        """
        self.edges.append(edge)
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            type=edge.type,
            weight=edge.weight,
            metadata=edge.metadata
        )
    
    def add_collapse_point(self, collapse_point: CollapsePoint):
        """
        Add a collapse point to the trace.
        
        Args:
            collapse_point: The collapse point to add
        """
        self.collapse_points.append(collapse_point)
    
    def set_metadata(self, key: str, value: Any):
        """
        Set a metadata value.
        
        Args:
            key: The metadata key
            value: The metadata value
        """
        self.metadata[key] = value
    
    def get_nodes_by_type(self, node_type: str) -> List[Trace]
def get_nodes_by_type(self, node_type: str) -> List[TraceNode]:
        """
        Get all nodes of a specific type.
        
        Args:
            node_type: The node type to filter by
            
        Returns:
            A list of nodes of the specified type
        """
        return [node for node in self.nodes.values() if node.type == node_type]
    
    def get_nodes_by_depth(self, depth: int) -> List[TraceNode]:
        """
        Get all nodes at a specific recursive depth.
        
        Args:
            depth: The recursive depth to filter by
            
        Returns:
            A list of nodes at the specified depth
        """
        return [node for node in self.nodes.values() if node.depth == depth]
    
    def get_edges_by_type(self, edge_type: str) -> List[TraceEdge]:
        """
        Get all edges of a specific type.
        
        Args:
            edge_type: The edge type to filter by
            
        Returns:
            A list of edges of the specified type
        """
        return [edge for edge in self.edges if edge.type == edge_type]
    
    def get_edges_by_node(self, node_id: str) -> List[TraceEdge]:
        """
        Get all edges connected to a specific node.
        
        Args:
            node_id: The node ID to filter by
            
        Returns:
            A list of edges connected to the specified node
        """
        return [
            edge for edge in self.edges 
            if edge.source_id == node_id or edge.target_id == node_id
        ]
    
    def get_collapse_points_by_type(self, collapse_type: str) -> List[CollapsePoint]:
        """
        Get all collapse points of a specific type.
        
        Args:
            collapse_type: The collapse type to filter by
            
        Returns:
            A list of collapse points of the specified type
        """
        return [cp for cp in self.collapse_points if cp.type == collapse_type]
    
    def get_collapse_points_by_depth(self, depth: int) -> List[CollapsePoint]:
        """
        Get all collapse points at a specific recursive depth.
        
        Args:
            depth: The recursive depth to filter by
            
        Returns:
            A list of collapse points at the specified depth
        """
        return [cp for cp in self.collapse_points if cp.depth == depth]
    
    def get_max_depth(self) -> int:
        """
        Get the maximum recursive depth in the trace.
        
        Returns:
            The maximum recursive depth
        """
        if not self.nodes:
            return 0
        return max(node.depth for node in self.nodes.values())
    
    def get_coherence_by_depth(self) -> Dict[int, float]:
        """
        Get average coherence at each recursive depth.
        
        Returns:
            A dictionary mapping depths to average coherence values
        """
        depth_coherence = defaultdict(list)
        
        for node in self.nodes.values():
            depth_coherence[node.depth].append(node.coherence)
        
        return {
            depth: sum(coherences) / len(coherences)
            for depth, coherences in depth_coherence.items()
        }
    
    def find_path(self, start_id: str, end_id: str) -> List[str]:
        """
        Find a path between two nodes.
        
        Args:
            start_id: The ID of the start node
            end_id: The ID of the end node
            
        Returns:
            A list of node IDs forming a path, or an empty list if no path exists
        """
        try:
            path = nx.shortest_path(self.graph, start_id, end_id)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def find_cycles(self) -> List[List[str]]:
        """
        Find all cycles in the trace.
        
        Returns:
            A list of cycles, where each cycle is a list of node IDs
        """
        cycles = list(nx.simple_cycles(self.graph))
        return cycles
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary for serialization.
        
        Returns:
            A dictionary representation of the trace
        """
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "collapse_points": [cp.to_dict() for cp in self.collapse_points],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RecursiveTrace':
        """
        Create a RecursiveTrace from a dictionary.
        
        Args:
            data: The dictionary to create from
            
        Returns:
            A RecursiveTrace object
        """
        trace = cls()
        
        # Add nodes
        for node_id, node_data in data["nodes"].items():
            trace.add_node(TraceNode.from_dict(node_data))
        
        # Add edges
        for edge_data in data["edges"]:
            trace.add_edge(TraceEdge.from_dict(edge_data))
        
        # Add collapse points
        for cp_data in data["collapse_points"]:
            trace.add_collapse_point(CollapsePoint.from_dict(cp_data))
        
        # Set metadata
        trace.metadata = data.get("metadata", {})
        
        return trace
    
    def save_to_file(self, filename: str):
        """
        Save to a JSON file.
        
        Args:
            filename: The filename to save to
        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'RecursiveTrace':
        """
        Load from a JSON file.
        
        Args:
            filename: The filename to load from
            
        Returns:
            A RecursiveTrace object
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
    
    def analyze_stability(self) -> Dict[str, Any]:
        """
        Analyze the stability of the trace.
        
        Returns:
            A dictionary of stability metrics
        """
        metrics = {}
        
        # Average coherence
        if self.nodes:
            metrics["average_coherence"] = sum(node.coherence for node in self.nodes.values()) / len(self.nodes)
        else:
            metrics["average_coherence"] = 0.0
        
        # Coherence by depth
        metrics["coherence_by_depth"] = self.get_coherence_by_depth()
        
        # Collapse points
        metrics["total_collapse_points"] = len(self.collapse_points)
        
        if self.collapse_points:
            metrics["average_collapse_severity"] = sum(cp.severity for cp in self.collapse_points) / len(self.collapse_points)
            metrics["collapse_types"] = {
                cp_type: len(self.get_collapse_points_by_type(cp_type))
                for cp_type in set(cp.type for cp in self.collapse_points)
            }
            metrics["collapse_by_depth"] = {
                depth: len(self.get_collapse_points_by_depth(depth))
                for depth in set(cp.depth for cp in self.collapse_points)
            }
        else:
            metrics["average_collapse_severity"] = 0.0
            metrics["collapse_types"] = {}
            metrics["collapse_by_depth"] = {}
        
        # Graph metrics
        metrics["graph_density"] = nx.density(self.graph)
        metrics["graph_reciprocity"] = nx.reciprocity(self.graph)
        
        # Connected components
        metrics["weakly_connected_components"] = nx.number_weakly_connected_components(self.graph)
        metrics["strongly_connected_components"] = nx.number_strongly_connected_components(self.graph)
        
        # Cycles
        metrics["number_of_cycles"] = len(self.find_cycles())
        
        # Path metrics
        if len(self.graph) > 1:
            try:
                metrics["average_shortest_path_length"] = nx.average_shortest_path_length(self.graph)
            except nx.NetworkXError:
                # Graph is not strongly connected
                metrics["average_shortest_path_length"] = float('inf')
        else:
            metrics["average_shortest_path_length"] = 0.0
        
        return metrics


class RecursiveTracer:
    """
    Builds recursive traces through symbolic structures.
    """
    def __init__(self):
        """Initialize a recursive tracer."""
        self.trace = RecursiveTrace()
        self.next_node_id = 0
    
    def reset(self):
        """Reset the tracer to an empty state."""
        self.trace = RecursiveTrace()
        self.next_node_id = 0
    
    def _get_next_node_id(self) -> str:
        """
        Get the next node ID.
        
        Returns:
            A unique node ID
        """
        node_id = f"node_{self.next_node_id}"
        self.next_node_id += 1
        return node_id
    
    def add_node(
        self,
        type: str,
        content: str,
        depth: int,
        coherence: float,
        position: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a node to the trace.
        
        Args:
            type: The node type
            content: The node content
            depth: The recursive depth
            coherence: The coherence value
            position: The position in source text (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            The ID of the added node
        """
        node_id = self._get_next_node_id()
        
        node = TraceNode(
            id=node_id,
            type=type,
            content=content,
            depth=depth,
            coherence=coherence,
            position=position,
            metadata=metadata or {}
        )
        
        self.trace.add_node(node)
        
        return node_id
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        type: str,
        weight: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add an edge to the trace.
        
        Args:
            source_id: The ID of the source node
            target_id: The ID of the target node
            type: The edge type
            weight: The edge weight
            metadata: Additional metadata (optional)
        """
        edge = TraceEdge(
            source_id=source_id,
            target_id=target_id,
            type=type,
            weight=weight,
            metadata=metadata or {}
        )
        
        self.trace.add_edge(edge)
    
    def add_collapse_point(
        self,
        node_id: str,
        type: str,
        severity: float,
        cause: str,
        depth: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a collapse point to the trace.
        
        Args:
            node_id: The ID of the node where collapse occurred
            type: The collapse type
            severity: The collapse severity
            cause: The collapse cause
            depth: The recursive depth
            metadata: Additional metadata (optional)
        """
        collapse_point = CollapsePoint(
            node_id=node_id,
            type=type,
            severity=severity,
            cause=cause,
            depth=depth,
            metadata=metadata or {}
        )
        
        self.trace.add_collapse_point(collapse_point)
    
    def get_trace(self) -> RecursiveTrace:
        """
        Get the current trace.
        
        Returns:
            The current RecursiveTrace
        """
        return self.trace
    
    def trace_from_text(
        self,
        text: str,
        coherence_function: callable,
        collapse_threshold: float = 0.5,
        max_depth: Optional[int] = None
    ) -> RecursiveTrace:
        """
        Build a recursive trace from text.
        This is a placeholder for more sophisticated tracing.
        
        Args:
            text: The text to trace
            coherence_function: A function that calculates coherence
            collapse_threshold: The coherence threshold for collapse detection
            max_depth: The maximum recursive depth (optional)
            
        Returns:
            The completed RecursiveTrace
        """
        # Reset tracer
        self.reset()
        
        # Simple example implementation
        # In a real system, this would involve sophisticated NLP and tracing
        
        # Split text into sentences (very simplistic)
        sentences = text.split('.')
        
        # Create root node
        root_id = self.add_node(
            type="root",
            content=text[:100] + "..." if len(text) > 100 else text,
            depth=0,
            coherence=1.0,
            position=0,
            metadata={"full_text": text}
        )
        
        # Process each sentence
        prev_node_id = root_id
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Calculate depth and coherence
            depth = min(i + 1, max_depth) if max_depth is not None else i + 1
            coherence = coherence_function(sentence, depth)
            
            # Create node
            node_id = self.add_node(
                type="sentence",
                content=sentence,
                depth=depth,
                coherence=coherence,
                position=text.find(sentence),
                metadata={"sentence_index": i}
            )
            
            # Create edge from previous node
            self.add_edge(
                source_id=prev_node_id,
                target_id=node_id,
                type="sequence",
                weight=1.0,
                metadata={"sequence_index": i}
            )
            
            # Check for collapse
            if coherence < collapse_threshold:
                self.add_collapse_point(
                    node_id=node_id,
                    type="coherence_collapse",
                    severity=(collapse_threshold - coherence) / collapse_threshold,
                    cause="Low coherence",
                    depth=depth,
                    metadata={"threshold": collapse_threshold}
                )
            
            prev_node_id = node_id
        
        return self.trace


class TraceVisualizer:
    """
    Visualizes recursive traces.
    """
    def __init__(self):
        """Initialize a trace visualizer."""
        pass
    
    def plot_trace_graph(
        self,
        trace: RecursiveTrace,
        figsize=(12, 10),
        node_size_factor=100,
        edge_width_factor=1.0,
        depth_colormap='viridis',
        coherence_colormap='RdYlGn',
        by_coherence=False
    ) -> plt.Figure:
        """
        Plot a trace as a graph.
        
        Args:
            trace: The trace to visualize
            figsize: Figure size (default: (12, 10))
            node_size_factor: Factor for node sizes (default: 100)
            edge_width_factor: Factor for edge widths (default: 1.0)
            depth_colormap: Colormap for depth (default: 'viridis')
            coherence_colormap: Colormap for coherence (default: 'RdYlGn')
            by_coherence: Color nodes by coherence instead of depth (default: False)
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get graph
        G = trace.to_networkx()
        
        # Set positions using a hierarchical layout
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        
        # Get node attributes
        node_depths = nx.get_node_attributes(G, 'depth')
        node_coherences = nx.get_node_attributes(G, 'coherence')
        node_types = nx.get_node_attributes(G, 'type')
        
        # Get edge attributes
        edge_weights = nx.get_edge_attributes(G, 'weight')
        edge_types = nx.get_edge_attributes(G, 'type')
        
        # Normalize depths and coherences for colormaps
        max_depth = max(node_depths.values()) if node_depths else 0
        
        node_colors = []
        for node in G.nodes():
            if by_coherence:
                # Color by coherence (0.0 to 1.0)
                node_colors.append(node_coherences.get(node, 0.5))
            else:
                # Color by depth (normalized to 0.0 to 1.0)
                depth = node_depths.get(node, 0)
                node_colors.append(depth / max(max_depth, 1))
        
        # Set node sizes based on coherence
        node_sizes = [node_coherences.get(node, 0.5) * node_size_factor for node in G.nodes()]
        
        # Set edge widths based on weight
        edge_widths = [edge_weights.get(edge, 0.5) * edge_width_factor for edge in G.edges()]
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            cmap=coherence_colormap if by_coherence else depth_colormap,
            alpha=0.8,
            ax=ax
        )
        
        # Draw edges
        edges = nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            alpha=0.5,
            edge_color='gray',
            arrows=True,
            ax=ax
        )
        
        # Draw labels
        labels = {node: node_types.get(node, "") for node in G.nodes()}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=8,
            ax=ax
        )
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(
            cmap=coherence_colormap if by_coherence else depth_colormap,
            norm=plt.Normalize(0, 1)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Coherence' if by_coherence else 'Depth')
        
        # Highlight collapse points
        collapse_nodes = [cp.node_id for cp in trace.collapse_points]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=collapse_nodes,
            node_color='red',
            node_shape='s',
            node_size=[200] * len(collapse_nodes),
            alpha=0.5,
            ax=ax
        )
        
        # Remove axis
        ax.axis('off')
        
        # Add title
        plt.title('Recursive Trace')
        
        return fig
    
    def plot_coherence_by_depth(
        self,
        trace: RecursiveTrace,
        figsize=(10, 6),
        show_collapse=True
    ) -> plt.Figure:
        """
        Plot coherence by recursive depth.
        
        Args:
            trace: The trace to visualize
            figsize: Figure size (default: (10, 6))
            show_collapse: Show collapse points (default: True)
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get coherence by depth
        coherence_by_depth = trace.get_coherence_by_depth()
        
        # Plot coherence
        depths = sorted(coherence_by_depth.keys())
        coherences = [coherence_by_depth[d] for d in depths]
        
        ax.plot(depths, coherences, 'b-', marker='o', label='Coherence')
        
        # Add collapse threshold line
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Collapse Threshold')
        
        # Show collapse points
        if show_collapse:
            collapse_depths = [cp.depth for cp in trace.collapse_points]
            collapse_severities = [cp.severity for cp in trace.collapse_points]
            
            # Count collapses at each depth
            depth_counts = defaultdict(int)
            for depth in collapse_depths:
                depth_counts[depth] += 1
            
            # Plot collapse counts
            collapse_x = sorted(depth_counts.keys())
            collapse_y = [depth_counts[d] for d in collapse_x]
            
            ax2 = ax.twinx()
            ax2.bar(collapse_x, collapse_y, color='r', alpha=0.3, label='Collapses')
            ax2.set_ylabel('Collapse Count')
        
        ax.set_xlabel('Recursive Depth')
        ax.set_ylabel('Coherence')
        ax.set_title('Coherence by Recursive Depth')
        
        # Set x-axis to integer ticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        if show_collapse:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
        else:
            ax.legend(loc='best')
        
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_collapse_heatmap(
        self,
        trace: RecursiveTrace,
        figsize=(12, 8)
    ) -> plt.Figure:
        """
        Plot collapse points as a heatmap.
        
        Args:
            trace: The trace to visualize
            figsize: Figure size (default: (12, 8))
            
        Returns:
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get collapse types and depths
        collapse_types = sorted(set(cp.type for cp in trace.collapse_points))
        collapse_depths = sorted(set(cp.depth for cp in trace.collapse_points))
        
        if not collapse_types or not collapse_depths:
            ax.text(0.5, 0.5, "No collapse points", ha='center', va='center')
            ax.axis('off')
            return fig
        
        # Create heatmap data
        heatmap_data = np.zeros((len(collapse_types), len(collapse_depths)))
        
        for i, cp_type in enumerate(collapse_types):
            for j, depth in enumerate(collapse_depths):
                # Find matching collapse points
                matching = [
                    cp for cp in trace.collapse_points
                    if cp.type == cp_type and cp.depth == depth
                ]
                
                # Sum severities
                heatmap_data[i, j] = sum(cp.severity for cp in matching)
        
        # Plot heatmap
        im = ax.imshow(heatmap_data, cmap='inferno')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Severity Sum", rotation=-90, va="bottom")
        
        # Set labels
        ax.set_xticks(np.arange(len(collapse_depths)))
        ax.set_yticks(np.arange(len(collapse_types)))
        ax.set_xticklabels(collapse_depths)
        ax.set_yticklabels(collapse_types)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add title
        ax.set_title("Collapse Heatmap")
        ax.set_xlabel("Recursive Depth")
        ax.set_ylabel("Collapse Type")
        
        # Add values to cells
        for i in range(len(collapse_types)):
            for j in range(len(collapse_depths)):
                if heatmap_data[i, j] > 0:
                    text = ax.text(j, i, f"{heatmap_data[i, j]:.2f}",
                                  ha="center", va="center", color="w")
        
        fig.tight_layout()
        return fig


class CollapseDetector:
    """
    Detects coherence collapse in recursive traces.
    """
    def __init__(
        self,
        coherence_threshold: float = 0.5,
        depth_penalty: float = 0.05
    ):
        """
        Initialize a collapse detector.
        
        Args:
            coherence_threshold: The coherence threshold for collapse (default: 0.5)
            depth_penalty: The coherence penalty per depth level (default: 0.05)
        """
        self.coherence_threshold = coherence_threshold
        self.depth_penalty = depth_penalty
    
    def detect_collapses(
        self,
        trace: RecursiveTrace
    ) -> List[CollapsePoint]:
        """
        Detect coherence collapses in a trace.
        
        Args:
            trace: The trace to analyze
            
        Returns:
            A list of detected collapse points
        """
        collapse_points = []
        
        # Check each node for collapse
        for node_id, node in trace.nodes.items():
            # Apply depth penalty
            adjusted_threshold = self.coherence_threshold - (node.depth * self.depth_penalty)
            adjusted_threshold = max(0.1, adjusted_threshold)  # Ensure minimum threshold
            
            if node.coherence < adjusted_threshold:
                # Calculate severity (0.0 to 1.0)
                severity = (adjusted_threshold - node.coherence) / adjusted_threshold
                
                # Create collapse point
                collapse_point = CollapsePoint(
                    node_id=node_id,
                    type="coherence_collapse",
                    severity=severity,
                    cause="Coherence below threshold",
                    depth=node.depth,
                    metadata={
                        "threshold": adjusted_threshold,
                        "coherence": node.coherence
                    }
                )
                
                collapse_points.append(collapse_point)
        
        return collapse_points
    
    def analyze_trace(
        self,
        trace: RecursiveTrace,
        update_trace: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a trace for collapses and return metrics.
        
        Args:
            trace: The trace to analyze
            update_trace: Whether to update the trace with detected collapses (default: True)
            
        Returns:
            A dictionary of collapse metrics
        """
        # Detect collapses
        collapses = self.detect_collapses(trace)
        
        # Update trace if requested
        if update_trace:
            trace.collapse_points.extend(collapses)
        
        # Calculate metrics
        metrics = {
            "total_collapses": len(collapses),
            "average_severity": (
                sum(cp.severity for cp in collapses) / len(collapses)
                if collapses else 0.0
            ),
            "collapse_by_depth": defaultdict(int)
        }
        
        # Count collapses by depth
        for cp in collapses:
            metrics["collapse_by_depth"][cp.depth] += 1
        
        # Convert to regular dict
        metrics["collapse_by_depth"] = dict(metrics["collapse_by_depth"])
        
        # Maximum collapse depth
        if collapses:
            metrics["max_collapse_depth"] = max(cp.depth for cp in collapses)
        else:
            metrics["max_collapse_depth"] = 0
        
        return metrics


def simple_coherence_function(text: str, depth: int, base_decay: float = 0.05) -> float:
    """
    A simple function to calculate text coherence, decreasing with depth.
    
    Args:
        text: The text to analyze
        depth: The recursive depth
        base_decay: Base coherence decay per depth level (default: 0.05)
        
    Returns:
        A coherence value between 0.0 and 1.0
    """
    # This is a very simplistic placeholder
    # In a real system, this would involve sophisticated coherence analysis
    
    # Basic length-based coherence (longer = more complex = lower coherence)
    length_factor = min(1.0, 100 / max(10, len(text)))
    
    # Depth decay
    depth_factor = max(0.0, 1.0 - (depth * base_decay))
    
    # Combine factors
    coherence = length_factor * depth_factor
    
    return coherence


def build_trace_from_document(
    document: str,
    coherence_function: callable = simple_coherence_function,
    max_depth: Optional[int] = None
) -> RecursiveTrace:
    """
    Build a recursive trace from a document.
    
    Args:
        document: The document to analyze
        coherence_function: Function to calculate coherence (default: simple_coherence_function)
        max_depth: Maximum recursive depth (default: None)
        
    Returns:
        A RecursiveTrace object
    """
    tracer = RecursiveTracer()
    return tracer.trace_from_text(
        text=document,
        coherence_function=coherence_function,
        max_depth=max_depth
    )
