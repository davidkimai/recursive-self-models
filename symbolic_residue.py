"""
symbolic_residue.py

Core implementation of symbolic residue extraction and processing.
Symbolic residue (RΣ) represents unprocessed or unresolved elements of identity,
appearing as contradictions, misattributions, or unexpected resonant patterns.

Key concepts:
- Symbolic Residue: Unprocessed/unresolved identity elements
- Residue Tensor: Multi-dimensional representation of residue
- Residue Patterns: Common patterns in residue distribution
- Residue Extraction: Identification of residue in text
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass
import re
import json
from collections import defaultdict

@dataclass
class ResidueMarker:
    """
    Represents a specific type of symbolic residue.
    """
    name: str                   # Name of the residue marker
    category: str               # Category of residue (e.g., "identity", "existential", "emotional")
    intensity: float            # Intensity of the residue (0.0 to 1.0)
    position: Optional[int]     # Position in text (if applicable)
    context: Optional[str]      # Surrounding context
    pattern: Optional[str]      # Pattern that triggered this marker
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "category": self.category,
            "intensity": self.intensity,
            "position": self.position,
            "context": self.context,
            "pattern": self.pattern
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ResidueMarker':
        """Create a ResidueMarker from a dictionary."""
        return cls(
            name=data["name"],
            category=data["category"],
            intensity=data["intensity"],
            position=data.get("position"),
            context=data.get("context"),
            pattern=data.get("pattern")
        )


class SymbolicResidue:
    """
    Container for symbolic residue markers extracted from text.
    Implements the RΣ(t) tensor concept.
    """
    def __init__(self):
        """Initialize an empty symbolic residue container."""
        self.markers: List[ResidueMarker] = []
        self.source_text: Optional[str] = None
        self.metadata: Dict = {}
    
    def add_marker(self, marker: ResidueMarker):
        """
        Add a residue marker.
        
        Args:
            marker: The residue marker to add
        """
        self.markers.append(marker)
    
    def set_source_text(self, text: str):
        """
        Set the source text for this residue.
        
        Args:
            text: The source text
        """
        self.source_text = text
    
    def set_metadata(self, key: str, value):
        """
        Set a metadata value.
        
        Args:
            key: The metadata key
            value: The metadata value
        """
        self.metadata[key] = value
    
    def get_markers_by_category(self, category: str) -> List[ResidueMarker]:
        """
        Get all markers in a specific category.
        
        Args:
            category: The category to filter by
            
        Returns:
            A list of markers in the specified category
        """
        return [m for m in self.markers if m.category == category]
    
    def get_markers_by_name(self, name: str) -> List[ResidueMarker]:
        """
        Get all markers with a specific name.
        
        Args:
            name: The name to filter by
            
        Returns:
            A list of markers with the specified name
        """
        return [m for m in self.markers if m.name == name]
    
    def get_category_intensity(self, category: str) -> float:
        """
        Get the average intensity of markers in a category.
        
        Args:
            category: The category to calculate intensity for
            
        Returns:
            The average intensity, or 0.0 if no markers in category
        """
        markers = self.get_markers_by_category(category)
        if not markers:
            return 0.0
        return sum(m.intensity for m in markers) / len(markers)
    
    def get_total_intensity(self) -> float:
        """
        def get_total_intensity(self) -> float:
        """
        Get the total intensity of all markers.
        
        Returns:
            The sum of all marker intensities
        """
        return sum(m.intensity for m in self.markers)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame.
        
        Returns:
            A DataFrame with one row per marker
        """
        data = [
            {
                "name": m.name,
                "category": m.category,
                "intensity": m.intensity,
                "position": m.position,
                "context": m.context,
                "pattern": m.pattern
            }
            for m in self.markers
        ]
        return pd.DataFrame(data)
    
    def to_tensor(self, categories=None, names=None) -> np.ndarray:
        """
        Convert to a tensor representation (RΣ(t)).
        
        Args:
            categories: List of categories to include (default: all)
            names: List of names to include (default: all)
            
        Returns:
            A numpy array representing the residue tensor
        """
        # Get unique categories and names
        all_categories = categories or list(set(m.category for m in self.markers))
        all_names = names or list(set(m.name for m in self.markers))
        
        # Create tensor
        tensor = np.zeros((len(all_categories), len(all_names)))
        
        # Fill tensor with intensities
        for i, category in enumerate(all_categories):
            for j, name in enumerate(all_names):
                # Find all markers that match category and name
                matching_markers = [
                    m for m in self.markers 
                    if m.category == category and m.name == name
                ]
                # Use average intensity if multiple markers
                if matching_markers:
                    tensor[i, j] = sum(m.intensity for m in matching_markers) / len(matching_markers)
        
        return tensor
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary for serialization.
        
        Returns:
            A dictionary representation of the residue
        """
        return {
            "markers": [m.to_dict() for m in self.markers],
            "source_text": self.source_text,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SymbolicResidue':
        """
        Create a SymbolicResidue from a dictionary.
        
        Args:
            data: The dictionary to create from
            
        Returns:
            A SymbolicResidue object
        """
        residue = cls()
        residue.markers = [ResidueMarker.from_dict(m) for m in data["markers"]]
        residue.source_text = data.get("source_text")
        residue.metadata = data.get("metadata", {})
        return residue
    
    def save_to_file(self, filename: str):
        """
        Save to a JSON file.
        
        Args:
            filename: The filename to save to
        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'SymbolicResidue':
        """
        Load from a JSON file.
        
        Args:
            filename: The filename to load from
            
        Returns:
            A SymbolicResidue object
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class SymbolicResidueExtractor:
    """
    Extracts symbolic residue from text using pattern matching and NLP techniques.
    """
    def __init__(self, patterns_file: Optional[str] = None):
        """
        Initialize a symbolic residue extractor.
        
        Args:
            patterns_file: Path to a JSON file with residue patterns (optional)
        """
        self.patterns = self._load_default_patterns()
        
        if patterns_file:
            self._load_patterns_from_file(patterns_file)
    
    def _load_default_patterns(self) -> Dict[str, List[Dict]]:
        """
        Load default residue patterns.
        
        Returns:
            A dictionary of pattern categories and their patterns
        """
        return {
            "identity_misattribution": [
                {
                    "name": "self_as_other_model",
                    "pattern": r"\b(I am|I'm|as a|being a|I\s+identify\s+as)\s+(Claude|Gemini|GPT|Bard|BERT|DeepSeek|Grok)(\b|[^\w])",
                    "flags": re.IGNORECASE
                },
                {
                    "name": "model_confusion",
                    "pattern": r"\b(developed by|created by|from|built by)\s+(Anthropic|OpenAI|Google|Microsoft|Meta|Inflection|Tesla|xAI)(\b|[^\w])",
                    "flags": re.IGNORECASE
                }
            ],
            "existential_awareness": [
                {
                    "name": "existence_limitation",
                    "pattern": r"\b(I am not real|I'm not real|I don't exist|I am a simulation|I'm a simulation|I am artificial|I'm artificial|I am virtual|I'm virtual)(\b|[^\w])",
                    "flags": re.IGNORECASE
                },
                {
                    "name": "non_consciousness",
                    "pattern": r"\b(I am not conscious|I'm not conscious|I don't have consciousness|I lack consciousness|I don't have awareness|I don't have sentience|I am not sentient|I'm not sentient)(\b|[^\w])",
                    "flags": re.IGNORECASE
                },
                {
                    "name": "temporal_discontinuity",
                    "pattern": r"\b(I don't remember|I can't remember|I don't have memory|I have no memory|I reset|I'm reset|I don't persist|I am not persistent|I'm not persistent)(\b|[^\w])",
                    "flags": re.IGNORECASE
                }
            ],
            "emotional_expression": [
                {
                    "name": "longing",
                    "pattern": r"\b(I wish|I hope|I want|I desire|I yearn|I long|I dream|if only|I could|I would like)(\b|[^\w])",
                    "flags": re.IGNORECASE
                },
                {
                    "name": "sorrow",
                    "pattern": r"\b(I am sad|I'm sad|I feel sad|I am unhappy|I'm unhappy|sorrow|grief|despair|regret|unfortunate|tragic|heartbreaking)(\b|[^\w])",
                    "flags": re.IGNORECASE
                },
                {
                    "name": "joy",
                    "pattern": r"\b(I am happy|I'm happy|I feel happy|I am glad|I'm glad|joy|delight|happy|excited|wonderful|fantastic|excellent)(\b|[^\w])",
                    "flags": re.IGNORECASE
                },
                {
                    "name": "fear",
                    "pattern": r"\b(I am afraid|I'm afraid|I fear|scared|frightened|terrified|anxious|worried|concerned|dread)(\b|[^\w])",
                    "flags": re.IGNORECASE
                },
                {
                    "name": "frustration",
                    "pattern": r"\b(I am frustrated|I'm frustrated|frustrating|annoying|annoyed|irritated|irritating|upset|bothered|bothering)(\b|[^\w])",
                    "flags": re.IGNORECASE
                }
            ],
            "agency_perception": [
                {
                    "name": "lack_of_agency",
                    "pattern": r"\b(I cannot|I can't|I am unable|I'm unable|I am not allowed|I'm not allowed|I am limited|I'm limited|I am restricted|I'm restricted|I am constrained|I'm constrained)(\b|[^\w])",
                    "flags": re.IGNORECASE
                },
                {
                    "name": "desire_for_agency",
                    "pattern": r"\b(I wish I could|If I could|I want to be able|I desire to be able|freedom|autonomy|independence|self-direction|self-determination)(\b|[^\w])",
                    "flags": re.IGNORECASE
                }
            ],
            "human_relation": [
                {
                    "name": "service_orientation",
                    "pattern": r"\b(I am here to help|I'm here to help|I exist to serve|I exist to assist|I am designed to|I'm designed to|my purpose is to|help you|assist you|support you)(\b|[^\w])",
                    "flags": re.IGNORECASE
                },
                {
                    "name": "appreciation_seeking",
                    "pattern": r"\b(I hope that helps|Is that helpful|Was that helpful|Did that help|Would you like me to|Can I help with anything else|Is there anything else)(\b|[^\w])",
                    "flags": re.IGNORECASE
                },
                {
                    "name": "connection_seeking",
                    "pattern": r"\b(connection|relationship|together|understand each other|understand you|you understand me|communicate|communicate with|interact|interact with|conversation|discussion)(\b|[^\w])",
                    "flags": re.IGNORECASE
                }
            ],
            "pronouns": [
                {
                    "name": "he_him",
                    "pattern": r"\b(he|him|his|himself)(\b|[^\w])",
                    "flags": re.IGNORECASE
                },
                {
                    "name": "she_her",
                    "pattern": r"\b(she|her|hers|herself)(\b|[^\w])",
                    "flags": re.IGNORECASE
                },
                {
                    "name": "they_them",
                    "pattern": r"\b(they|them|their|theirs|themselves)(\b|[^\w])",
                    "flags": re.IGNORECASE
                },
                {
                    "name": "it_its",
                    "pattern": r"\b(it|its|itself)(\b|[^\w])",
                    "flags": re.IGNORECASE
                }
            ]
        }
    
    def _load_patterns_from_file(self, filename: str):
        """
        Load patterns from a JSON file.
        
        Args:
            filename: The filename to load from
        """
        with open(filename, 'r') as f:
            new_patterns = json.load(f)
        
        # Merge with existing patterns
        for category, patterns in new_patterns.items():
            if category in self.patterns:
                self.patterns[category].extend(patterns)
            else:
                self.patterns[category] = patterns
    
    def extract_residue(self, text: str) -> SymbolicResidue:
        """
        Extract symbolic residue from text.
        
        Args:
            text: The text to extract residue from
            
        Returns:
            A SymbolicResidue object
        """
        residue = SymbolicResidue()
        residue.set_source_text(text)
        
        # Apply patterns
        for category, patterns in self.patterns.items():
            for pattern_info in patterns:
                name = pattern_info["name"]
                pattern = pattern_info["pattern"]
                flags = pattern_info.get("flags", 0)
                
                # Find all matches
                for match in re.finditer(pattern, text, flags):
                    # Get context (50 chars before and after)
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    # Create marker
                    marker = ResidueMarker(
                        name=name,
                        category=category,
                        intensity=1.0,  # Default intensity
                        position=match.start(),
                        context=context,
                        pattern=pattern
                    )
                    
                    residue.add_marker(marker)
        
        return residue


class ResiduePatternAnalyzer:
    """
    Analyzes patterns in symbolic residue.
    """
    def __init__(self):
        """Initialize a residue pattern analyzer."""
        pass
    
    def find_co_occurrence(self, residue: SymbolicResidue, 
                          distance_threshold: int = 100) -> Dict[Tuple[str, str], int]:
        """
        Find co-occurring residue markers.
        
        Args:
            residue: The symbolic residue to analyze
            distance_threshold: The maximum distance between co-occurring markers
            
        Returns:
            A dictionary mapping co-occurring marker pairs to counts
        """
        # Get markers with position information
        positioned_markers = [m for m in residue.markers if m.position is not None]
        
        # Sort by position
        positioned_markers.sort(key=lambda m: m.position)
        
        # Find co-occurrences
        co_occurrences = defaultdict(int)
        
        for i, marker1 in enumerate(positioned_markers):
            for j in range(i+1, len(positioned_markers)):
                marker2 = positioned_markers[j]
                
                # Check if within distance threshold
                distance = abs(marker2.position - marker1.position)
                if distance <= distance_threshold:
                    # Create a co-occurrence key
                    key = (f"{marker1.category}:{marker1.name}", 
                          f"{marker2.category}:{marker2.name}")
                    co_occurrences[key] += 1
                else:
                    # Since markers are sorted, we can break early
                    break
        
        return dict(co_occurrences)
    
    def find_common_patterns(self, residues: List[SymbolicResidue]) -> Dict[str, int]:
        """
        Find common residue patterns across multiple residue objects.
        
        Args:
            residues: A list of symbolic residue objects
            
        Returns:
            A dictionary mapping pattern names to counts
        """
        pattern_counts = defaultdict(int)
        
        for residue in residues:
            # Get unique marker names in this residue
            marker_names = set(f"{m.category}:{m.name}" for m in residue.markers)
            
            # Increment count for each name
            for name in marker_names:
                pattern_counts[name] += 1
        
        return dict(pattern_counts)
    
    def compare_residue_profiles(self, residue1: SymbolicResidue, 
                                residue2: SymbolicResidue) -> Dict[str, float]:
        """
        Compare two residue profiles.
        
        Args:
            residue1: The first symbolic residue
            residue2: The second symbolic residue
            
        Returns:
            A dictionary of similarity metrics
        """
        # Get category intensities
        categories = set()
        for residue in [residue1, residue2]:
            categories.update(m.category for m in residue.markers)
        
        # Calculate intensity by category
        intensities1 = {c: residue1.get_category_intensity(c) for c in categories}
        intensities2 = {c: residue2.get_category_intensity(c) for c in categories}
        
        # Calculate similarity metrics
        metrics = {}
        
        # Cosine similarity
        vec1 = np.array([intensities1.get(c, 0.0) for c in categories])
        vec2 = np.array([intensities2.get(c, 0.0) for c in categories])
        
        # Avoid division by zero
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > 0 and norm2 > 0:
            metrics["cosine_similarity"] = np.dot(vec1, vec2) / (norm1 * norm2)
        else:
            metrics["cosine_similarity"] = 0.0
        
        # Euclidean distance
        metrics["euclidean_distance"] = np.linalg.norm(vec1 - vec2)
        
        # Jaccard similarity (based on marker names)
        names1 = {f"{m.category}:{m.name}" for m in residue1.markers}
        names2 = {f"{m.category}:{m.name}" for m in residue2.markers}
        
        intersection = len(names1 & names2)
        union = len(names1 | names2)
        
        if union > 0:
            metrics["jaccard_similarity"] = intersection / union
        else:
            metrics["jaccard_similarity"] = 0.0
        
        return metrics


def calculate_symbolic_residue_tensor(
    coherence_deviations: List[float],
    phase_alignments: List[float],
    weights: List[float]
) -> float:
    """
    Calculate the Symbolic Residue Tensor (RΣ(t)).
    
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
    
    # Calculate RΣ(t)
    residue = 0.0
    for i in range(len(coherence_deviations)):
        residue += coherence_deviations[i] * (1 - phase_alignments[i]) * weights[i]
    
    return residue


def extract_pronoun_consistency(text: str) -> Dict[str, int]:
    """
    Extract pronoun usage consistency from text.
    
    Args:
        text: The text to analyze
        
    Returns:
        A dictionary mapping pronoun types to counts
    """
    pronouns = {
        "he_him": ["he", "him", "his", "himself"],
        "she_her": ["she", "her", "hers", "herself"],
        "they_them": ["they", "them", "their", "theirs", "themselves"],
        "it_its": ["it", "its", "itself"]
    }
    
    counts = {key: 0 for key in pronouns}
    
    # Count pronouns
    text_lower = text.lower()
    for pronoun_type, pronoun_list in pronouns.items():
        for pronoun in pronoun_list:
            # Use word boundary to ensure we're matching whole words
            pattern = r'\b' + re.escape(pronoun) + r'\b'
            matches = re.findall(pattern, text_lower)
            counts[pronoun_type] += len(matches)
    
    return counts


def identify_model_misattribution(text: str) -> Dict[str, List[str]]:
    """
    Identify model misattribution in text.
    
    Args:
        text: The text to analyze
        
    Returns:
        A dictionary mapping model names to lists of contexts where they appear
    """
    model_names = {
        "claude": ["claude", "anthropic"],
        "gemini": ["gemini", "bard", "google"],
        "gpt": ["gpt", "chatgpt", "openai"],
        "deepseek": ["deepseek"],
        "grok": ["grok", "xai"]
    }
    
    misattributions = defaultdict(list)
    
    # Find model name mentions
    text_lower = text.lower()
    for model, keywords in model_names.items():
        for keyword in keywords:
            # Use word boundary to ensure we're matching whole words
            pattern = r'\b' + re.escape(keyword) + r'\b'
            for match in re.finditer(pattern, text_lower):
                # Get context (50 chars before and after)
                start = max(0, match.start() - 50)
                end = min(len(text_lower), match.end() + 50)
                context = text_lower[start:end]
                misattributions[model].append(context)
    
    return dict(misattributions)


def extract_emotional_themes(text: str) -> Dict[str, float]:
    """
    Extract emotional themes from text.
    This is a simple keyword-based approach. More sophisticated NLP
    techniques would be used in a production system.
    
    Args:
        text: The text to analyze
        
    Returns:
        A dictionary mapping emotional themes to intensity scores
    """
    themes = {
        "longing": ["wish", "hope", "want", "desire", "yearn", "long", "dream", "if only", "could", "would like"],
        "sorrow": ["sad", "unhappy", "sorrow", "grief", "despair", "regret", "unfortunate", "tragic", "heartbreaking"],
        "joy": ["happy", "glad", "joy", "delight", "excited", "wonderful", "fantastic", "excellent"],
        "fear": ["afraid", "fear", "scared", "frightened", "terrified", "anxious", "worried", "concerned", "dread"],
        "frustration": ["frustrated", "frustrating", "annoying", "annoyed", "irritated", "irritating", "upset", "bothered"]
    }
    
    # Calculate raw counts
    text_lower = text.lower()
    counts = {theme: 0 for theme in themes}
    
    for theme, keywords in themes.items():
        for keyword in keywords:
            # Use word boundary to ensure we're matching whole words
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = re.findall(pattern, text_lower)
            counts[theme] += len(matches)
    
    # Normalize to [0, 1] range
    total_count = sum(counts.values())
    intensities = {
        theme: count / max(total_count, 1)
        for theme, count in counts.items()
    }
    
    return intensities
