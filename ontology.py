"""
SNOMED-CT RF2 Ontology Loader
=============================
Loads and parses real SNOMED-CT International Edition RF2 files.

SNOMED-CT RF2 File Structure:
-----------------------------
Your SNOMED-CT download should have this structure:

SnomedCT_InternationalRF2_PRODUCTION_YYYYMMDD/
├── Full/
│   ├── Terminology/
│   │   ├── sct2_Concept_Full_INT_YYYYMMDD.txt
│   │   ├── sct2_Description_Full-en_INT_YYYYMMDD.txt
│   │   ├── sct2_Relationship_Full_INT_YYYYMMDD.txt      <-- MAIN FILE FOR HIERARCHY
│   │   └── sct2_StatedRelationship_Full_INT_YYYYMMDD.txt
│   └── Refset/
│       └── ... (reference sets)
├── Snapshot/
│   └── Terminology/
│       ├── sct2_Concept_Snapshot_INT_YYYYMMDD.txt
│       ├── sct2_Relationship_Snapshot_INT_YYYYMMDD.txt  <-- USE THIS FOR CURRENT STATE
│       └── ...
└── Delta/
    └── ...

Key Files:
----------
1. sct2_Concept_*.txt - All concepts (nodes)
2. sct2_Relationship_*.txt - All relationships (edges)
3. sct2_Description_*.txt - Human-readable names for concepts

Relationship File Columns:
--------------------------
id | effectiveTime | active | moduleId | sourceId | destinationId | relationshipGroup | typeId | characteristicTypeId | modifierId

- sourceId: Child concept (more specific)
- destinationId: Parent concept (more general)  
- typeId: Type of relationship (116680003 = "Is a" for hierarchy)
- active: 1 = active, 0 = inactive

IMPORTANT: typeId = 116680003 identifies IS-A relationships (the hierarchy)
"""

import os
import csv
import pickle
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
import warnings


# SNOMED CT Constants
IS_A_RELATIONSHIP = "116680003"  # The typeId for hierarchical IS-A relationships
ROOT_CONCEPT = "138875005"  # SNOMED CT Concept (root of entire hierarchy)


@dataclass
class OntologyStats:
    """Statistics about the loaded ontology"""
    num_nodes: int
    num_edges: int
    max_depth: int
    avg_children: float
    num_roots: int
    num_leaves: int
    num_polyhierarchy: int = 0  # Nodes with multiple parents


@dataclass 
class SNOMEDConcept:
    """Represents a SNOMED-CT concept"""
    concept_id: str
    term: str = ""
    active: bool = True
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)


class SNOMEDLoader:
    """
    Loads SNOMED-CT RF2 files and builds the ontology graph.
    
    Usage:
    ------
    loader = SNOMEDLoader("/path/to/SnomedCT_InternationalRF2_PRODUCTION_YYYYMMDD")
    adj_list, stats = loader.load()
    
    # Or load from Snapshot (recommended - current state only):
    loader = SNOMEDLoader("/path/to/snomed", release_type="Snapshot")
    adj_list, stats = loader.load()
    """
    
    def __init__(self, 
                 snomed_path: str,
                 release_type: str = "Snapshot",
                 language: str = "en",
                 use_cache: bool = True,
                 cache_dir: Optional[str] = None):
        """
        Args:
            snomed_path: Path to SNOMED-CT release folder
            release_type: "Full", "Snapshot", or "Delta" (Snapshot recommended)
            language: Language code for descriptions (default: "en")
            use_cache: Whether to cache parsed results for faster reloading
            cache_dir: Directory for cache files (default: same as snomed_path)
        """
        self.snomed_path = Path(snomed_path)
        self.release_type = release_type
        self.language = language
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else self.snomed_path
        
        # Data structures
        self.concepts: Dict[str, SNOMEDConcept] = {}
        self.adj_list: Dict[int, List[int]] = {}  # parent_idx -> [child_idx]
        self.reverse_adj: Dict[int, List[int]] = {}  # child_idx -> [parent_idx]
        self.concept_id_to_idx: Dict[str, int] = {}
        self.idx_to_concept_id: Dict[int, str] = {}
        self.depth: Dict[int, int] = {}
        
    def _find_file(self, pattern: str) -> Optional[Path]:
        """Find a file matching the pattern in the terminology folder"""
        term_path = self.snomed_path / self.release_type / "Terminology"
        
        if not term_path.exists():
            # Try alternative structures
            alt_paths = [
                self.snomed_path / "Terminology",
                self.snomed_path / self.release_type,
                self.snomed_path,
            ]
            for alt in alt_paths:
                if alt.exists():
                    term_path = alt
                    break
        
        if not term_path.exists():
            raise FileNotFoundError(
                f"Could not find Terminology folder. Tried:\n"
                f"  {self.snomed_path / self.release_type / 'Terminology'}\n"
                f"  {self.snomed_path / 'Terminology'}\n"
                f"Please check your SNOMED-CT path."
            )
        
        # Search for matching files
        matches = list(term_path.glob(f"*{pattern}*"))
        if not matches:
            # Try recursive search
            matches = list(term_path.rglob(f"*{pattern}*"))
        
        if matches:
            return matches[0]
        return None
    
    def _get_cache_path(self) -> Path:
        """Get path for cache file"""
        return self.cache_dir / f"snomed_cache_{self.release_type}.pkl"
    
    def _load_from_cache(self) -> bool:
        """Try to load from cache"""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            print(f"Loading from cache: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                self.concepts = data['concepts']
                self.adj_list = data['adj_list']
                self.reverse_adj = data['reverse_adj']
                self.concept_id_to_idx = data['concept_id_to_idx']
                self.idx_to_concept_id = data['idx_to_concept_id']
                return True
            except Exception as e:
                print(f"Cache load failed: {e}, will reload from files")
        return False
    
    def _save_to_cache(self):
        """Save parsed data to cache"""
        cache_path = self._get_cache_path()
        print(f"Saving to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'concepts': self.concepts,
                'adj_list': self.adj_list,
                'reverse_adj': self.reverse_adj,
                'concept_id_to_idx': self.concept_id_to_idx,
                'idx_to_concept_id': self.idx_to_concept_id,
            }, f)
    
    def _load_concepts(self) -> Set[str]:
        """Load active concepts from concept file"""
        concept_file = self._find_file("sct2_Concept")
        if not concept_file:
            raise FileNotFoundError("Could not find sct2_Concept file")
        
        print(f"Loading concepts from: {concept_file}")
        active_concepts = set()
        
        with open(concept_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                # For Full release, we only want the most recent active version
                # For Snapshot, each concept appears once
                if row['active'] == '1':
                    concept_id = row['id']
                    active_concepts.add(concept_id)
                    if concept_id not in self.concepts:
                        self.concepts[concept_id] = SNOMEDConcept(
                            concept_id=concept_id,
                            active=True
                        )
        
        print(f"  Loaded {len(active_concepts):,} active concepts")
        return active_concepts
    
    def _load_descriptions(self, active_concepts: Set[str]):
        """Load descriptions (names) for concepts"""
        desc_file = self._find_file(f"sct2_Description")
        if not desc_file:
            print("  Warning: Could not find description file, concepts will have no names")
            return
        
        print(f"Loading descriptions from: {desc_file}")
        
        # Preferred description type: Fully Specified Name or Synonym
        FSN_TYPE = "900000000000003001"  # Fully Specified Name
        
        with open(desc_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if row['active'] == '1':
                    concept_id = row['conceptId']
                    if concept_id in self.concepts:
                        # Prefer FSN, but take any description if none exists
                        if (not self.concepts[concept_id].term or 
                            row.get('typeId') == FSN_TYPE):
                            self.concepts[concept_id].term = row['term']
    
    def _load_relationships(self, active_concepts: Set[str]):
        """
        Load IS-A relationships to build the hierarchy.
        
        In SNOMED-CT relationships:
        - sourceId = child (more specific concept)
        - destinationId = parent (more general concept)
        - typeId = 116680003 means "Is a" relationship
        """
        rel_file = self._find_file("sct2_Relationship")
        if not rel_file:
            raise FileNotFoundError("Could not find sct2_Relationship file")
        
        print(f"Loading relationships from: {rel_file}")
        edge_count = 0
        
        with open(rel_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                # Only active IS-A relationships
                if row['active'] == '1' and row['typeId'] == IS_A_RELATIONSHIP:
                    source_id = row['sourceId']  # Child
                    dest_id = row['destinationId']  # Parent
                    
                    # Only include if both concepts are active
                    if source_id in active_concepts and dest_id in active_concepts:
                        # Add to concept's parent/child lists
                        if source_id in self.concepts:
                            self.concepts[source_id].parents.append(dest_id)
                        if dest_id in self.concepts:
                            self.concepts[dest_id].children.append(source_id)
                        edge_count += 1
        
        print(f"  Loaded {edge_count:,} IS-A relationships")
    
    def _build_index_mapping(self):
        """Create contiguous integer indices for all concepts"""
        print("Building index mapping...")
        
        # Sort concept IDs for deterministic ordering
        sorted_ids = sorted(self.concepts.keys())
        
        for idx, concept_id in enumerate(sorted_ids):
            self.concept_id_to_idx[concept_id] = idx
            self.idx_to_concept_id[idx] = concept_id
        
        print(f"  Mapped {len(self.concept_id_to_idx):,} concepts to indices 0-{len(self.concept_id_to_idx)-1}")
    
    def _build_adjacency_lists(self):
        """Build adjacency lists using integer indices"""
        print("Building adjacency lists...")
        
        self.adj_list = {}
        self.reverse_adj = {i: [] for i in range(len(self.concept_id_to_idx))}
        
        for concept_id, concept in self.concepts.items():
            parent_idx = self.concept_id_to_idx[concept_id]
            
            children_idx = []
            for child_id in concept.children:
                if child_id in self.concept_id_to_idx:
                    child_idx = self.concept_id_to_idx[child_id]
                    children_idx.append(child_idx)
                    self.reverse_adj[child_idx].append(parent_idx)
            
            if children_idx:
                self.adj_list[parent_idx] = children_idx
        
        num_edges = sum(len(children) for children in self.adj_list.values())
        print(f"  Built adjacency list with {num_edges:,} edges")
    
    def _compute_depths(self) -> int:
        """Compute depth of each node using BFS from roots"""
        print("Computing node depths...")
        
        # Find roots (nodes with no parents)
        roots = []
        for idx in range(len(self.concept_id_to_idx)):
            if not self.reverse_adj.get(idx, []):
                roots.append(idx)
                self.depth[idx] = 0
        
        # BFS to compute depths
        queue = deque(roots)
        max_depth = 0
        
        while queue:
            node = queue.popleft()
            current_depth = self.depth[node]
            max_depth = max(max_depth, current_depth)
            
            for child in self.adj_list.get(node, []):
                if child not in self.depth:
                    self.depth[child] = current_depth + 1
                    queue.append(child)
        
        print(f"  Max depth: {max_depth}")
        return max_depth
    
    def _compute_stats(self) -> OntologyStats:
        """Compute statistics about the loaded ontology"""
        num_nodes = len(self.concept_id_to_idx)
        num_edges = sum(len(children) for children in self.adj_list.values())
        
        # Count roots and leaves
        num_roots = sum(1 for idx in range(num_nodes) if not self.reverse_adj.get(idx, []))
        num_leaves = sum(1 for idx in range(num_nodes) if idx not in self.adj_list or not self.adj_list[idx])
        
        # Count polyhierarchy (multiple parents)
        num_poly = sum(1 for idx in range(num_nodes) if len(self.reverse_adj.get(idx, [])) > 1)
        
        # Average children
        non_leaves = num_nodes - num_leaves
        avg_children = num_edges / max(1, non_leaves)
        
        max_depth = self._compute_depths()
        
        return OntologyStats(
            num_nodes=num_nodes,
            num_edges=num_edges,
            max_depth=max_depth,
            avg_children=avg_children,
            num_roots=num_roots,
            num_leaves=num_leaves,
            num_polyhierarchy=num_poly
        )
    
    def load(self) -> Tuple[Dict[int, List[int]], OntologyStats]:
        """
        Load SNOMED-CT and return adjacency list for DAG constraint layer.
        
        Returns:
            adj_list: Dict mapping parent_idx -> [child_indices]
            stats: OntologyStats with summary information
        """
        print(f"\n{'='*60}")
        print(f"Loading SNOMED-CT from: {self.snomed_path}")
        print(f"Release type: {self.release_type}")
        print(f"{'='*60}\n")
        
        # Try cache first
        if self.use_cache and self._load_from_cache():
            stats = self._compute_stats()
            self._print_stats(stats)
            return self.adj_list, stats
        
        # Load from files
        active_concepts = self._load_concepts()
        self._load_descriptions(active_concepts)
        self._load_relationships(active_concepts)
        self._build_index_mapping()
        self._build_adjacency_lists()
        
        # Save to cache
        if self.use_cache:
            self._save_to_cache()
        
        # Compute and print stats
        stats = self._compute_stats()
        self._print_stats(stats)
        
        return self.adj_list, stats
    
    def _print_stats(self, stats: OntologyStats):
        """Print ontology statistics"""
        print(f"\n{'='*60}")
        print("SNOMED-CT Ontology Statistics")
        print(f"{'='*60}")
        print(f"  Total concepts:     {stats.num_nodes:,}")
        print(f"  Total IS-A edges:   {stats.num_edges:,}")
        print(f"  Root concepts:      {stats.num_roots:,}")
        print(f"  Leaf concepts:      {stats.num_leaves:,}")
        print(f"  Max depth:          {stats.max_depth}")
        print(f"  Avg children:       {stats.avg_children:.2f}")
        print(f"  Polyhierarchy:      {stats.num_polyhierarchy:,} ({100*stats.num_polyhierarchy/stats.num_nodes:.1f}%)")
        print(f"{'='*60}\n")
    
    def get_ancestors(self, node_idx: int) -> Set[int]:
        """Get all ancestors of a node"""
        ancestors = set()
        queue = deque(self.reverse_adj.get(node_idx, []))
        while queue:
            parent = queue.popleft()
            if parent not in ancestors:
                ancestors.add(parent)
                queue.extend(self.reverse_adj.get(parent, []))
        return ancestors
    
    def get_concept_name(self, idx: int) -> str:
        """Get human-readable name for a concept index"""
        concept_id = self.idx_to_concept_id.get(idx, "")
        if concept_id and concept_id in self.concepts:
            return self.concepts[concept_id].term
        return f"Unknown ({idx})"
    
    def get_subtree(self, root_concept_id: str) -> Tuple[Dict[int, List[int]], Dict[str, int]]:
        """
        Extract a subtree rooted at a specific concept.
        Useful for working with specific domains (e.g., "Clinical finding").
        
        Args:
            root_concept_id: SNOMED concept ID for the subtree root
            
        Returns:
            adj_list: Adjacency list for the subtree
            id_mapping: Mapping from original concept IDs to new indices
        """
        if root_concept_id not in self.concept_id_to_idx:
            raise ValueError(f"Concept {root_concept_id} not found")
        
        root_idx = self.concept_id_to_idx[root_concept_id]
        
        # BFS to find all descendants
        descendants = {root_idx}
        queue = deque([root_idx])
        
        while queue:
            node = queue.popleft()
            for child in self.adj_list.get(node, []):
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)
        
        # Create new mapping
        new_mapping = {}
        for new_idx, old_idx in enumerate(sorted(descendants)):
            concept_id = self.idx_to_concept_id[old_idx]
            new_mapping[concept_id] = new_idx
        
        # Build new adjacency list
        old_to_new = {old: new for new, old in enumerate(sorted(descendants))}
        new_adj_list = {}
        
        for old_idx in descendants:
            new_idx = old_to_new[old_idx]
            children = [old_to_new[c] for c in self.adj_list.get(old_idx, []) if c in descendants]
            if children:
                new_adj_list[new_idx] = children
        
        print(f"Extracted subtree with {len(descendants):,} concepts")
        return new_adj_list, new_mapping


# Common SNOMED-CT top-level hierarchy concept IDs
SNOMED_HIERARCHIES = {
    "Clinical finding": "404684003",
    "Procedure": "71388002", 
    "Observable entity": "363787002",
    "Body structure": "123037004",
    "Organism": "410607006",
    "Substance": "105590001",
    "Pharmaceutical / biologic product": "373873005",
    "Specimen": "123038009",
    "Physical object": "260787004",
    "Physical force": "78621006",
    "Event": "272379006",
    "Environment or geographical location": "308916002",
    "Social context": "243796009",
    "Situation with explicit context": "243796009",
    "Staging and scales": "254291000",
    "Linkage concept": "106237007",
    "Qualifier value": "362981000",
    "Record artifact": "419891008",
    "SNOMED CT Model Component": "900000000000441003",
}


# def print_file_structure_guide():
#     """Print a guide for locating SNOMED-CT files"""
#     guide = """
#     ╔══════════════════════════════════════════════════════════════════════════════╗
#     ║                    SNOMED-CT RF2 FILE STRUCTURE GUIDE                        ║
#     ╠══════════════════════════════════════════════════════════════════════════════╣
#     ║                                                                              ║
#     ║  Your SNOMED-CT download folder should look like:                            ║
#     ║                                                                              ║
#     ║  SnomedCT_InternationalRF2_PRODUCTION_20240101T120000Z/                      ║
#     ║  ├── Full/                    <- Complete history                            ║
#     ║  │   ├── Terminology/                                                        ║
#     ║  │   │   ├── sct2_Concept_Full_INT_20240101.txt                              ║
#     ║  │   │   ├── sct2_Description_Full-en_INT_20240101.txt                       ║
#     ║  │   │   └── sct2_Relationship_Full_INT_20240101.txt    <-- HIERARCHY        ║
#     ║  │   └── Refset/                                                             ║
#     ║  │                                                                           ║
#     ║  ├── Snapshot/                <- Current state only (RECOMMENDED)            ║
#     ║  │   ├── Terminology/                                                        ║
#     ║  │   │   ├── sct2_Concept_Snapshot_INT_20240101.txt                          ║
#     ║  │   │   ├── sct2_Description_Snapshot-en_INT_20240101.txt                   ║
#     ║  │   │   └── sct2_Relationship_Snapshot_INT_20240101.txt                     ║
#     ║  │   └── Refset/                                                             ║
#     ║  │                                                                           ║
#     ║  └── Delta/                   <- Changes since last release                  ║
#     ║                                                                              ║
#     ╠══════════════════════════════════════════════════════════════════════════════╣
#     ║  KEY FILE: sct2_Relationship_Snapshot_*.txt                                  ║
#     ║                                                                              ║
#     ║  Columns:                                                                    ║
#     ║  - sourceId      = Child concept (more specific)                             ║
#     ║  - destinationId = Parent concept (more general)                             ║
#     ║  - typeId        = 116680003 for IS-A relationships                          ║
#     ║  - active        = 1 (active) or 0 (inactive)                                ║
#     ║                                                                              ║
#     ╠══════════════════════════════════════════════════════════════════════════════╣
#     ║  USAGE:                                                                      ║
#     ║                                                                              ║
#     ║  from dag_constraint import SNOMEDLoader                                     ║
#     ║                                                                              ║
#     ║  loader = SNOMEDLoader(                                                      ║
#     ║      "/path/to/SnomedCT_InternationalRF2_PRODUCTION_20240101T120000Z",       ║
#     ║      release_type="Snapshot"  # Use Snapshot for current state               ║
#     ║  )                                                                           ║
#     ║  adj_list, stats = loader.load()                                             ║
#     ║                                                                              ║
#     ╚══════════════════════════════════════════════════════════════════════════════╝
#     """
#     print(guide)


# if __name__ == "__main__":
#     print_file_structure_guide()