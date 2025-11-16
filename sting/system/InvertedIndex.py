# Proof of concept inverted index (additional testing needed)
# Potential option for managing indices on entries in the System class

from sortedcontainers import SortedList
from dataclasses import dataclass

class InvertedIndex:
    """
    Data structure for storing uniquely-identified entries with multiple indices.
    
    This structure behaves like a dictionary while also supporting user-defined 
    indices that map categories to entries. Each index can be based on entry 
    attributes and can also be ordered by entry attributes. 
    - Note: We assume every entry has a unique "id" attribute.
    - Warning: Modifications to underlying entries will not be reflected in indices!  
        To ensure the correct behavior please delete the entry first, modify it, and
        then add it back to the inverted index. 
    
    Runtime Analysis
    ----------------
    - N: the total number of entries.
    - M_i: the number of entries currently belonging to index i.
    - k: the total indices, where typically k << M_i and k << N.
        
    Operations:
    - Entry and index access: O(1)
        - Mirrors standard dict performance.
    - Adding or deleting an entry: O(k * log(M_i))
        - Each entry may belong to any of the k indices.
        - Each index maintains sorted order, accessing costs 
          log(M_i) per index, using binary search.
    - Adding a new index: O(N)
        - The index must be evaluated for all existing entries.
    """
    def __init__(self):
        self.entries = {}
        self.indices = {}
        self.index_rules = {}

    def __getitem__(self, key):
        return self.entries[key]

    def __setitem__(self, key, entry):
        entry.id = key
        self.add(entry)

    def __delitem__(self, key):
        entry = self.entries[key]
        # remove from indices
        for name in self.indices.keys():
            self.indices[name].discard(entry)
        
        del self.entries[key]

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def add(self, entry):
        """Add a new entry to the index."""
        self.entries[entry.id] = entry
        for name, rule in self.index_rules.items():
            if rule(entry):
                self.indices[name].add(entry)

    def add_index(self, name, rule, key):
        """Define an index with filtering (rule) and ordering (key)."""
        self.index_rules[name] = rule
        sorted_list = SortedList(key=key)
        for entry in self.entries.values():
            if rule(entry):
                sorted_list.add(entry)

        self.indices[name] = sorted_list

    def sel(self, name):
        """Return a list of entry objects for an index."""
        return list(self.indices[name])

    def __repr__(self):
        return f"System({self.entries!r})"

    

## Example use ##
@dataclass
class Book:
    id: int
    title: str
    pages: int
    tags: list[str]

bio1 = Book(2, "Biology 101", 350, ["science", "biology", "intro-bio"])
bio2 = Book(1, "Biology 101", 500, ["science", "biology"])
phys = Book(3, "Physics for Beginners", 120, ["science"])
poet = Book(4, "Poetry of the Night", 80, ["art"])

sys = InvertedIndex()

# Add books
sys.add(bio1)
sys.add(bio2)
sys.add(phys)
sys.add(poet)

# Index on science textbooks (sorted on title and then books id)
sys.add_index(
    "science", 
    rule=lambda b: "science" in b.tags, 
    key=lambda b: (b.title, b.id)
)
# Index on bio textbooks (sorted from least to most pages)
sys.add_index(
    "biology",
    rule=lambda b: "biology" in b.tags, 
    key=lambda b: b.pages
)
# Index on books over 100 pages (sorted by number of tags)
sys.add_index(
    "long_books",
    rule=lambda b: b.pages > 100, 
    key=lambda b: len(b.tags)
)
# Return all books sorted on their index
sys.add_index(
    "all",
    rule=lambda b: True,
    key=lambda b: (b.id)
)

# Get science textbooks
sys.sel("science")