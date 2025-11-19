from typing import Any, Dict, Iterable, List, Tuple, Iterator, Optional

class ListMap:
    """
    List-backed map with O(1) lookup, insert, and delete operations.

    A ListMap stores values in fixed categories, where each category
    maintains a list indexed by an integer ID. Keys are (category, id)
    tuples. Reading by category (lm["math"]) returns all values from that
    category in ID order.

    You may also define *views*: named groups of categories that yield
    values in the order listed when the view was created.
    """

    def __init__(self, categories: Iterable[str]):
        categories = list(categories)

        self._categories = categories                    
        self._data: Dict[str, List[Any]] = {c: [] for c in categories}
        self._views: Dict[str, List[str]] = {}
        self._len: Dict[str, int] = {c: 0 for c in categories}

        # Warn about large unintended sparse jumps
        self.max_growth = 1024

    # ------------------------------------------------------------
    # Basic mapping operations
    # ------------------------------------------------------------

    def __getitem__(self, key: str | Tuple[str, int]):
        if isinstance(key, tuple):
            category, idx = key

            arr = self._data[category]
            if idx < 0 or idx >= len(arr):
                raise KeyError(key)

            value = arr[idx]
            if value is None:
                raise KeyError(key)

            return value

        if isinstance(key, str):
            # Return all values from given category
            return self.values(key)

        raise TypeError("Key must be a string category or (category, id) tuple.")

    def __setitem__(self, key: Tuple[str, int], value: Any):
        category, idx = key

        if idx < 0:
            raise KeyError(key)

        arr = self._data[category]

        # Grow list if needed
        if idx >= len(arr):
            n_slots = idx + 1 - len(arr)
            if n_slots > self.max_growth:
                raise UserWarning(
                    f"Attempting to grow '{category}' by {n_slots} slots "
                    f"(limit = {self.max_growth})."
                )
            arr.extend([None] * n_slots)

        # Occupancy update
        if arr[idx] is None:
            self._len[category] += 1

        arr[idx] = value

    def __delitem__(self, key: Tuple[str, int]):
        category, idx = key
        arr = self._data[category]

        if idx < 0 or idx >= len(arr):
            raise KeyError(key)

        if arr[idx] is not None:
            self._len[category] -= 1

        arr[idx] = None

    def __contains__(self, key: Tuple[str, int]):
        try:
            self[key]
            return True
        except KeyError:
            return False

    # ------------------------------------------------------------
    # Views
    # ------------------------------------------------------------

    def add_view(self, name: str, categories: List[str]):
        """Create a named view grouping categories."""
        self._views[name] = categories

    def drop_view(self, name: str):
        del self._views[name]

    def view(self, name: str):
        """Return generator of values inside the view."""
        return self.values(self._views[name])

    # ------------------------------------------------------------
    # Iteration and listing
    # ------------------------------------------------------------

    def next_open(self, category: str):
        """Get the next open ID in O(1) time."""
        return len(self._data[category])

    def values(self, categories: Optional[Iterable[str]] = None) -> Iterator[Any]:
        for _, _, v in self.items(categories):
            yield v

    def items(self, categories: Optional[Iterable[str]] = None):
        if categories is None:
            categories = self._categories
        elif isinstance(categories, str):
            categories = [categories]

        for c in categories:
            arr = self._data[c]
            for idx, item in enumerate(arr):
                if item is not None:
                    yield (c, idx, item)

    def keys(self, categories: Optional[Iterable[str]] = None):
        for cat, idx, _ in self.items(categories):
            yield (cat, idx)

    def clear(self, category: str):
        self._data[category] = []
        self._len[category] = 0

    # ------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------

    def __repr__(self):
        # Categories and counts
        data_lines = [
            f"  - ({self._len[c]}) {c}"
            for c in self._categories
        ]
        view_lines = [
            f"  - ({sum(self._len[c] for c in cats)}) {name}: {cats}"
            for name, cats in self._views.items()
        ]

        return (
            "Categories:\n"
            + ("\n".join(data_lines) or "  (none)")
            + "\n\nViews:\n"
            + ("\n".join(view_lines) or "  (none)")
        )
