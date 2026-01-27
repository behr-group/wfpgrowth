import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class WFPNode:
    """A node in the FP tree with weights."""

    value: Any
    weight: float
    parent: Optional["WFPNode"]
    link: Optional["WFPNode"] = None
    children: List["WFPNode"] = field(default_factory=list)

    def has_child(self, value: Any) -> bool:
        """Return ``True`` if a child with ``value`` exists."""
        return any(node.value == value for node in self.children)

    def get_child(self, value: Any) -> Optional["WFPNode"]:
        """Return a child node with a particular value if present."""
        for node in self.children:
            if node.value == value:
                return node
        return None

    def add_child(self, value: Any, weight: float=1.) -> "WFPNode":
        """Add and return a new child node."""
        child = WFPNode(value, weight, self)
        self.children.append(child)
        return child


class WFPTree:
    """A frequent pattern tree with weights."""

    def __init__(
        self,
        transactions: Iterable[Iterable[Any]],
        weights: Optional[Iterable[float]],
        threshold: float,
        root_value: Optional[Any],
        root_weight: Optional[float],
    ) -> None:
        """Initialize the tree."""
        if weights is None:
            weights = [1.] * len(transactions)

        self.frequent: Dict[Any, float] = self.find_frequent_items(transactions, weights, threshold)
        self.headers: Dict[Any, Optional[WFPNode]] = self.build_header_table(self.frequent)
        self.root: WFPNode = self.build_wfptree(
            transactions, weights, root_value, root_weight
        )

    @staticmethod
    def find_frequent_items(
        transactions: Iterable[Iterable[Any]],
        weights: Iterable[float],
        threshold: float,
    ) -> Dict[Any, float]:
        """Return items having at least ``threshold`` weight."""
        items: Dict[Any, float] = {}

        for transaction, weight in zip(transactions, weights):
            for item in transaction:
                items[item] = items.get(item, 0.) + weight

        return {item: weight for item, weight in items.items() if weight >= threshold}

    @staticmethod
    def build_header_table(frequent: Dict[Any, float]) -> Dict[Any, Optional[WFPNode]]:
        """Build the header table."""
        return {key: None for key in frequent}

    def build_wfptree(
        self,
        transactions: Iterable[Iterable[Any]],
        weights: Iterable[float],
        root_value: Optional[Any],
        root_weight: Optional[float],
    ) -> WFPNode:
        """Build the FP tree and return the root node."""
        root = WFPNode(root_value, root_weight if root_weight is not None else 1., None)

        for transaction, weight in zip(transactions, weights):
            sorted_items = sorted(
                (item for item in transaction if item in self.frequent),
                key=self.frequent.get,
                reverse=True,
            )

            if sorted_items:
                self.insert_tree(sorted_items, weight, root, self.headers)

        return root

    def insert_tree(
        self,
        items: List[Any],
        weight: float,
        node: WFPNode,
        headers: Dict[Any, Optional[WFPNode]],
    ) -> None:
        """Recursively grow FP tree."""
        first = items[0]
        child = node.get_child(first)
        if child is not None:
            child.weight += weight
        else:
            # Add new child.
            child = node.add_child(first, weight)

            # Link it to header structure.
            if headers[first] is None:
                headers[first] = child
            else:
                current = headers[first]
                while current.link is not None:
                    current = current.link
                current.link = child

        # Call function recursively.
        remaining_items = items[1:]
        if remaining_items:
            self.insert_tree(remaining_items, weight, child, headers)

    def tree_has_single_path(self, node: WFPNode) -> bool:
        """Return ``True`` if the tree has a single path starting at ``node``."""
        num_children = len(node.children)
        if num_children > 1:
            return False
        if num_children == 0:
            return True
        return self.tree_has_single_path(node.children[0])

    def mine_patterns(self, threshold: float) -> Dict[Tuple[Any, ...], float]:
        """Mine the constructed FP tree for frequent patterns."""
        if self.tree_has_single_path(self.root):
            return self.generate_pattern_list()
        return self.zip_patterns(self.mine_sub_trees(threshold))

    def zip_patterns(
        self, patterns: Dict[Tuple[Any, ...], float]
    ) -> Dict[Tuple[Any, ...], float]:
        """Append suffix to patterns in dictionary if in a conditional tree."""
        suffix = self.root.value

        if suffix is None:
            return patterns

        patterns = {
            tuple(sorted((*key, suffix))): value
            for key, value in patterns.items()
        }
        patterns[(suffix,)] = self.root.weight

        return patterns

    def generate_pattern_list(self) -> Dict[Tuple[Any, ...], float]:
        """Generate a list of patterns with support weights."""
        patterns: Dict[Tuple[Any, ...], float] = {}
        items = list(self.frequent)

        # If we are in a conditional tree, the suffix is a pattern on its own.
        if self.root.value is None:
            suffix_value: List[Any] = []
        else:
            suffix_value = [self.root.value]
            patterns[tuple(suffix_value)] = self.root.weight

        for i in range(1, len(items) + 1):
            for subset in itertools.combinations(items, i):
                pattern = tuple(sorted((*subset, *suffix_value)))
                patterns[pattern] = min(self.frequent[x] for x in subset)

        return patterns

    def mine_sub_trees(self, threshold: float) -> Dict[Tuple[Any, ...], float]:
        """Generate subtrees and mine them for patterns."""
        patterns: Dict[Tuple[Any, ...], float] = {}
        mining_order = sorted(self.frequent, key=self.frequent.get)

        # Get items in tree in reverse order of occurrences.
        for item in mining_order:
            suffixes: List[WFPNode] = []
            conditional_tree_input: List[List[Any]] = []
            conditional_weights: List[float] = []
            node = self.headers[item]

            # Follow node links to get a list of
            # all occurrences of a certain item.
            while node is not None:
                suffixes.append(node)
                node = node.link

            # For each occurrence of the item, 
            # trace the path back to the root node.
            for suffix in suffixes:
                weight = suffix.weight
                path: List[Any] = []
                parent = suffix.parent

                while parent and parent.parent is not None:
                    path.append(parent.value)
                    parent = parent.parent

                conditional_tree_input.append(path)
                conditional_weights.append(weight)

            # Now we have the input for a subtree,
            # so construct it and grab the patterns.
            subtree = WFPTree(conditional_tree_input, conditional_weights, threshold,
                             item, self.frequent[item])
            subtree_patterns = subtree.mine_patterns(threshold)

            # Insert subtree patterns into main patterns dictionary.
            for pattern in subtree_patterns.keys():
                if pattern in patterns:
                    patterns[pattern] += subtree_patterns[pattern]
                else:
                    patterns[pattern] = subtree_patterns[pattern]

        return patterns


def find_frequent_patterns(
    transactions: Iterable[Iterable[Any]],
    weights: Optional[Iterable[float]],
    support_threshold: float,
) -> Dict[Tuple[Any, ...], float]:
    """Given a set of transactions, return patterns meeting ``support_threshold``."""
    tree = WFPTree(transactions, weights, support_threshold, None, None)
    return tree.mine_patterns(support_threshold)


def generate_association_rules(
    patterns: Dict[Tuple[Any, ...], float], confidence_threshold: float
) -> Dict[Tuple[Any, ...], List[Tuple[Tuple[Any, ...], float]]]:
    """Return rules grouped by antecedent with confidence >= ``confidence_threshold``."""
    rules: Dict[Tuple[Any, ...], List[Tuple[Tuple[Any, ...], float]]] = {}
    for itemset, upper_support in patterns.items():
        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))

                if antecedent in patterns:
                    lower_support = patterns[antecedent]
                    confidence = upper_support / float(lower_support)

                    if confidence >= confidence_threshold:
                        rules.setdefault(antecedent, []).append(
                            (consequent, confidence)
                        )

    return rules
