import unittest
from wfpgrowth import WFPNode, WFPTree, find_frequent_patterns, generate_association_rules

# Test from pyfpgrowth
class FPNodeTests(unittest.TestCase):
    """
    Tests for the FPNode class.
    """

    def setUp(self):
        """
        Build a node then test the features of it.
        """
        self.node = WFPNode(1, 2, None)
        self.node.add_child(2)

    def test_has_child(self):
        """
        Create a root node and test that it has no parent.
        """
        self.assertFalse(self.node.has_child(3))
        self.assertTrue(self.node.has_child(2))

    def test_get_child(self):
        """
        Test that getChild() returns a node for a valid value
        and None for an invalid value.
        """
        self.assertIsNotNone(self.node.get_child(2))
        self.assertIsNone(self.node.get_child(5))

    def test_add_child(self):
        """
        Test that addChild() successfully adds a child node.
        """
        self.assertIsNone(self.node.get_child(3))
        self.node.add_child(3)
        self.assertIsNotNone(self.node.get_child(3))
        self.assertEqual(type(self.node.get_child(3)), type(self.node))

class FPTreeTests(unittest.TestCase):
    """
    Tests for the FPTree class.
    """

    def test_build_header_table(self):
        """
        Test that buildHeaderTable() returns a dict with all None values.
        """
        tree = WFPTree([], None, 3, None, None)
        frequent = {1: 12, 2: 43, 6: 32}
        headers = tree.build_header_table(frequent)

        self.assertIsNone(headers[1])
        self.assertIsNone(headers[2])
        self.assertIsNone(headers[6])

    def test_tree_has_single_path_true(self):
        """Tree with a single path is detected correctly."""
        tree = WFPTree([[1], [1], [1]], None, 1, None, None)
        self.assertTrue(tree.tree_has_single_path(tree.root))

    def test_tree_has_single_path_false(self):
        """Tree branching results in ``False``."""
        tree = WFPTree([[1, 2], [1, 3]], None, 1, None, None)
        self.assertFalse(tree.tree_has_single_path(tree.root))

    def test_insert_tree_increments_weight(self):
        """Repeated items increment node weight."""
        tree = WFPTree([[1], [1]], None, 1, None, None)
        child = tree.root.get_child(1)
        self.assertIsNotNone(child)
        self.assertEqual(child.weight, 2.)

    def test_zip_patterns(self):
        """zip_patterns appends the suffix when present."""
        tree = WFPTree([], None, 1, 'b', 1)
        zipped = tree.zip_patterns({('a',): 2})
        self.assertEqual(zipped, {('a', 'b'): 2, ('b',): 1})  # different from pyfpgrowth

class FPGrowthTests(unittest.TestCase):
    """
    Tests everything together.
    """
    support_threshold = 2
    transactions = [[1, 2, 5],
                    [2, 4],
                    [2, 3],
                    [1, 2, 4],
                    [1, 3],
                    [2, 3],
                    [1, 3],
                    [1, 2, 3, 5],
                    [1, 2, 3]]

    def test_find_frequent_patterns(self):
        patterns = find_frequent_patterns(self.transactions, None, self.support_threshold)

        expected = {(1, 2): 4., (1, 2, 3): 2., (1, 3): 4., (1,): 6., (2,): 7., (2, 4): 2.,
                    (1, 5): 2., (5,): 2., (2, 3): 4., (2, 5): 2., (4,): 2., (1, 2, 5): 2.,
                    (3,): 6.}  # different to pyfpgrowth
        self.assertEqual(patterns, expected)

    def test_generate_association_rules(self):
        patterns = find_frequent_patterns(self.transactions, None, self.support_threshold)
        rules = generate_association_rules(patterns, 0.7)

        expected = {
            (5,): [((1,), 1.0), ((2,), 1.0), ((1, 2), 1.0)],
            (1, 5): [((2,), 1.0)],
            (2, 5): [((1,), 1.0)],
            (4,): [((2,), 1.0)],
        }
        self.assertEqual(rules, expected)

    def test_generate_association_rules_no_results(self):
        """High confidence threshold yields no rules."""
        patterns = find_frequent_patterns(self.transactions, None, self.support_threshold)
        rules = generate_association_rules(patterns, 1.1)
        self.assertEqual(rules, {})

# additional tests
class WFPGrowthTests(unittest.TestCase):
    """
    Additional tests for wfpgrowth
    """

    def test_none_weights(self):
        item_sets = ([]
            + 3 * [(1, 2)]
            + 2 * [(1, 3)]
            + 1 * [(1,)]
            + 5 * [(2, 3)]
        )
        patterns = find_frequent_patterns(item_sets, None, 2)
        expected = {(2, 3): 5., (1, 2): 3., (2,): 8., (3,): 7., (1, 3): 2., (1,): 6.}
        self.assertEqual(patterns, expected)

    def test_weighted(self):
        item_sets = [(1, 2), (1, 3), (1,), (2, 3), (1, 2, 3)]
        weights = [0.2, 0.15, 0.1, 0.5, 0.05]
        patterns = find_frequent_patterns(item_sets, weights, 0.3)
        expected = {(1,): 0.5, (2,): 0.75, (3,): 0.7, (2, 3): 0.55}
        self.assertEqual(patterns.keys(), expected.keys())
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                self.assertAlmostEqual(
                    patterns.get(pattern, 0.),
                    expected.get(pattern, 0.), 7
                )
        

    def test_named_items(self):
        item_sets = [("Bread", "Butter"), ("Bread", "Milk"), ("Bread",), ("Butter",)]
        weights = [6, 1, 3, 2]
        patterns = find_frequent_patterns(item_sets, weights, 3)
        expected = {("Bread", "Butter"): 6., ("Bread",): 10., ("Butter",): 8.}
        self.assertEqual(patterns, expected)


if __name__ == '__main__':
    import sys
    sys.exit(unittest.main())
