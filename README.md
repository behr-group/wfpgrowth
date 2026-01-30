# wFP-Growth

A Python implementation of the Frequent Pattern Growth algorithm with optional weights for transactions.

## Installation

This package can be installed from PyPI using `pip`:
```bash
pip install wfpgrowth
```

## Usage

The main functionality is provided by function `find_frequent_patterns`, which takes transactions with corresponding weights and identifies items, which appear frequently together.

```python
from wfpgrowth import find_frequent_patterns
transactions = [(1, 2), (1, 3), (1,), (2, 3)]
weights = [3, 2, 1, 5]
find_frequent_patterns(transactions, weights, 2)
```

If instead of weights `None` is used as parameter, all transactions will be considered with weight 1.
```python
find_frequent_patterns(transactions, None, 2)
```

## Acknowledgement

This package extends [`pyfpgrowth` by Evan Dempsey](https://github.com/evandempsey/fp-growth).
