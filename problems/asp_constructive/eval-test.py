
import numpy as np

import itertools
import math
import numpy as np

TRIPLES = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 2), (0, 2, 1), (1, 1, 1), (2, 2, 2)]
INT_TO_WEIGHT = [0, 1, 1, 2, 2, 3, 3]


def expand_admissible_set(
    pre_admissible_set: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
  """Expands a pre-admissible set into an admissible set."""
  num_groups = len(pre_admissible_set[0])
  admissible_set = []
  for row in pre_admissible_set:
    rotations = [[] for _ in range(num_groups)]
    for i in range(num_groups):
      x, y, z = TRIPLES[row[i]]
      rotations[i].append((x, y, z))
      if not x == y == z:
        rotations[i].append((z, x, y))
        rotations[i].append((y, z, x))
    product = list(itertools.product(*rotations))
    concatenated = [sum(xs, ()) for xs in product]
    admissible_set.extend(concatenated)
  return admissible_set


def get_surviving_children(extant_elements, new_element, valid_children):
  """Returns the indices of `valid_children` that remain valid after adding `new_element` to `extant_elements`."""
  bad_triples = set([
      (0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3), (0, 4, 4), (0, 5, 5),
      (0, 6, 6), (1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 2, 3), (1, 2, 4),
      (1, 3, 3), (1, 4, 4), (1, 5, 5), (1, 6, 6), (2, 2, 2), (2, 3, 3),
      (2, 4, 4), (2, 5, 5), (2, 6, 6), (3, 3, 3), (3, 3, 4), (3, 4, 4),
      (3, 4, 5), (3, 4, 6), (3, 5, 5), (3, 6, 6), (4, 4, 4), (4, 5, 5),
      (4, 6, 6), (5, 5, 5), (5, 5, 6), (5, 6, 6), (6, 6, 6)])

  # Compute.
  valid_indices = []
  for index, child in enumerate(valid_children):
    # Invalidate based on 2 elements from `new_element` and 1 element from a
    # potential child.
    if all(INT_TO_WEIGHT[x] <= INT_TO_WEIGHT[y]
           for x, y in zip(new_element, child)):
      continue
    # Invalidate based on 1 element from `new_element` and 2 elements from a
    # potential child.
    if all(INT_TO_WEIGHT[x] >= INT_TO_WEIGHT[y]
           for x, y in zip(new_element, child)):
      continue
    # Invalidate based on 1 element from `extant_elements`, 1 element from
    # `new_element`, and 1 element from a potential child.
    is_invalid = False
    for extant_element in extant_elements:
      if all(tuple(sorted((x, y, z))) in bad_triples
             for x, y, z in zip(extant_element, new_element, child)):
        is_invalid = True
        break
    if is_invalid:
      continue

    valid_indices.append(index)
  return valid_indices


def solve(n: int, w: int) -> tuple[np.ndarray, np.ndarray]:
  """Generates a symmetric constant-weight admissible set I(n, w)."""
  num_groups = n // 3
  assert 3 * num_groups == n

  # Compute the scores of all valid (weight w) children.
  valid_children = []
  for child in itertools.product(range(7), repeat=num_groups):
    weight = sum(INT_TO_WEIGHT[x] for x in child)
    if weight == w:
      valid_children.append(np.array(child, dtype=np.int32))
  valid_scores = np.array([
      priority(sum([TRIPLES[x] for x in xs], ()), n, w)
      for xs in valid_children])

  # Greedy search guided by the scores.
  pre_admissible_set = np.empty((0, num_groups), dtype=np.int32)
  while valid_children:
    max_index = np.argmax(valid_scores)
    max_child = valid_children[max_index]
    surviving_indices = get_surviving_children(pre_admissible_set, max_child,
                                               valid_children)
    valid_children = [valid_children[i] for i in surviving_indices]
    valid_scores = valid_scores[surviving_indices]

    pre_admissible_set = np.concatenate([pre_admissible_set, max_child[None]],
                                        axis=0)

  return pre_admissible_set, np.array(expand_admissible_set(pre_admissible_set))


# @funsearch.run
def evaluate(n: int, w: int) -> int:
  """Returns the size of the expanded admissible set."""
  _, admissible_set = solve(n, w)
  return len(admissible_set)


def priority(el: tuple, n: int, w: int) -> float:
    max_penalty = max(el) % 10
    diff_sum = sum((el[i] - el[i - 1]) ** 2 for i in range(1, len(el)))
    factor = 1.0  # Adjust this factor as needed
    penalty = len(el) * max(el)
    desc_penalty = sum(
        1 for i in range(1, len(el)) if el[i] < el[i - 1])  # Penalty for consecutive elements in descending order
    repeat_penalty = len(el) - len(set(el))
    score = factor * (sum(el) + diff_sum + w) - penalty + max_penalty + desc_penalty + repeat_penalty
    return score

#
# def priority(el: tuple[int, ...], n: int, w: int) -> float:
#     score = 0.0
#     for i in range(n):
#         if el[i] == 1:
#             score -= 0.9 ** (i % 4)
#         if el[i] == 2:
#             score -= 0.98 ** (30 - (i % 4))
#         if el[i] == 1 and el[i - 4] == 1:
#             score -= 0.98 ** (30 - (i % 4))
#         if el[i] == 2 and el[i - 4] != 0:
#             score -= 0.98 ** (30 - (i % 4))
#         if el[i] == 2 and el[i - 4] == 1 and el[i - 8] == 2:
#             score -= 0.98 ** (30 - (i % 4))
#             score -= 6.3
#         if el[i] == 2 and el[i - 4] == 2 and el[i - 8] == 1:
#             score -= 0.98 ** (30 - (i % 4))
#         if el[i] == 2 and el[i - 4] == 1 and el[i - 8] == 1:
#             score -= 6.3
#         if el[i] == 2 and el[i - 4] == 0 and el[i - 8] == 2:
#             score -= 6.3
#         if el[i] == 1 and el[i - 4] == 1 and el[i - 8] == 0:
#             score -= 2.2
#     return score


if __name__ == '__main__':
    print("[*] Running ...")
    n = 15
    w = 10
    obj = evaluate(n, w)
    print("[*] Obj:")
    print(obj)
