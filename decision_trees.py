# Awful hack to import past chapter modules
import math
from collections import Counter, defaultdict
from typing import List, Any, NamedTuple, TypeVar, Dict, Tuple, Union

from multiple_regression import bootstrap_sample

# Type and class definitions
DecisionNode = Union['Leaf', 'Split']
RandomForest = List[DecisionNode]
T = TypeVar('T')


class Leaf(NamedTuple):
    value: Any


class Split(NamedTuple):
    attribute: str
    subtrees: Dict[Any, DecisionNode]
    default_value: Any = None


def entropy(class_probs: List[float]) -> float:
    """
    Given a list of the class sample probabilities for ALL known classes, computes the entropy of the overall dataset
    """
    return -1 * sum([p * (math.log(p, 2) if p > 0 else 0) for p in class_probs])


def class_probabilities(labels: List[Any]) -> List[float]:
    """
    Given a list of class labels in a data-set (raw list of all observed labels), compute class probabilities for each
    """
    counts: Counter = Counter(labels)
    total = len(labels)
    return [float(count) / total for count in counts.values()]


def dataset_entropy(labels: List[Any]) -> float:
    """
    composes class probabilities and entropy to get the entropy of the whole dataset
    """
    return entropy(class_probabilities(labels))


def partition_entropy(partitions: List[List[Any]]) -> float:
    """
    Given a partitioning of a dataset, compute the entropy of that partitioning as 
    the sum of the entropy of each partition multiplied by the proportion of labels it contains
    """
    total_count: int = sum(len(part) for part in partitions)
    return sum(dataset_entropy(part) * len(part) / total_count for part in partitions)


def partition_by(points: List[T], attribute: str) -> Dict[Any, List[T]]:
    """
    Given a list of points, partition them by the attribute retrievable at <attribute>
    """   
    parts: Dict[Any, List[T]] = defaultdict(list)
    for p in points:
        key = getattr(p, attribute)
        parts[key].append(p)
    return parts


def partition_entropy_by(points: List[T], attribute: str, label_attribute: str) -> Tuple[Dict[Any, List[T]], float]:
    """
    Given a list of points, partition them by the attribute retrievable at <attribute> and compute the entropy of 
    each partition with respect to the attribute retrievable at <label_attribute>
    """
    parts: Dict[Any, List[T]] = partition_by(points, attribute)
        
    # partition_entropy requires us to process this data into just label buckets
    just_labels = []
    for key, vals in parts.items():
        just_labels.append([getattr(v, label_attribute) for v in vals])
    return parts, partition_entropy(just_labels)


def classify(node: DecisionNode, input: Any):
    """
    Given a decision tree node and input to classify with that decision tree, 
    determine the appropriate label for the input
    """
    while True:
        if isinstance(node, Leaf):
            return node.value
        else:
            # split: follow the right branch
            value = getattr(input, node.attribute)
            if value not in node.subtrees:
                return node.default_value
            # continue
            node = node.subtrees[value]


def build_tree_id3(data: List[Any], split_attributes: List[str], target_attribute: str) -> DecisionNode:
    """
    Given a dataset, a set of attributes belonging to the items in that dataset, and a target attribute, return
    the root node of a deicision tree which splits upon split_attributes, seeking to predict an inputs target
    attribute
    """
    counts = Counter([getattr(p, target_attribute) for p in data])
    most_common_target_attr = counts.most_common(1)[0][0]
    if len(counts) == 1:
        # everything has the same target attribute, just return a node with it
        return Leaf(most_common_target_attr)
    elif len(split_attributes) == 0:
        # nothing left to split on, just return the most common target attribute
        return Leaf(most_common_target_attr)
    parts_by_entropy: Dict[float, Tuple[str, Dict[Any, List[T]]]] = {}
    for attr in split_attributes:
        parts, entropy = partition_entropy_by(data, attr, target_attribute)
        # duplicate keys not an issue: if two share the lowest entropy we split on one arbitrarily
        parts_by_entropy[entropy] = (attr, parts)
    # find the partitioning with lowest entropy and create a Split with it
    attr, parts = parts_by_entropy[min(parts_by_entropy.keys())]
    subtrees: Dict[Any, DecisionNode] = {}
    for attr_val in parts:
        # recurse over sub-trees with one less split attribute
        subtrees[attr_val] = build_tree_id3(parts[attr_val], [a for a in split_attributes if a != attr],
                                            target_attribute)
    return Split(attr, subtrees, most_common_target_attr)


def random_bootstrapped_forest(data: List[Any], split_attributes: List[str], target_attribute: str, n: int = 10,
                               sampling_size: int = 0) -> RandomForest:
    """
    Train n decision trees using bootstrapped sub-samples of the input dataset
    """
    forest: RandomForest = []
    if sampling_size <= 0:
        sampling_size = len(data)
    for i in range(n):
        sub_sample: List[Any] = bootstrap_sample(data, sampling_size)
        forest.append(build_tree_id3(sub_sample, split_attributes, target_attribute))
    return forest

        
def random_forest_vote_classify(forest: RandomForest, x: Any) -> Any:
    """
    Classify using a random forest, using votes as the response mechanism
    """
    votes: List[Any] = []
    for tree in forest:
        votes.append(classify(tree, x))
    return Counter(votes).most_common(1)[0][0]
