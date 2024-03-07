import ast
import os
import pickle
import numpy as np
import json
import copy
import random
import torch
import torch.nn as nn

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.nasbenchnlp.conversions import (convert_compact_to_recipe, convert_recipe_to_compact,
                                                          make_compact_immutable, make_compact_mutable)
from naslib.search_spaces.nasbenchnlp.encodings import encode_nlp
from naslib.utils.encodings import EncodingType
from naslib.utils import get_project_root


HIDDEN_TUPLE_SIZE = 2
INTERMEDIATE_VERTICES = 7
MAIN_OPERATIONS = ['linear', 'blend', 'elementwise_prod', 'elementwise_sum']
MAIN_WEIGHTS = [3., 1., 1., 1.]
MAIN_PROBABILITIES = np.array(MAIN_WEIGHTS) / np.sum(MAIN_WEIGHTS)
LINEAR_CONNECTIONS = [2, 3]
LINEAR_CONNECTION_WEIGHTS = [4, 1]
LINEAR_CONNECTION_PROBABILITIES = np.array(LINEAR_CONNECTION_WEIGHTS) / np.sum(LINEAR_CONNECTION_WEIGHTS)
ACTIVATIONS = ['activation_tanh', 'activation_sigm', 'activation_leaky_relu']
ACTIVATION_WEIGHTS = [1., 1., 1.]
ACTIVATION_PROBABILITIES = np.array(ACTIVATION_WEIGHTS) / np.sum(ACTIVATION_WEIGHTS)

METRIC_TO_NBNLP = {
    Metric.TRAIN_LOSS: "train_losses",
    Metric.TRAIN_TIME: "wall_times",
    Metric.VAL_LOSS: "val_losses",
    Metric.TEST_LOSS: "test_losses",
    # There's not really any such thing as "accuracy" on this benchmark, but we keep the name for compatibility.
    # These will return (100 - perplexity).
    Metric.TRAIN_ACCURACY: "train_losses",
    Metric.VAL_ACCURACY: "val_losses",
    Metric.TEST_ACCURACY: "test_losses",
}


def _generate_redundant_graph(recipe, base_nodes):
    """
    This code is from NAS-Bench-NLP https://arxiv.org/abs/2006.07116
    """
    i = 0
    activation_nodes = []
    while i < HIDDEN_TUPLE_SIZE + INTERMEDIATE_VERTICES:
        op = np.random.choice(MAIN_OPERATIONS, 1, p=MAIN_PROBABILITIES)[0]
        if op == 'linear':
            num_connections = np.random.choice(LINEAR_CONNECTIONS, 1,
                                               p=LINEAR_CONNECTION_PROBABILITIES)[0]
            connection_candidates = base_nodes + activation_nodes
            if num_connections > len(connection_candidates):
                num_connections = len(connection_candidates)

            connections = np.random.choice(connection_candidates, num_connections, replace=False)
            recipe[f'node_{i}'] = {'op': op, 'input': connections}
            i += 1

            # after linear force add activation node tied to the new node, if possible (nodes budget)
            op = np.random.choice(ACTIVATIONS, 1, p=ACTIVATION_PROBABILITIES)[0]
            recipe[f'node_{i}'] = {'op': op, 'input': [f'node_{i - 1}']}
            activation_nodes.append(f'node_{i}')
            i += 1

        elif op in ['blend', 'elementwise_prod', 'elementwise_sum']:
            # inputs must exclude x
            if op == 'blend':
                num_connections = 3
            else:
                num_connections = 2
            connection_candidates = list(set(base_nodes) - set('x')) + list(recipe.keys())
            if num_connections <= len(connection_candidates):
                connections = np.random.choice(connection_candidates, num_connections, replace=False)
                recipe[f'node_{i}'] = {'op': op, 'input': connections}
                i += 1


def _create_hidden_nodes(recipe):
    """
    This code is from NAS-Bench-NLP https://arxiv.org/abs/2006.07116
    """
    new_hiddens_map = {}
    for k in np.random.choice(list(recipe.keys()), HIDDEN_TUPLE_SIZE, replace=False):
        new_hiddens_map[k] = f'h_new_{len(new_hiddens_map)}'

    for k in new_hiddens_map:
        recipe[new_hiddens_map[k]] = recipe[k]
        del recipe[k]

    for k in recipe:
        recipe[k]['input'] = [new_hiddens_map.get(x, x) for x in recipe[k]['input']]


def _remove_redundant_nodes(recipe):
    """
    This code is from NAS-Bench-NLP https://arxiv.org/abs/2006.07116
    """
    q = [f'h_new_{i}' for i in range(HIDDEN_TUPLE_SIZE)]
    visited = set(q)
    while len(q) > 0:
        if q[0] in recipe:
            for node in recipe[q[0]]['input']:
                if node not in visited:
                    q.append(node)
                    visited.add(node)
        q = q[1:]

    for k in list(recipe.keys()):
        if k not in visited:
            del recipe[k]

    return visited


class NasBenchNLPSearchSpace(Graph):
    """
    Contains the interface to the tabular benchmark of nas-bench-nlp.
    Note: currently we do not support building a naslib object for
    nas-bench-nlp architectures.
    """

    QUERYABLE = True

    def __init__(self):
        super().__init__()
        self.load_labeled = False
        self.compact = None
        self.max_epoch = 50
        self.max_nodes = 12
        self.accs = None

    def load_labeled_architecture(self, dataset_api=None, max_nodes=12):
        """
        This is meant to be called by a new NasBenchNLPSearchSpace() object.
        It samples a random architecture from the nas-bench-nlp data.
        """
        while True:
            index = np.random.choice(len(dataset_api['nlp_arches']))
            compact = dataset_api['nlp_arches'][index]
            if len(compact[1]) <= max_nodes:
                break
        self.load_labeled = True
        self.set_compact(compact)

    def query(
        self,
        metric=None,
        dataset=None,
        path=None,
        epoch=-1,
        full_lc=False,
        dataset_api=None,
    ):
        """
        Query results from nas-bench-nlp
        """
        if dataset_api is None:
            raise NotImplementedError('Must pass in dataset_api to query NAS-Bench-301')

        if metric not in METRIC_TO_NBNLP:
            raise NotImplementedError(f"Metric not available: {metric.name}")
        orig_metric = metric
        metric = METRIC_TO_NBNLP[metric]

        if self.load_labeled:
            """
            If we loaded the architecture from the nas-bench-nlp data (using 
            load_labeled_architecture()), then self.compact will contain the architecture spec.
            """
            query_results = dataset_api["nlp_data"][self.compact]

            # Pretend like perplexities are accuracies.
            convert = (lambda x: 100 - x) if "ACC" in orig_metric.name else (lambda x: x)

            if metric == "wall_times":
                # Technically we have the full learning curve for this too, but we're only going to return single
                # values. These values will be cumulative time up to the requested epoch.
                if epoch == -1:
                    epoch = self.max_epoch
                return sum(query_results[metric][:epoch])
            elif full_lc and epoch == -1:
                # full learning curve
                return [convert(loss) for loss in query_results[metric]]
            elif full_lc and epoch != -1:
                # learning curve up to specified epoch
                return [convert(loss) for loss in query_results[metric][:epoch]]
            else:
                # return the value of the metric only at the specified epoch
                return convert(query_results[metric][epoch])
        else:
            """
            If we did not load the architecture using load_labeled_architecture(), then we can
            query the learning curve by using the nas-bench-nlp surrogate.
            The surrogate outputs a learning curve of (100 - validation loss)
            """
            if self.accs is not None:
                raise NotImplementedError("Training with extra epochs not yet supported")

            if metric == "wall_times":
                # todo: right now it uses the average train time (in seconds)
                if epoch == -1:
                    return 9747
                else:
                    return int(9747 * epoch / self.max_epoch)
            elif "LOSS" in orig_metric.name or orig_metric == Metric.TRAIN_ACCURACY:
                raise NotImplementedError(f"Metric {orig_metric.name} not available from unlabeled architectures.")

            # Apparently this is just validation accuracy, but we'll return it for test accuracy too.
            arch = encode_nlp(self, encoding_type=EncodingType.ADJACENCY_MIX, max_nodes=self.max_nodes, accs=None)
            lc = dataset_api['nlp_model'].predict(config=arch, representation='compact', search_space='nlp')
            if full_lc and epoch == -1:
                # full learning curve
                return lc
            elif full_lc and epoch != -1:
                # learning curve up to specified epoch
                return lc[:epoch]
            else:
                # return the value of the metric only at the specified epoch
                return lc[epoch]

    def get_compact(self):
        assert self.compact is not None
        return self.compact
    
    def get_hash(self):
        return self.get_compact()

    def __str__(self) -> str:
        return str(convert_compact_to_recipe(self.get_compact()))

    def set_from_string(self, arch_str: str) -> None:
        self.set_spec(convert_recipe_to_compact(ast.literal_eval(arch_str)))

    def set_compact(self, compact):
        self.compact = make_compact_immutable(compact)

    def get_arch_iterator(self, dataset_api=None):
        # currently set up for nasbenchnlp data, not surrogate
        arch_list = np.array(dataset_api["nlp_arches"])
        random.shuffle(arch_list)
        return arch_list

    def set_spec(self, compact, dataset_api=None):
        # this is just to unify the setters across search spaces
        # TODO: change it to set_spec on all search spaces
        self.set_compact(compact)

    def sample_random_architecture(self, dataset_api):

        while True:
            prev_hidden_nodes = [f'h_prev_{i}' for i in range(HIDDEN_TUPLE_SIZE)]
            base_nodes = ['x'] + prev_hidden_nodes

            recipe = {}
            _generate_redundant_graph(recipe, base_nodes)
            _create_hidden_nodes(recipe)
            visited_nodes = _remove_redundant_nodes(recipe)
            valid_recipe = True

            # check that all input nodes are in the graph
            for node in base_nodes:
                if node not in visited_nodes:
                    valid_recipe = False
                    break

            # constraint: prev hidden nodes are not connected directly to new hidden nodes
            for i in range(HIDDEN_TUPLE_SIZE):
                if len(set(recipe[f'h_new_{i}']['input']) & set(prev_hidden_nodes)) > 0:
                    valid_recipe = False
                    break

            if valid_recipe:
                compact = convert_recipe_to_compact(recipe)
                if len(compact[1]) > self.max_nodes:
                    continue
                self.set_compact(compact)
                return compact
            
    def mutate(self, parent, mutation_rate=1):
        """
        This will mutate the cell in one of two ways:
        change an edge; change an op.
        Todo: mutate by adding/removing nodes.
        Todo: mutate the list of hidden nodes.
        Todo: edges between initial hidden nodes are not mutated.
        """
        parent_compact = parent.get_compact()
        parent_compact = make_compact_mutable(parent_compact)
        compact = copy.deepcopy(parent_compact)

        edges, ops, hiddens = compact
        max_node_idx = max([max(edge) for edge in edges])

        for _ in range(int(mutation_rate)):
            mutation_type = np.random.choice(2) + 1

            if mutation_type == 0:
                # change a hidden node. Note: currently not being used
                hiddens.pop(np.random.choice(len(hiddens)))
                choices = [i for i in range(4, max_node_idx) if i not in hiddens]
                hiddens.append(np.random.choice(choices))
                hiddens.sort()

            elif mutation_type == 1:
                # change an edge
                # Currently cannot change an edge to/from an h_prev node
                edge_choices = [i for i in range(len(edges)) if edges[i][0] >= 4]
                if len(edge_choices) > 0:
                    i = np.random.choice(edge_choices)
                    node_choices = [j for j in range(4, edges[i][1])]
                    if len(node_choices) > 0:
                        edges[i][0] = np.random.choice(node_choices)

            else:
                # change an op. Note: the first 4 nodes don't have ops
                idx_choices = [i for i in range(len(ops)) if ops[i] not in [0, 6, 7]]
                if len(idx_choices) > 0:
                    idx = np.random.choice(idx_choices)
                    num_inputs = len([edge for edge in edges if edge[1] == idx])

                    # each operation can have 1, 2, [2,3], or 3 inputs only
                    groups = [[0], [1, 2, 3], [4, 5]]
                    group = groups[num_inputs]
                    choices = [i for i in group if i != ops[idx]]
                    ops[idx] = np.random.choice(choices)

        compact = (edges, ops, hiddens)
        self.set_compact(compact)

    def get_nbhd(self, dataset_api=None):
        """
        Return all neighbors of the architecture
        Currently has the same todo's as in mutate()
        """
        compact = self.get_compact()
        compact = make_compact_mutable(compact)
        edges, ops, hiddens = compact
        nbhd = []

        def add_to_nbhd(new_compact, nbhd):
            nbr = NasBenchNLPSearchSpace()
            nbr.set_compact(new_compact)
            nbr_model = torch.nn.Module()
            nbr_model.arch = nbr
            nbhd.append(nbr_model)
            return nbhd

        # add op neighbors
        idx_choices = [i for i in range(len(ops)) if ops[i] not in [0, 6, 7]]
        for idx in idx_choices:
            num_inputs = len([edge for edge in edges if edge[1] == idx])
            groups = [[0], [1, 2, 3], [4, 5]]
            group = groups[num_inputs]
            choices = [i for i in group if i != ops[idx]]
            for choice in choices:
                new_ops = ops.copy()
                new_ops[idx] = choice
                nbhd = add_to_nbhd([copy.deepcopy(edges), new_ops, hiddens.copy()], nbhd)

        # add edge neighbors
        edge_choices = [i for i in range(len(edges)) if edges[i][0] >= 4]
        for i in edge_choices:
            node_choices = [j for j in range(4, edges[i][1])]
            for j in node_choices:
                new_edges = copy.deepcopy(edges)
                new_edges[i][0] = j
                nbhd = add_to_nbhd([new_edges, ops.copy(), hiddens.copy()], nbhd)

        random.shuffle(nbhd)
        return nbhd

    def get_type(self):
        return 'nlp'

    def get_max_epochs(self):
        return 49

    def encode(self, encoding_type=EncodingType.ADJACENCY_ONE_HOT):
        return encode_nlp(self, encoding_type=encoding_type)


class NasBenchNLPQuerySpace:
    """
    Interface to query the NAS-Bench-NLP benchmark, powered by a surrogate model.
    """

    def __init__(self):
        self.compact = None
        self.max_epoch = 50
        self.max_nodes = 12

    def get_compact(self):
        return self.compact

    def get_hash(self):
        return self.get_compact()

    def __hash__(self):
        return hash(self.get_hash())

    def __str__(self) -> str:
        return str(convert_compact_to_recipe(self.get_compact()))

    def set_from_string(self, arch_str: str) -> None:
        self.set_compact(convert_recipe_to_compact(ast.literal_eval(arch_str)))

    def set_compact(self, compact):
        self.compact = make_compact_immutable(compact)

    def load_labeled_architecture(self, dataset_api=None, max_nodes=12):
        """
        This is meant to be called by a new NasBenchNLPSearchSpace() object.
        It samples a random architecture from the nas-bench-nlp data.
        """
        while True:
            index = np.random.choice(len(dataset_api["nlp_arches"]))
            compact = dataset_api["nlp_arches"][index]
            if len(compact[1]) <= max_nodes:
                break
        self.set_compact(compact)

    def sample_random_architecture(self, dataset_api: dict = None, load_labeled: bool = False) -> None:
        if load_labeled:
            self.load_labeled_architecture(dataset_api, self.max_nodes)
            return

        while True:
            prev_hidden_nodes = [f'h_prev_{i}' for i in range(HIDDEN_TUPLE_SIZE)]
            base_nodes = ['x'] + prev_hidden_nodes

            recipe = {}
            _generate_redundant_graph(recipe, base_nodes)
            _create_hidden_nodes(recipe)
            visited_nodes = _remove_redundant_nodes(recipe)
            valid_recipe = True

            # check that all input nodes are in the graph
            for node in base_nodes:
                if node not in visited_nodes:
                    valid_recipe = False
                    break

            # constraint: prev hidden nodes are not connected directly to new hidden nodes
            for i in range(HIDDEN_TUPLE_SIZE):
                if len(set(recipe[f'h_new_{i}']['input']) & set(prev_hidden_nodes)) > 0:
                    valid_recipe = False
                    break

            if valid_recipe:
                compact = convert_recipe_to_compact(recipe)
                if len(compact[1]) > self.max_nodes:
                    continue
                self.set_compact(compact)
                return

    def mutate(self, parent, mutation_rate=1):
        """
        This will mutate the cell in one of two ways:
        change an edge; change an op.
        Todo: mutate by adding/removing nodes.
        Todo: mutate the list of hidden nodes.
        Todo: edges between initial hidden nodes are not mutated.
        """
        parent_compact = parent.get_compact()
        parent_compact = make_compact_mutable(parent_compact)
        compact = copy.deepcopy(parent_compact)

        edges, ops, hiddens = compact
        max_node_idx = max([max(edge) for edge in edges])

        for _ in range(int(mutation_rate)):
            mutation_type = np.random.choice(2) + 1

            if mutation_type == 0:
                # change a hidden node. Note: currently not being used
                hiddens.pop(np.random.choice(len(hiddens)))
                choices = [i for i in range(4, max_node_idx) if i not in hiddens]
                hiddens.append(np.random.choice(choices))
                hiddens.sort()

            elif mutation_type == 1:
                # change an edge
                # Currently cannot change an edge to/from an h_prev node
                edge_choices = [i for i in range(len(edges)) if edges[i][0] >= 4]
                if len(edge_choices) > 0:
                    i = np.random.choice(edge_choices)
                    node_choices = [j for j in range(4, edges[i][1])]
                    if len(node_choices) > 0:
                        edges[i][0] = np.random.choice(node_choices)

            else:
                # change an op. Note: the first 4 nodes don't have ops
                idx_choices = [i for i in range(len(ops)) if ops[i] not in [0, 6, 7]]
                if len(idx_choices) > 0:
                    idx = np.random.choice(idx_choices)
                    num_inputs = len([edge for edge in edges if edge[1] == idx])

                    # each operation can have 1, 2, [2,3], or 3 inputs only
                    groups = [[0], [1, 2, 3], [4, 5]]
                    group = groups[num_inputs]
                    choices = [i for i in group if i != ops[idx]]
                    ops[idx] = np.random.choice(choices)

        compact = (edges, ops, hiddens)
        self.set_compact(compact)

    def get_neighbors(self, dataset_api=None):
        """
        Return all neighbors of the architecture
        Currently has the same todo's as in mutate()
        """
        compact = self.get_compact()
        assert compact is not None
        compact = make_compact_mutable(compact)
        edges, ops, hiddens = compact
        nbhd = []

        def add_to_nbhd(new_compact, nbhd):
            nbr = NasBenchNLPQuerySpace()
            nbr.set_compact(new_compact)
            nbhd.append(nbr)

        # add op neighbors
        idx_choices = [i for i in range(len(ops)) if ops[i] not in [0, 6, 7]]
        for idx in idx_choices:
            num_inputs = len([edge for edge in edges if edge[1] == idx])
            groups = [[0], [1, 2, 3], [4, 5]]
            group = groups[num_inputs]
            choices = [i for i in group if i != ops[idx]]
            for choice in choices:
                new_ops = ops.copy()
                new_ops[idx] = choice
                add_to_nbhd([copy.deepcopy(edges), new_ops, hiddens.copy()], nbhd)

        # add edge neighbors
        edge_choices = [i for i in range(len(edges)) if edges[i][0] >= 4]
        for i in edge_choices:
            node_choices = [j for j in range(4, edges[i][1])]
            for j in node_choices:
                new_edges = copy.deepcopy(edges)
                new_edges[i][0] = j
                add_to_nbhd([new_edges, ops.copy(), hiddens.copy()], nbhd)

        return nbhd

    def query(
        self,
        metric=None,
        dataset=None,
        path=None,
        epoch=-1,
        full_lc=False,
        dataset_api=None,
    ):
        """
        Query results from nas-bench-nlp
        """
        if dataset_api is None:
            raise NotImplementedError('Must pass in dataset_api to query NAS-Bench-301')

        if metric not in METRIC_TO_NBNLP:
            raise NotImplementedError(f"Metric not available: {metric.name}")
        orig_metric = metric
        metric = METRIC_TO_NBNLP[metric]

        if self.compact in dataset_api["nlp_data"]:
            # This is a labeled architecture, so we can query the train loss or val accuracy at a specific epoch
            # (also, querying will give 'real' answers, since these arches were actually trained).
            query_results = dataset_api["nlp_data"][self.compact]

            # Pretend like perplexities are accuracies.
            convert = (lambda x: 100 - x) if "ACC" in orig_metric.name else (lambda x: x)

            if metric == "wall_times":
                # Technically we have the full learning curve for this too, but we're only going to return single
                # values. These values will be cumulative time up to the requested epoch.
                if epoch == -1:
                    epoch = self.max_epoch
                return sum(query_results[metric][:epoch])
            elif full_lc and epoch == -1:
                # full learning curve
                return [convert(loss) for loss in query_results[metric]]
            elif full_lc and epoch != -1:
                # learning curve up to specified epoch
                return [convert(loss) for loss in query_results[metric][:epoch]]
            else:
                # return the value of the metric only at the specified epoch
                return convert(query_results[metric][epoch])
        else:
            # If we did not load the architecture using load_labeled_architecture(), then we can
            # query the learning curve by using the nas-bench-nlp surrogate.
            # The surrogate outputs a learning curve of (100 - validation loss)
            assert not epoch or epoch in [-1, self.max_epoch]

            if metric == "wall_times":
                # todo: right now it uses the average train time (in seconds)
                if epoch == -1:
                    return 9747
                else:
                    return int(9747 * epoch / self.max_epoch)
            elif "LOSS" in orig_metric.name or orig_metric == Metric.TRAIN_ACCURACY:
                raise NotImplementedError(f"Metric {orig_metric.name} not available from unlabeled architectures.")

            # Ensure the cache for this location is initialized, and check if this metric was already computed.
            cache = dataset_api.setdefault("nlp_cache", {}).setdefault(self.compact, {})
            if metric in cache:
                lc = cache[metric]
            else:
                # Apparently this is just validation accuracy, but we'll return it for test accuracy too.
                arch = encode_nlp(self, encoding_type=EncodingType.ADJACENCY_MIX, max_nodes=self.max_nodes, accs=None)
                lc = dataset_api["nlp_model"].predict(config=arch, representation="compact", search_space="nlp")
                dataset_api["nlp_cache"][self.compact][metric] = lc

            if full_lc and epoch == -1:
                # full learning curve
                return lc
            elif full_lc and epoch != -1:
                # learning curve up to specified epoch
                return lc[:epoch]
            else:
                # return the value of the metric only at the specified epoch
                return lc[epoch]
