"""
Title: TransNAS-Bench-101: Improving Transferability and Generalizability of Cross-Task Neural Architecture Search
Author: Duan, Yawen and Chen, Xin and Xu, Hang and Chen, Zewei and Liang, Xiaodan and Zhang, Tong and Li, Zhenguo
Date: 2021
Availability: https://github.com/yawen-d/TransNASBench
"""

import numpy as np
import random
import itertools
import torch
import torch.nn as nn

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.nasbench101.primitives import ModelWrapper
from naslib.search_spaces.nasbench301.primitives import FactorizedReduce
from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.core.primitives import Sequential
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.transbench101.conversions import (
    convert_naslib_to_op_indices,
    convert_macro_str_op_indices,
    convert_micro_str_op_indices,
    convert_op_indices_to_naslib,
    convert_op_indices_micro_to_str,
    convert_op_indices_macro_to_str,
    convert_op_indices_micro_to_model,
    convert_op_indices_macro_to_model,

)
from naslib.search_spaces.transbench101.loss import SoftmaxCrossEntropyWithLogits
from naslib.search_spaces.transbench101.encodings import (
    encode_tb101,
    encode_adjacency_one_hot_transbench_micro_op_indices,
    encode_adjacency_one_hot_transbench_macro_op_indices

)
from naslib.utils.encodings import EncodingType
import torch.nn.functional as F

OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1"]

# noinspection PyDictCreation
TASK_2_METRIC = {}
TASK_2_METRIC["class_scene"] = {
    Metric.TRAIN_ACCURACY: 'train_top1',
    Metric.VAL_ACCURACY: 'valid_top1',
    Metric.TEST_ACCURACY: 'test_top1',
    Metric.TRAIN_LOSS: 'train_loss',
    Metric.VAL_LOSS: 'valid_loss',
    Metric.TEST_LOSS: 'test_loss',
    Metric.TRAIN_TIME: 'time_elapsed',
}
TASK_2_METRIC["class_object"] = TASK_2_METRIC["class_scene"]
TASK_2_METRIC["jigsaw"] = TASK_2_METRIC["class_scene"]
TASK_2_METRIC["room_layout"] = {
    Metric.TRAIN_ACCURACY: 'train_neg_loss',
    Metric.VAL_ACCURACY: 'valid_neg_loss',
    Metric.TEST_ACCURACY: 'test_neg_loss',
    Metric.TRAIN_LOSS: 'train_loss',
    Metric.VAL_LOSS: 'valid_loss',
    Metric.TEST_LOSS: 'test_loss',
    Metric.TRAIN_TIME: 'time_elapsed',
}
TASK_2_METRIC["segmentsemantic"] = {
    Metric.TRAIN_ACCURACY: 'train_acc',
    Metric.VAL_ACCURACY: 'valid_acc',
    Metric.TEST_ACCURACY: 'test_acc',
    Metric.TRAIN_LOSS: 'train_loss',
    Metric.VAL_LOSS: 'valid_loss',
    Metric.TEST_LOSS: 'test_loss',
    Metric.TRAIN_TIME: 'time_elapsed',
}
TASK_2_METRIC["normal"] = {
    Metric.TRAIN_ACCURACY: 'train_ssim',
    Metric.VAL_ACCURACY: 'valid_ssim',
    Metric.TEST_ACCURACY: 'test_ssim',
    Metric.TRAIN_LOSS: 'train_l1_loss',
    Metric.VAL_LOSS: 'valid_l1_loss',
    Metric.TEST_LOSS: 'test_l1_loss',
    Metric.TRAIN_TIME: 'time_elapsed',
}
TASK_2_METRIC["autoencoder"] = TASK_2_METRIC["normal"]

METRIC_2_MODEL_INFO = {
    Metric.PARAMETERS: "model_params",
    Metric.FLOPS: "model_FLOPs",
}


def get_metric_conversion(tnb_metric):
    if tnb_metric.endswith("ssim") or tnb_metric.endswith("neg_loss"):
        return lambda x: x * 100
    else:
        return lambda x: x


def is_valid_macro_code(op_indices):
    op_indices = np.array(op_indices)
    return ((4 <= len(op_indices) <= 6) and  # between 4 and 6 modules
            (1 <= (op_indices > 2).sum() <= 4) and  # between 1 and 4 downsamples
            (1 <= ((op_indices % 2) == 0).sum() <= 3))  # between 1 and 3 channel doublings


def generate_all_valid_macro_codes():
    all_codes = []
    for r in (4, 5, 6):
        all_codes.extend(itertools.product(range(1, 5), repeat=r))
    return set(filter(is_valid_macro_code, all_codes))


class TransBench101SearchSpaceMicro(Graph):
    """
    Implementation of the transbench 101 search space.
    It also has an interface to the tabular benchmark of transbench 101.
    """

    OPTIMIZER_SCOPE = [
        "r_stage_1",
        "n_stage_1",
        "r_stage_2",
        "n_stage_2",
        "r_stage_3"
    ]

    QUERYABLE = True

    def __init__(self, dataset='jigsaw', use_small_model=True,
                 create_graph=False, n_classes=10, in_channels=3):
        super().__init__()
        if dataset == "jigsaw":
            self.num_classes = 1000
        elif dataset == "class_object":
            self.num_classes = 100
        elif dataset == "class_scene":
            self.num_classes = 63
        else:
            self.num_classes = n_classes
        self.op_indices = None

        self.use_small_model = use_small_model
        self.max_epoch = 199
        self.in_channels = in_channels
        self.space_name = 'transbench101'
        self.dataset = dataset
        self.create_graph = create_graph
        self.labeled_archs = None
        self.instantiate_model = True
        self.sample_without_replacement = False

        if self.create_graph == True:
            self._create_graph()
        else:
            self.add_edge(1, 2)

    def _create_graph(self):
        #
        # Cell definition
        #
        cell = Graph()
        cell.name = "cell"  # Use the same name for all cells with shared attributes

        # Input node
        cell.add_node(1)

        # Intermediate nodes
        cell.add_node(2)
        cell.add_node(3)

        # Output node
        cell.add_node(4)

        # Edges
        cell.add_edges_densly()

        #
        # Makrograph definition
        #
        self.name = "makrograph"

        self.n_modules = 3 if self.use_small_model else 5  # short: 3
        self.blocks_per_module = [2] * self.n_modules  # Change to customize number of blocks per module
        self.module_stages = ["r_stage_1", "n_stage_1", "r_stage_2", "n_stage_2", "r_stage_3"]
        # self.base_channels = 16 if self.use_small_model else 64
        self.base_channels = 64  # short: 16

        n_nodes = 1 + self.n_modules + 1  # Stem, modules, decoder

        # Add nodes and edges
        self.add_nodes_from(range(1, n_nodes + 1))
        for node in range(1, n_nodes):
            self.add_edge(node, node + 1)

        # Preprocessing for jigsaw
        self.edges[1, 2].set('op', self._get_stem_for_task(self.dataset))

        # Add modules
        for idx, node in enumerate(range(2, 2 + self.n_modules)):
            # Create module
            module = self._create_module(self.blocks_per_module[idx], self.module_stages[idx], cell)
            module.set_scope(f"module_{idx + 1}", recursively=False)

            # Add module as subgraph
            self.nodes[node]["subgraph"] = module
            module.set_input([node - 1])

        # Assign operations to cell edges
        C_in = self.base_channels
        for module_node, stage in zip(range(2, 2 + self.n_modules), self.module_stages):
            module = self.nodes[module_node]["subgraph"]
            self._set_cell_ops_for_module(module, C_in, stage)
            C_in = self._get_module_n_output_channels(module)

        # Add decoder depending on the task
        self.edges[node, node + 1].set('op',
                                       self._get_decoder_for_task(self.dataset,
                                                                  n_channels=self._get_module_n_output_channels(module))
                                       )

    def _get_stem_for_task(self, task):
        if task == "jigsaw":
            return ops.StemJigsaw(C_out=self.base_channels)
        elif task in ["class_object", "class_scene"]:
            return ops.Stem(C_out=self.base_channels)
        elif task == "autoencoder":
            return ops.Stem(C_out=self.base_channels)
        else:
            return ops.Stem(C_in=self.in_channels, C_out=self.base_channels)

    def _get_decoder_for_task(self, task, n_channels):
        if task == "jigsaw":
            return ops.SequentialJigsaw(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(n_channels * 9, self.num_classes)
            )
        elif task in ["class_object", "class_scene"]:
            return ops.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(n_channels, self.num_classes)
            )
        elif task == "autoencoder":
            if self.use_small_model:
                return ops.GenerativeDecoder((64, 32), (256, 2048))  # Short
            else:
                return ops.GenerativeDecoder((512, 32), (512, 2048))  # Full TNB

        else:
            return ops.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(n_channels, self.num_classes)
            )

    def _get_module_n_output_channels(self, module):
        last_cell_in_module = module.edges[1, 2]['op'].op[-1]
        edge_to_last_node = last_cell_in_module.edges[3, 4]
        relu_conv_bn = [op for op in edge_to_last_node['op'] if isinstance(op, ops.ReLUConvBN)][0]
        conv = [m for m in relu_conv_bn.op if isinstance(m, nn.Conv2d)][0]

        return conv.out_channels

    def _is_reduction_stage(self, stage):
        return "r_stage" in stage

    def _set_cell_ops_for_module(self, module, C_in, stage):
        assert isinstance(module, Graph)
        assert module.name == 'module'

        cells = module.edges[1, 2]['op'].op

        for idx, cell in enumerate(cells):
            downsample = self._is_reduction_stage(stage) and idx == 0
            cell.update_edges(
                update_func=lambda edge: _set_op(edge, C_in, downsample),
                private_edge_data=True
            )

            if downsample:
                C_in *= 2

    def _create_module(self, n_blocks, scope, cell):
        blocks = []
        for _ in range(n_blocks):
            blocks.append(cell.copy().set_scope(scope))

        return self._wrap_with_graph(Sequential(*blocks))

    def _wrap_with_graph(self, module):
        container = Graph()
        container.name = 'module'
        container.add_nodes_from([1, 2])
        container.add_edge(1, 2)
        container.edges[1, 2].set('op', module)
        return container

    def query(self, metric=None, dataset=None, path=None, epoch=-1, full_lc=False, dataset_api=None):
        """
        Query results from TransNAS-Bench-101.
        """
        assert isinstance(metric, Metric)
        if metric == Metric.ALL:
            raise NotImplementedError()
        if dataset_api is None:
            raise ValueError('Must pass in dataset_api to query transbench101')

        arch_str = str(self)
        query_results = dataset_api['api']

        if metric == Metric.RAW:
            # return all data
            return query_results.get_arch_result(arch_str).query_all_results()[dataset]
        if metric in (Metric.FLOPS, Metric.PARAMETERS):
            return query_results.get_model_info(arch_str, dataset, METRIC_2_MODEL_INFO[metric])

        tnb_metric = TASK_2_METRIC[dataset][metric]
        if full_lc:
            all_epochs = query_results.get_single_metric(arch_str, dataset, tnb_metric, mode="list")
            if epoch != -1:
                return all_epochs[:epoch]
            return all_epochs
        else:
            return query_results.get_single_metric(arch_str, dataset, tnb_metric, mode=epoch)

    def get_op_indices(self):
        if self.op_indices is None:
            if self.create_graph == True:
                self.op_indices = convert_naslib_to_op_indices(self)
            else:
                # if there is a model, but it's simply the original implementation of the model put on edge 1-2
                if isinstance(self.edges[1, 2]['op'], ModelWrapper):
                    raise NotImplementedError('Conversion from original model to op_indices is not implemented')
                # if there's no op indices set, and no model on edge 1-2 either
                else:
                    raise NotImplementedError('Neither op_indices nor the model is set')
        return self.op_indices

    def get_hash(self):
        return tuple(self.get_op_indices())

    def __hash__(self):
        return hash(self.get_hash())

    def __str__(self) -> str:
        return convert_op_indices_micro_to_str(self.get_op_indices())

    def set_from_string(self, arch_str: str) -> None:
        self.set_op_indices(convert_micro_str_op_indices(arch_str))

    def set_op_indices(self, op_indices):
        # This will update the edges in the naslib object to op_indices
        self.op_indices = op_indices

        if self.instantiate_model == True:
            if self.create_graph == True:
                convert_op_indices_to_naslib(op_indices, self)
            else:
                model = convert_op_indices_micro_to_model(self.op_indices, self.dataset)
                self.edges[1, 2].set('op', model)

    def set_spec(self, op_indices, dataset_api=None):
        # this is just to unify the setters across search spaces
        # TODO: change it to set_spec on all search spaces
        self.set_op_indices(op_indices)

    def get_arch_iterator(self, dataset_api=None):
        return itertools.product(range(4), repeat=6)

    def sample_random_labeled_architecture(self):
        assert self.labeled_archs is not None, "Labeled archs not provided to sample from"

        op_indices = random.choice(self.labeled_archs)

        if self.sample_without_replacement == True:
            self.labeled_archs.pop(self.labeled_archs.index(op_indices))

        self.set_spec(op_indices)

    def sample_random_architecture(self, dataset_api=None, load_labeled=False, allow_invalid=True):
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        """

        if load_labeled == True:
            return self.sample_random_labeled_architecture()

        def is_valid_arch(op_indices):
            # Invalid archs are disconnected (there are no-ops along all paths from input to output).
            return not ((op_indices[0] == op_indices[1] == op_indices[2] == 1) or
                        (op_indices[2] == op_indices[4] == op_indices[5] == 1))

        while True:
            op_indices = np.random.randint(len(OP_NAMES), size=6)
            if allow_invalid or is_valid_arch(op_indices):
                break

        self.set_op_indices(op_indices)

    def mutate(self, parent, dataset_api=None):
        """
        This will mutate one op from the parent op indices, and then
        update the naslib object and op_indices
        """
        parent_op_indices = parent.get_op_indices()
        op_indices = list(parent_op_indices)

        edge = np.random.choice(len(parent_op_indices))
        available = [o for o in range(len(OP_NAMES)) if o != parent_op_indices[edge]]
        op_index = np.random.choice(available)
        op_indices[edge] = op_index
        # print('op_indices mu =', op_indices)
        self.set_op_indices(op_indices)

    def get_nbhd(self, dataset_api=None):
        # return all neighbors of the architecture
        self.get_op_indices()
        nbrs = []
        for edge in range(len(self.op_indices)):
            available = [o for o in range(len(OP_NAMES)) if o != self.op_indices[edge]]

            for op_index in available:
                nbr_op_indices = list(self.op_indices).copy()
                nbr_op_indices[edge] = op_index
                nbr = TransBench101SearchSpaceMicro()
                nbr.set_op_indices(nbr_op_indices)
                nbr_model = torch.nn.Module()
                nbr_model.arch = nbr
                nbrs.append(nbr_model)

        random.shuffle(nbrs)
        return nbrs

    def get_type(self):
        return 'transbench101_micro'

    def get_loss_fn(self):
        if self.dataset in ['class_object', 'class_scene']:
            loss_fn = SoftmaxCrossEntropyWithLogits()
        elif self.dataset in ['autoencoder', 'normal']:
            loss_fn = nn.L1Loss()
        elif self.dataset == 'room_layout':
            loss_fn = nn.MSELoss()
        else:
            loss_fn = F.cross_entropy

        return loss_fn

    def _forward_before_global_avg_pool(self, x):
        outputs = []

        def hook_fn(module, inputs, output_t):
            # print(f'Input tensor shape: {inputs[0].shape}')
            # print(f'Output tensor shape: {output_t.shape}')
            outputs.append(inputs[0])

        for m in self.modules():
            if isinstance(m, torch.nn.AdaptiveAvgPool2d):
                m.register_forward_hook(hook_fn)

        self.forward(x, None)

        assert len(outputs) == 1
        return outputs[0]

    def _forward_before_last_conv(self, x):
        outputs = []

        def hook_fn(module, inputs, output_t):
            # print(f'Input tensor shape: {inputs[0].shape}')
            # print(f'Output tensor shape: {output_t.shape}')
            outputs.append(inputs[0])

        model = self.edges[1, 2]['op'].model
        decoder = model.decoder

        if self.dataset == 'segmentsemantic':
            conv = decoder.model[-1]
        else:
            conv = decoder.conv14

        conv.register_forward_hook(hook_fn)

        self.forward(x, None)

        assert len(outputs) == 1
        return outputs[0]

    def forward_before_global_avg_pool(self, x):
        if (self.create_graph == True and self.dataset in ['ninapro', 'svhn', 'scifar100']) or \
                (self.dataset in ['class_scene', 'class_object', 'room_layout', 'jigsaw']):
            return self._forward_before_global_avg_pool(x)
        elif self.create_graph == False:
            return self._forward_before_last_conv(x)
        else:
            raise Exception(
                f"forward_before_global_avg_pool method not implemented for NASLib graph for dataset {self.dataset}")

    def encode(self, encoding_type="adjacency_one_hot"):
        return encode_tb101(self, encoding_type=encoding_type)

    def encode_spec(self, encoding_type='adjacency_one_hot'):
        if encoding_type == 'adjacency_one_hot':
            return encode_adjacency_one_hot_transbench_micro_op_indices(self)
        else:
            raise NotImplementedError(
                f'No implementation found for encoding search space TransBench101SearchSpaceMicro with {encoding_type}')


class TransBench101SearchSpaceMacro(Graph):
    """
    Implementation of the transbench 101 search space.
    It also has an interface to the tabular benchmark of transbench 101.
    """

    OPTIMIZER_SCOPE = [
        "stage_1",
        "stage_2",
        "stage_3",
    ]

    QUERYABLE = True

    def __init__(self, dataset='jigsaw', *arg, **kwargs):
        super().__init__()
        if dataset == "jigsaw":
            self.num_classes = 1000
        elif dataset == "class_object":
            self.num_classes = 100
        elif dataset == "class_scene":
            self.num_classes = 63
        else:
            self.num_classes = -1

        self.dataset = dataset
        self.op_indices = None

        self.max_epoch = 199
        self.space_name = 'transbench101'
        self.labeled_archs = None
        self.instantiate_model = True
        self.sample_without_replacement = False

        self.add_edge(1, 2)

    def query(self, metric=None, dataset=None, path=None, epoch=-1, full_lc=False, dataset_api=None):
        """
        Query results from TransNAS-Bench-101.
        """
        assert isinstance(metric, Metric)
        if metric == Metric.ALL:
            raise NotImplementedError()
        if dataset_api is None:
            raise ValueError('Must pass in dataset_api to query transbench101')

        arch_str = str(self)
        query_results = dataset_api['api']

        if metric == Metric.RAW:
            # return all data
            return query_results.get_arch_result(arch_str).query_all_results()[dataset]
        if metric in (Metric.FLOPS, Metric.PARAMETERS):
            return query_results.get_model_info(arch_str, dataset, METRIC_2_MODEL_INFO[metric])

        tnb_metric = TASK_2_METRIC[dataset][metric]
        if full_lc:
            all_epochs = query_results.get_single_metric(arch_str, dataset, tnb_metric, mode="list")
            if epoch != -1:
                return all_epochs[:epoch]
            return all_epochs
        else:
            return query_results.get_single_metric(arch_str, dataset, tnb_metric, mode=epoch)

    def get_op_indices(self):
        if self.op_indices is None:
            raise ValueError('op_indices not set')
        return self.op_indices

    def get_hash(self):
        return tuple(self.get_op_indices())

    def __hash__(self):
        return hash(self.get_hash())

    def __str__(self) -> str:
        return convert_op_indices_macro_to_str(self.get_op_indices())

    def set_from_string(self, arch_str: str) -> None:
        self.set_op_indices(convert_macro_str_op_indices(arch_str))

    def set_op_indices(self, op_indices):
        # This will update the edges in the naslib object to op_indices
        self.op_indices = op_indices

        if self.instantiate_model == True:
            model = convert_op_indices_macro_to_model(op_indices, self.dataset)
            self.edges[1, 2].set('op', model)

    def set_spec(self, op_indices, dataset_api=None):
        self.set_op_indices(op_indices)

    def sample_random_labeled_architecture(self):
        assert self.labeled_archs is not None, "Labeled archs not provided to sample from"

        op_indices = random.choice(self.labeled_archs)

        if self.sample_without_replacement == True:
            self.labeled_archs.pop(self.labeled_archs.index(op_indices))

        self.set_spec(op_indices)

    def sample_random_architecture(self, dataset_api=None, load_labeled=False):
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        """

        if load_labeled == True:
            return self.sample_random_labeled_architecture()

        r = random.randint(0, 2)
        p = random.randint(1, 4)
        q = random.randint(1, 3)
        u = [2 * int(i < p) for i in range(r + 4)]
        v = [int(i < q) for i in range(r + 4)]

        random.shuffle(u)
        random.shuffle(v)

        w = [1 + sum(x) for x in zip(u, v)]
        op_indices = np.array(w)

        while len(op_indices) < 6:
            op_indices = np.append(op_indices, 0)

        self.set_op_indices(op_indices)

    def mutate(self, parent, dataset_api=None):
        """
        This will mutate one op from the parent op indices, and then
        update the naslib object and op_indices
        """
        parent_op_indices = list(parent.get_op_indices())
        parent_op_ind = parent_op_indices[parent_op_indices != 0]

        def f(g):
            r = len(g)
            p = sum([int(i == 4 or i == 3) for i in g])
            q = sum([int(i == 4 or i == 2) for i in g])
            return r, p, q

        def g(r, p, q):
            u = [2 * int(i < p) for i in range(r)]
            v = [int(i < q) for i in range(r)]
            w = [1 + sum(x) for x in zip(u, v)]
            return np.random.permutation(w)

        a, b, c = f(parent_op_ind)

        a_available = [i for i in [4, 5, 6] if i != a]
        b_available = [i for i in range(1, 5) if i != b]
        c_available = [i for i in range(1, 4) if i != c]

        dic1 = {1: a, 2: b, 3: c}
        dic2 = {1: a_available, 2: b_available, 3: c_available}

        numb = random.randint(1, 3)

        dic1[numb] = random.choice(dic2[numb])

        op_indices = g(dic1[1], dic1[2], dic1[3])
        while len(op_indices) < 6:
            op_indices = np.append(op_indices, 0)

        self.set_op_indices(op_indices)

    def get_nbhd(self, dataset_api=None):
        # return all neighbors of the architecture
        self.get_op_indices()
        op_ind = list(self.op_indices[self.op_indices != 0])
        nbrs = []

        def f(g):
            r = len(g)
            p = sum([int(i == 4 or i == 3) for i in g])
            q = sum([int(i == 4 or i == 2) for i in g])
            return r, p, q

        def g(r, p, q):
            u = [2 * int(i < p) for i in range(r)]
            v = [int(i < q) for i in range(r)]
            w = [1 + sum(x) for x in zip(u, v)]
            return np.random.permutation(w)

        a, b, c = f(op_ind)

        a_available = [i for i in [4, 5, 6] if i != a]
        b_available = [i for i in range(1, 5) if i != b]
        c_available = [i for i in range(1, 4) if i != c]

        for r in a_available:
            for p in b_available:
                for q in c_available:
                    nbr_op_indices = g(r, p, q)
                    while len(nbr_op_indices) < 6:
                        nbr_op_indices = np.append(nbr_op_indices, 0)
                    nbr = TransBench101SearchSpaceMacro()
                    nbr.set_op_indices(nbr_op_indices)
                    nbr_model = torch.nn.Module()
                    nbr_model.arch = nbr
                    nbrs.append(nbr_model)

        random.shuffle(nbrs)
        return nbrs

    def get_type(self):
        return 'transbench101_macro'

    def get_loss_fn(self):
        if self.dataset in ['class_object', 'class_scene']:
            loss_fn = SoftmaxCrossEntropyWithLogits()
        elif self.dataset in ['autoencoder', 'normal']:
            loss_fn = nn.L1Loss()
        elif self.dataset == 'room_layout':
            loss_fn = nn.MSELoss()
        else:
            loss_fn = F.cross_entropy

        return loss_fn

    def _forward_before_global_avg_pool(self, x):
        outputs = []

        def hook_fn(module, inputs, output_t):
            # print(f'Input tensor shape: {inputs[0].shape}')
            # print(f'Output tensor shape: {output_t.shape}')
            outputs.append(inputs[0])

        for m in self.modules():
            if isinstance(m, torch.nn.AdaptiveAvgPool2d):
                m.register_forward_hook(hook_fn)

        self.forward(x, None)

        assert len(outputs) == 1
        return outputs[0]

    def _forward_before_last_conv(self, x):
        outputs = []

        def hook_fn(module, inputs, output_t):
            # print(f'Input tensor shape: {inputs[0].shape}')
            # print(f'Output tensor shape: {output_t.shape}')
            outputs.append(inputs[0])

        model = self.edges[1, 2]['op'].model
        decoder = model.decoder

        if self.dataset == 'segmentsemantic':
            conv = decoder.model[-1]
        else:
            conv = decoder.conv14

        conv.register_forward_hook(hook_fn)

        self.forward(x, None)

        assert len(outputs) == 1
        return outputs[0]

    def forward_before_global_avg_pool(self, x):

        if self.dataset in ['class_scene', 'class_object', 'room_layout', 'jigsaw']:
            return self._forward_before_global_avg_pool(x)
        else:
            return self._forward_before_last_conv(x)

    def encode(self, encoding_type=EncodingType.ADJACENCY_ONE_HOT):
        return encode_tb101(self, encoding_type=encoding_type)


def _set_op(edge, C_in, downsample):
    C_out = C_in
    stride = 1

    if downsample:
        if edge.head == 1:
            C_out = C_in * 2
            stride = 2
        else:
            C_in *= 2
            C_out = C_in
            stride = 1

    edge.data.set("op", [
        ops.Identity() if stride == 1 else FactorizedReduce(C_in, C_out, stride, affine=False),
        ops.Zero(stride=stride, C_in=C_in, C_out=C_out),
        ops.ReLUConvBN(C_in, C_out, kernel_size=3, stride=stride),
        ops.ReLUConvBN(C_in, C_out, kernel_size=1, stride=stride),
    ])


class TransBench101QuerySpace:
    """
    Implementation of the TransNAS-Bench-101 tabular benchmark.
    """

    def __init__(self):
        self.op_indices = None

    def get_op_indices(self):
        if self.op_indices is None:
            raise ValueError('op_indices not set')
        return self.op_indices

    def get_hash(self):
        return tuple(self.get_op_indices())

    def __hash__(self):
        return hash(self.get_hash())

    def __eq__(self, other):
        return np.all(self.op_indices == other.get_op_indices())

    def set_op_indices(self, op_indices):
        self.op_indices = np.array(op_indices, dtype=np.int8)

    def set_spec(self, op_indices):
        self.set_op_indices(op_indices)

    def query(self, metric=None, dataset=None, path=None, epoch=-1, full_lc=False, dataset_api=None):
        """
        Query results from TransNAS-Bench-101.
        """
        assert isinstance(metric, Metric)
        if metric == Metric.ALL:
            raise NotImplementedError()
        if dataset_api is None:
            raise ValueError('Must pass in dataset_api to query transbench101')

        arch_str = str(self)
        query_results = dataset_api['api']

        if metric == Metric.RAW:
            # return all data
            return query_results.get_arch_result(arch_str).query_all_results()[dataset]
        if metric in (Metric.FLOPS, Metric.PARAMETERS):
            return query_results.get_model_info(arch_str, dataset, METRIC_2_MODEL_INFO[metric])

        tnb_metric = TASK_2_METRIC[dataset][metric]
        convert = get_metric_conversion(tnb_metric)
        if full_lc:
            epochs = query_results.get_single_metric(arch_str, dataset, tnb_metric, mode="list")
            if epoch != -1:
                epochs = epochs[:epoch]
            return [convert(e) for e in epochs]
        else:
            return convert(query_results.get_single_metric(arch_str, dataset, tnb_metric, mode=epoch))


class TransBench101QuerySpaceMicro(TransBench101QuerySpace):
    """
    Implementation of the "micro" search space of the TransNAS-Bench-101 tabular benchmark.
    """

    def __init__(self):
        super().__init__()

    def __repr__(self) -> str:
        return convert_op_indices_micro_to_str(self.get_op_indices())

    def set_from_string(self, arch_str: str) -> None:
        self.set_op_indices(convert_micro_str_op_indices(arch_str))

    def sample_random_architecture(self, dataset_api: dict = None, load_labeled: bool = False,
                                   allow_invalid: bool = True) -> None:
        """
        This will sample a random architecture and update the op_indices edges accordingly.
        """

        def is_valid_arch(op_indices):
            # Invalid archs are disconnected (there are no-ops along all paths from input to output).
            return not ((op_indices[0] == op_indices[1] == op_indices[2] == 1) or
                        (op_indices[2] == op_indices[4] == op_indices[5] == 1))

        while True:
            op_indices = np.random.randint(len(OP_NAMES), size=6)
            if allow_invalid or is_valid_arch(op_indices):
                break

        self.set_op_indices(op_indices)

    def mutate(self, parent, dataset_api=None):
        """
        This will mutate one op from the parent op indices, and then update the op_indices.
        """
        parent_op_indices = parent.get_op_indices()
        op_indices = list(parent_op_indices)

        edge = np.random.choice(len(parent_op_indices))
        available = [o for o in range(len(OP_NAMES)) if o != parent_op_indices[edge]]
        op_index = np.random.choice(available)
        op_indices[edge] = op_index
        self.set_op_indices(op_indices)

    def get_neighbors(self, dataset_api=None):
        """
        return all neighbors of the architecture
        """
        nbrs = []
        for edge in range(len(self.op_indices)):
            available = [o for o in range(len(OP_NAMES)) if o != self.op_indices[edge]]
            for op_index in available:
                nbr_op_indices = list(self.op_indices).copy()
                nbr_op_indices[edge] = op_index
                nbr = TransBench101QuerySpaceMicro()
                nbr.set_op_indices(nbr_op_indices)
                nbrs.append(nbr)

        return nbrs


class TransBench101QuerySpaceMacro(TransBench101QuerySpace):
    """
    Implementation of the "macro" search space of the TransNAS-Bench-101 tabular benchmark.

    This space consists of 4-6 modules, where each module can be one of 4 types:
        1. Normal.
        2. Double the channels (out_channels = in_channels x 2).
        3. Downsample (stride = 2).
        4. Both.
    But with the following constraints:
        - Must double between 1 and 3 times.
        - Must downsample between 1 and 4 times.

    For example:
        - We cannot have "11311" because that doubles but does not downsample.
        - We cannot have "4444" because that doubles too many times.
        - On the other hand, "144431" and "444311" are valid--3 doubles and 4 downsamples.
    """

    def __init__(self):
        super().__init__()

    def __repr__(self) -> str:
        return convert_op_indices_macro_to_str(self.get_op_indices())

    def set_from_string(self, arch_str: str) -> None:
        self.set_op_indices(convert_macro_str_op_indices(arch_str))

    def sample_random_architecture(self, dataset_api: dict = None, load_labeled: bool = False) -> None:
        """
        This will sample a random architecture and update the op_indices edges accordingly.
        """
        all_macro_strs = list(dataset_api["api"].data["macro"].keys())
        self.set_from_string(random.choice(all_macro_strs))

    def mutate(self, parent, dataset_api=None):
        """
        This will mutate one op from the parent op indices, and then update the op_indices.

        WARNING: This is not a uniform distribution over all possible neighbors. Adding/deleting nodes may be favored.
        """
        parent_op_ind = parent.get_op_indices()
        parent_op_ind = list(parent_op_ind[parent_op_ind != 0])

        mutation_options = ["change"]
        if len(parent_op_ind) < 6:
            mutation_options.append("add")
        if len(parent_op_ind) > 4:
            mutation_options.append("remove")

        new_op = []
        possible_ops = [1, 2, 3, 4]  # module types
        while not is_valid_macro_code(new_op):
            mutation_type = random.choice(mutation_options)
            new_op = parent_op_ind.copy()
            if mutation_type == "change":
                # Change one of the existing ops.
                i = random.randint(0, len(parent_op_ind) - 1)
                new_op[i] = random.choice([o for o in possible_ops if o != parent_op_ind[i]])
            elif mutation_type == "add":
                # Add a module.
                i = random.randint(0, len(parent_op_ind))
                new_op.insert(i, random.choice(possible_ops))
            elif mutation_type == "remove":
                # Remove a module.
                i = random.randint(0, len(parent_op_ind) - 1)
                del new_op[i]

        self.set_op_indices(new_op)

    def get_neighbors(self, dataset_api=None):
        """
        return all neighbors of the architecture
        """
        op_ind = list(self.op_indices[self.op_indices != 0])
        possible_ops = [1, 2, 3, 4]  # module types
        nbrs = []

        def add_if_valid(new_op, nbrs):
            if is_valid_macro_code(new_op):
                nbr = TransBench101QuerySpaceMacro()
                nbr.set_op_indices(new_op)
                nbrs.append(nbr)

        # Start with all possible changes of the existing ops.
        for i in range(len(op_ind)):
            for op in filter(lambda o: o != op_ind[i], possible_ops):
                new_op = op_ind.copy()
                new_op[i] = op
                add_if_valid(new_op, nbrs)

        # Now consider adding nodes.
        if len(op_ind) < 6:
            # For each possible location...
            for i in range(len(op_ind) + 1):
                # And each possible op...
                for op in possible_ops:
                    # Try adding the op at that location.
                    new_op = op_ind.copy()
                    new_op.insert(i, op)
                    add_if_valid(new_op, nbrs)

        # Now consider removing nodes.
        if len(op_ind) > 4:
            # For each possible location...
            for i in range(len(op_ind)):
                # Try deleting the op at that location.
                new_op = op_ind.copy()
                del new_op[i]
                add_if_valid(new_op, nbrs)

        return nbrs
