# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Build context."""

import os
from typing import Union
import psutil

import mindspore.dataset as ds
from mindspore import context
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs

from mindformers.trainer.config_args import ContextConfig, ParallelContextConfig
from mindformers.trainer.training_args import TrainingArguments
from mindformers.tools import PARALLEL_MODE, MODE, get_output_subpath, check_in_modelarts
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerConfig
from mindformers.tools.utils import check_in_dynamic_cluster, set_strategy_save_path

CONTEXT_CONFIG = {
    'mode': 'GRAPH_MODE', 'device_target': 'Ascend', 'device_id': 0, 'save_graphs': False}
PARALLEL_CONFIG = {'parallel_mode': 'DATA_PARALLEL', 'gradients_mean': True}


def build_context(config: Union[dict, MindFormerConfig, TrainingArguments]):
    """Build context."""
    if isinstance(config, TrainingArguments):
        config = config.convert_args_to_mindformers_config()
    if isinstance(config, dict) and not isinstance(config, MindFormerConfig):
        config = MindFormerConfig(**config)
    if config.use_parallel and config.parallel_config.pipeline_stage > 1:
        config.parallel.pipeline_stages = config.parallel_config.pipeline_stage
    local_rank, device_num = init_context(use_parallel=config.use_parallel,
                                          context_config=config.context, parallel_config=config.parallel)

    if context.get_auto_parallel_context("parallel_mode") == "auto_parallel":
        set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)
    else:
        set_algo_parameters(elementwise_op_strategy_follow=True, fully_use_devices=True)
    _set_multi_subgraphs()

    config.device_num = device_num
    config.local_rank = local_rank
    use_cpu_affinity = os.environ.get("CPU_AFFINITY")
    if use_cpu_affinity and (use_cpu_affinity == '1' or use_cpu_affinity.lower() == 'true'):
        ds.config.set_numa_enable(True)
        cpu_affinity(local_rank, device_num)
        logger.info(f"cpu_affinity, rank_id: {local_rank}, device_num: {device_num}")

    if config.parallel.get("strategy_ckpt_load_file"):
        context.set_auto_parallel_context(strategy_ckpt_load_file=config.parallel.strategy_ckpt_load_file)


def cpu_affinity(rank_id, rank_size):
    """cpu affinity"""
    count = psutil.cpu_count()
    p = psutil.Process()
    used_cpus_num = count // rank_size
    used_cpus = [i for i in range(rank_id * used_cpus_num, (rank_id + 1) * used_cpus_num)]
    p.cpu_affinity(used_cpus)


def init_context(use_parallel=False, context_config=None, parallel_config=None):
    """Context initialization for MindSpore.
    Args:
        use_parallel (Optional[Union[bool]]):
            Whether to use distributed training. Default: False.
        context_config (Optional[Union[dict, ContextConfig]]):
            Context Config For Running Environment. Default: None.
        parallel_config (Optional[Union[dict, ParallelContextConfig]]):
            Parallel Config For Running Environment. Default: None.

    Returns: rank_id, device_num.
    """

    if isinstance(context_config, ContextConfig):
        context_config = context_config.__dict__
    if isinstance(parallel_config, ParallelContextConfig):
        parallel_config = parallel_config.__dict__

    if context_config is None:
        context_config = CONTEXT_CONFIG
    if parallel_config is None:
        parallel_config = PARALLEL_CONFIG

    _set_check_context_config(context_config)
    _set_check_parallel_config(parallel_config)

    device_num = 1
    rank_id = 0
    context_config['mode'] = MODE.get(context_config.get('mode'))

    context.set_context(max_device_memory=context_config.get('max_device_memory'),
                        mode=context_config.get('mode'))
    del context_config['mode']
    del context_config['max_device_memory']
    if use_parallel:
        device_id = int(os.getenv('DEVICE_ID', '0'))  # 0 ~ 7
        context_config['device_id'] = device_id
        if check_in_dynamic_cluster():
            # for dynamic cluster, we should not set device id in context.
            context_config.pop('device_id', None)
        parallel_config['parallel_mode'] = PARALLEL_MODE.get(parallel_config.get('parallel_mode'))
        context.set_context(**context_config)
        try:
            init()
        except:
            raise RuntimeError("Notice: if you are trying to run with a single device, please set "
                               "use_parallel=False. If not, please check the error message above.")
        rank_id = get_rank()  # local_rank
        device_num = get_group_size()  # world_size
        parallel_config.setdefault('device_num', device_num)
        context.reset_auto_parallel_context()
        set_strategy_save_path(parallel_config)
        context.set_auto_parallel_context(**parallel_config)
    else:
        context.set_context(**context_config)
    return rank_id, device_num


def _set_check_context_config(config):
    """Set context config."""
    mode = config.get('mode')
    if mode is None:
        config.setdefault('mode', 0)
    if mode not in MODE.keys():
        raise IndexError('Running mode should be in {}, but get {}'.format(MODE.keys(), mode))

    device = config.get('device_id')
    if device is None:
        config.setdefault('device_id', 0)

    max_device_memory = config.get('max_device_memory')
    if max_device_memory is None:
        config.setdefault('max_device_memory', '1024GB')

    save_graph = config.get('save_graphs')
    if save_graph:
        save_graphs_path = config.get('save_graphs_path') if not check_in_modelarts() else None
        if save_graphs_path is None:
            save_graphs_path = get_output_subpath("debug/graphs_info", append_rank=False)
        config['save_graphs_path'] = save_graphs_path
    enable_dump = config.get('enable_dump')
    if enable_dump:
        save_dump_path = config.get('save_dump_path') if not check_in_modelarts() else None
        if save_dump_path is None:
            save_dump_path = get_output_subpath("debug/dump_info", append_rank=False)
        config['save_dump_path'] = save_dump_path


def _set_check_parallel_config(config):
    """Set parallel config."""
    parallel_mode = config.get('parallel_mode')
    if parallel_mode is None:
        config.setdefault('parallel_mode', 0)

    if PARALLEL_MODE.get(config.get('parallel_mode')) not in \
            [context.ParallelMode.SEMI_AUTO_PARALLEL, context.ParallelMode.AUTO_PARALLEL] and config.get('full_batch'):
        logger.info("full_batch will be forced to False when the parallel mode is stand_alone or data_parallel")
        config.setdefault('full_batch', False)

    if parallel_mode not in PARALLEL_MODE.keys():
        raise IndexError(
            'Running parallel mode should be in {}, but get {}'.format(PARALLEL_MODE.keys(), parallel_mode))
