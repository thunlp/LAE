# Copyright 2023 Huawei Technologies Co., Ltd
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
"""LLaMA models' APIs."""
import copy

import mindspore.common.dtype as mstype

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import Tensor, nn
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.modules.layers import Linear
from mindformers.modules.transformer import LowerTriangularMaskWithDynamic
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister

from .llama_config import LlamaConfig
from .llama_layer import LlamaEmbedding, LlamaRMSNorm, FreqsMgr
from .llama_transformer import LLamaDecodeLayer
from .llama_interleave import LLamaDecodeLayerInterleave
from ..utils import cell_reuse
from ...tools.logger import logger

__all__ = ['LlamaModel', 'LlamaForCausalLM']


class LlamaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LlamaConfig
    base_model_prefix = "llama"


def layer_compute_dtype(layer, layer_id, offset, parallel_config, n_layers, select_recompute=False):
    r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.

        Args:
            layer(Cell) - Represents the transformer block
            parallel_config(dict) - Parallel Config
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(Union[int, List[int]]) - Means the layer_index needs a offset, if there are other modules in the net.
            n_layers(int) - The total layers used for the model.
    """
    pp = parallel_config.pipeline_stage
    if isinstance(offset, (list, tuple)):
        if len(offset) != pp:
            raise ValueError(f"The length of `offset` {len(offset)} do not match `pipeline stage` {pp}.")
        stage_layers_list = np.array([n_layers // pp] * pp) + np.array(offset)
        layer_list = np.array([np.sum(stage_layers_list[:i + 1]) for i in range(len(stage_layers_list))])
        pp_id = int(np.sum(layer_list < layer_id + 1))
    elif isinstance(offset, int):
        offset_layer = offset
        pp_dis = max(int((n_layers + 1) / pp), 1)
        pp_id = min((layer_id + offset_layer) // pp_dis, pp - 1)
    else:
        raise TypeError(f"`offset` must be `int` or list/tuple of `int`, but got {type(offset)}.")

    layer.pipeline_stage = pp_id

    # Used for optimizer's fusion tag
    dis = max(int((n_layers + 1) / parallel_config.gradient_aggregation_group), 1)
    if pp > 1:
        layer.set_comm_fusion(2)
    else:
        layer.set_comm_fusion(int((layer_id + offset_layer) / dis) + 1)
    if isinstance(parallel_config.recompute, bool):
        if parallel_config.recompute and not select_recompute:
            layer.recompute()
    else:
        if parallel_config.recompute.recompute and not select_recompute:
            layer.recompute(
                recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)


class LlamaModel(LlamaPreTrainedModel):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    Args:
        config(LlamaConfig): the config of network

    Returns:
            output: Tensor, the output of llama decoderlayer

    Examples:
        >>> from mindformers import LlamaModel
        >>> network = LlamaModel.from_pretrained('llama_7b')
        >>> type(network)
        <class 'mindformers.models.llama.llama.LlamaModel'>
    """
    _support_list = MindFormerBook.get_model_support_list()['llama']

    def __init__(self,
                 config: LlamaConfig = None):
        super().__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        if config.batch_size or config.use_past:
            Validator.check_positive_int(config.batch_size)
        self.dtype = config.compute_dtype
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.n_head = config.num_heads
        self.head_dim = self.hidden_size // self.n_head
        self.pad_token_id = config.pad_token_id
        self.is_first_iteration = True
        self.seq_length = config.seq_length
        self.use_past = config.use_past
        self.is_dynamic = config.is_dynamic
        self.use_flash_attention = config.use_flash_attention
        self.shape = P.Shape()
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.cast = P.Cast()
        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()
        self.gather = P.Gather()
        self.slice = P.StridedSlice()

        self.freqs_mgr = FreqsMgr(head_dim=self.head_dim,
                                  seq_length=config.seq_length,
                                  max_position_embedding=config.max_position_embedding,
                                  rotary_dtype=config.rotary_dtype,
                                  theta=config.theta,
                                  scaling_factor=config.scaling_factor,
                                  extend_method=config.extend_method,
                                  is_dynamic=config.is_dynamic)
        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=config.seq_length,
                                                          compute_type=config.compute_dtype,
                                                          is_dynamic=config.is_dynamic,
                                                          pad_token_id=config.pad_token_id,
                                                          use_flash_attention=config.use_flash_attention)
        self.tok_embeddings = LlamaEmbedding(vocab_table_size=config.vocab_size,
                                             embedding_size=config.hidden_size,
                                             param_init_type=config.param_init_type)
        self.layers = nn.CellList()
        for layer_id in range(config.num_layers):
            if config.fine_grain_interleave > 1 and config.parallel_config.model_parallel > 1:
                layer = LLamaDecodeLayerInterleave(config.batch_size,
                                                   config.seq_length,
                                                   layer_id,
                                                   dim=config.hidden_size,
                                                   n_heads=config.num_heads,
                                                   num_layers=config.num_layers,
                                                   multiple_of=config.multiple_of,
                                                   n_kv_heads=config.n_kv_heads,
                                                   intermediate_size=config.intermediate_size,
                                                   ffn_dim_multiplier=config.ffn_dim_multiplier,
                                                   norm_eps=config.rms_norm_eps,
                                                   qkv_has_bias=config.qkv_has_bias,
                                                   qkv_concat=config.qkv_concat,
                                                   compute_dtype=config.compute_dtype,
                                                   layernorm_compute_dtype=config.layernorm_compute_type,
                                                   softmax_compute_dtype=config.softmax_compute_type,
                                                   rotary_dtype=config.rotary_dtype,
                                                   param_init_type=config.param_init_type,
                                                   use_flash_attention=config.use_flash_attention,
                                                   fine_grain_interleave=config.fine_grain_interleave,
                                                   parallel_config=config.parallel_config)
            else:
                layer = LLamaDecodeLayer(config.seq_length,
                                         layer_id,
                                         dim=config.hidden_size,
                                         n_heads=config.num_heads,
                                         n_kv_heads=config.n_kv_heads,
                                         intermediate_size=config.intermediate_size,
                                         multiple_of=config.multiple_of,
                                         ffn_dim_multiplier=config.ffn_dim_multiplier,
                                         norm_eps=config.rms_norm_eps,
                                         qkv_has_bias=config.qkv_has_bias,
                                         qkv_concat=config.qkv_concat,
                                         compute_dtype=config.compute_dtype,
                                         layernorm_compute_dtype=config.layernorm_compute_type,
                                         softmax_compute_dtype=config.softmax_compute_type,
                                         rotary_dtype=config.rotary_dtype,
                                         param_init_type=config.param_init_type,
                                         use_past=config.use_past,
                                         use_flash_attention=config.use_flash_attention,
                                         block_size=config.block_size,
                                         num_blocks=config.num_blocks,
                                         is_dynamic=config.is_dynamic,
                                         use_rope_slice=config.use_rope_slice,
                                         moe_config=config.moe_config,
                                         parallel_config=config.parallel_config)
            layer_compute_dtype(layer, layer_id, config.offset, config.parallel_config,
                                config.num_layers, select_recompute=config.parallel_config.recompute.select_recompute)
            self.layers.append(layer)
        self.norm_out = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps,
                                     compute_type=config.layernorm_compute_type, is_dynamic=config.is_dynamic)
        dp = config.parallel_config.data_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.tok_embeddings.pipeline_stage = 0
            if config.parallel_config.pipeline_stage > 1:
                self.norm_out.pipeline_stage = config.parallel_config.pipeline_stage - 1
                self.tok_embeddings.set_comm_fusion(2)
                self.norm_out.set_comm_fusion(2)
            else:
                self.tok_embeddings.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
                self.norm_out.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

            self.tok_embeddings.shard(config.parallel_config)
            self.casual_mask.shard(config.parallel_config)
            if config.fine_grain_interleave > 1:
                self.norm_out.shard((dp, 1))
            else:
                self.norm_out.shard((dp, 1, 1))

    # pylint: disable=W0613
    def construct(self, tokens: Tensor, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None):
        """
        Forward of llama model.

        Args:
            tokens: the tokenized inputs with datatype int32
            input_position(Tensor): current position, used by model.predict.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
                past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.

        Returns:
            output: Tensor, the output of llama decoderlayer
        """
        # preprocess
        bs, seq_len = self.shape(tokens)
        mask = None
        if self.use_past:
            freqs_cis = (self.freqs_mgr.freqs_cos, self.freqs_mgr.freqs_sin, self.freqs_mgr.swap_mask)
        else:
            freqs_cis = self.freqs_mgr()
            mask = self.casual_mask(tokens)  # mask: [bs, seq, seq]

        # tokens: [bs, seq/1]
        h = self.tok_embeddings(tokens)
        h = self.reshape(h, (bs, seq_len, self.hidden_size))
        # h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            h = self.layers[i](h, freqs_cis, mask, batch_valid_length=batch_valid_length, block_tables=block_tables,
                               slot_mapping=slot_mapping)
        output = self.norm_out(h)
        return output


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class LlamaForCausalLM(LlamaPreTrainedModel):
    r"""
        Provide llama training loss or logits through network.

        Args:
            config (LlamaConfig): The config of llama model.

        Returns:
            output: Tensor, the output of llama decoderlayer

        Examples:
            >>> from mindformers.models.llama import LlamaConfig, LlamaForCausalLM
            >>> config = LlamaConfig(batch_size=2)
            >>> network = LlamaForCausalLM(config=config)
            >>> type(network)
            <class 'mindformers.models.llama.llama.LlamaForCausalLM'>
            >>> from mindformers import LlamaForCausalLM
            >>> network = LlamaForCausalLM.from_pretrained('llama_7b')
            >>> type(network)
            <class 'mindformers.models.llama.llama.LlamaForCausalLM'>
        """
    _support_list = MindFormerBook.get_model_support_list()['llama']

    @cell_reuse
    def __init__(self, config: LlamaConfig = None):
        super(LlamaForCausalLM, self).__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.config = config
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        if config.is_dynamic:
            self.reshape.add_prim_attr("skip_redistribution", True)
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.mul = P.Mul()
        self.add = P.Add()
        self.ones = P.Ones()
        self.gather = P.Gather(1)
        self.sub_batch_valid_len = P.Sub()
        self.model = LlamaModel(config=config)
        self.lm_head = Linear(in_channels=config.hidden_size,
                              out_channels=config.vocab_size,
                              has_bias=False,
                              compute_dtype=config.compute_dtype,
                              param_init_type=config.param_init_type,
                              skip_redistribution=config.is_dynamic,
                              weight_init="normal")  # meta default: xavier_normal

        mp = config.parallel_config.model_parallel
        vocab_size = config.vocab_size
        loss_parallel_config = copy.deepcopy(config.parallel_config)
        if vocab_size % mp != 0:
            logger.warning("The vocab size of Loss is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of Loss will be changed: mp = 1")
            loss_parallel_config.model_parallel = 1
        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config)
        self.seq_length = config.seq_length

        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.slice.shard(((dp, 1),))
            self.not_equal.shard(((dp, 1), ()))
            self.mul.shard(((dp, 1), (dp, 1)))
            self.add.shard(((dp, 1), ()))
            self.gather.shard(((dp, 1, 1), (dp,)))
            self.sub_batch_valid_len.shard(((1,), ()))
            if config.parallel_config.vocab_emb_dp or (vocab_size % mp != 0):
                self.lm_head.shard(strategy_matmul=((dp, 1), (1, 1)))
            else:
                self.lm_head.shard(strategy_matmul=((dp, 1), (mp, 1)))
            if config.parallel_config.pipeline_stage > 1:
                self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1

        self.load_checkpoint(config)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": Tensor(input_ids, mstype.int32)
        }

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None):
        r"""
        LlamaForCausalLM forward.

        Args:
            input_ids(Tensor): the tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            labels(Tensor): the tokenized labels with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            input_position(Tensor): current position, used by model.predict.
            position_ids(Tensor): Reserved param, not used.
            attention_mask(Tensor): Reserved param, not used.
            input_embeds(Tensor): Reserved param, not used.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
                past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.

        Returns:
            Tensor: The loss or (logits, tokens, input_mask) of the network.
        """
        bsz, seqlen = self.shape(input_ids)
        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mstype.int32)
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bsz, seqlen - 1), (1, 1))
        else:
            tokens = input_ids
        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))
        if not self.is_first_iteration:
            batch_valid_length = self.sub_batch_valid_len(batch_valid_length, 1)
        output = self.model(tokens, batch_valid_length, batch_index, zactivate_len, block_tables, slot_mapping)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        logits = self.lm_head(output)

        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            if labels.ndim > 1:
                if self.training:
                    labels = self.slice(labels, (0, 1), (bsz, seqlen), (1, 1))
                label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), mstype.float32)
                input_mask = self.mul(input_mask, label_mask)

        if not self.training:
            if not pre_gather:
                logits = self.reshape(logits, (bsz, seqlen, -1))
            logits = self.cast(logits, mstype.float32)
            # makes cast effective to avoid allgather issue in Mindspore1.10
            input_mask = self.add(input_mask, 1)
            return logits, tokens, input_mask

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        logits = self.cast(logits, mstype.float32)
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss
