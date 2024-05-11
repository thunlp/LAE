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
"""Pet model for llm model."""
from typing import Union

from mindspore._checkparam import args_type_check

from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.pet.constants import PetType
from mindformers.pet.models.lora import LoraModel
from mindformers.pet.pet_config import LoraConfig, PetConfig
from mindformers.pet.tuners.pet_adapter import PetAdapter
from mindformers.tools import logger


# Mapping of pet models.
PET_TYPE_TO_MODEL_MAPPING = {
    PetType.LORA.value: LoraModel,
}


# Mapping of pet configs.
PET_TYPE_TO_CONFIG_MAPPING = {
    PetType.LORA.value: LoraConfig,
}


class PetModel(PreTrainedModel):
    """
    PetModel define parameter efficient tuning model for LLM model.

    Args:
        config(PetConfig): pet config,define parameters efficient tuning algorithm.
        base_model(PreTrainedModel): pretrained model for tuning.
    """
    @args_type_check(config=(dict, PetConfig))
    def __init__(self, config: Union[dict, PetConfig], base_model: PreTrainedModel):
        super().__init__(base_model.config, auto_prefix=False)
        if not isinstance(config, PetConfig):
            pet_type = config.pop("pet_type")
            pet_config = PET_TYPE_TO_CONFIG_MAPPING[pet_type](**config)
        else:
            pet_type = config.pet_type
            pet_config = config
        self.config.pet_config = pet_config
        # pylint: disable=W0212
        self._support_list = base_model._support_list
        self.pet_model = PET_TYPE_TO_MODEL_MAPPING[pet_type](pet_config, base_model)
        self.load_checkpoint(self.config)
        PetAdapter.freeze_pretrained_model(self.pet_model, pet_type, pet_config.freeze_include,
                                           pet_config.freeze_exclude)

    def update_model_kwargs_before_generate(self, input_ids, model_kwargs: dict):
        return self.pet_model.update_model_kwargs_before_generate(input_ids, model_kwargs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.pet_model.prepare_inputs_for_generation(input_ids, **kwargs)

    def prepare_inputs_for_export(self, full_model=True):
        return self.pet_model.prepare_inputs_for_export(full_model)

    def slice_incremental_inputs(self, model_inputs: dict, current_index):
        return self.pet_model.slice_incremental_inputs(model_inputs, current_index)

    def construct(self, input_ids, labels=None, position_ids=None, attention_mask=None, input_position=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None):
        return self.pet_model(input_ids, labels, input_position, position_ids,
                              attention_mask, input_embeds, init_reset, batch_valid_length)

@args_type_check(config=(dict, PetConfig))
def get_pet_model(base_model: PreTrainedModel, config: Union[dict, PetConfig]):
    """
    Get model with pet model.

    Args:
        base_model (PreTrainedModel): The pretrained model for tuning.
        config (PetConfig): The config of parameter efficient tuning algrithm.

    Return:
        model(PreTrainedModel)
    """
    pet_type = config.get("pet_type")
    if not PET_TYPE_TO_MODEL_MAPPING.get(pet_type):
        logger.warning("%s doesn't have pet model currently.", pet_type)
        return base_model

    # return pet model.
    return PetModel(config=config, base_model=base_model)

def is_supported_pet_type(pet_type: str):
    """
    Return `pet_type` is supported or not.
    """

    return pet_type in PET_TYPE_TO_MODEL_MAPPING
