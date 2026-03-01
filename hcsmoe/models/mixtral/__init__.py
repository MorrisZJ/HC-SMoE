from .configuration_mixtral import MixtralConfig
from .modeling_mixtral import (
    MixtralForCausalLM,
    MixtralForSequenceClassification,
    MixtralModel,
    MixtralPreTrainedModel,
    MixtralSparseMoeBlock,
    MyMixtralForCausalLM,
)

# ---------------------------------------------------------------------------
# Register the HC-SMoE Mixtral variant with HuggingFace Auto classes so that
#   AutoConfig.from_pretrained(path)  and
#   AutoModelForCausalLM.from_pretrained(path)
# both work out-of-the-box after `import hcsmoe.models.mixtral`.
# ---------------------------------------------------------------------------
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("hcsmoe_mixtral", MixtralConfig, exist_ok=True)
AutoModelForCausalLM.register(MixtralConfig, MixtralForCausalLM, exist_ok=True)
