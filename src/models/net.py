import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def get_base_model(config, mode='train'):
    model_name = config.get('base_model_name')
    model_args = dict()
    model_args['torch_dtype'] = config.get('dtype', torch.float16)
    if config.get('load_in_8bit', False) and mode == 'train':
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model_args['quantization_config'] = quantization_config

    return AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", **model_args)


def wrap_lora(model, config):
    lora_config = LoraConfig(
        r=config.get('lora_rank', 8),
        lora_alpha=config.get('lora_alpha', 16),
        target_modules=config.get('lora_targets', ["q_proj", "v_proj"]),
        lora_dropout=config.get('lora_dropout', 0),
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    return model


def get_model(config, mode='train'):
    model = get_base_model(config, mode)
    if config.get('is_lora', False) and mode == 'train':
        # don't wrap lora adapter when model is eval mode
        model = wrap_lora(model, config)
    return model
