import pandas as pd
from transformers import LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset
import torch
from src.models.net import *
from src.utils.dataset import get_dataset, BaseDataSet
import argparse
from src.utils.file_handler import read_yaml, maybe_load_lora_adapter, save_generation_result
from src.utils.evaluate import generate_summary, evaluate_metrics, get_sources

import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'generate', 'evaluate'])
    parser.add_argument('--model-config', '-m', type=str)
    parser.add_argument('--data-config', '-d', type=str)
    parser.add_argument('--generate-config', '-g', type=str)
    parser.add_argument('--output-path', type=str, help='save path for trained models, or generation results.'
                                                        ' If not set, save_name in the config files will be used')
    parser.add_argument('--pretrained-model', type=str, help='path for pretrained model.'
                                                             ' If specified, it overrides model path defined in model-config')

    return parser.parse_args()


def get_default_model_path(model_config, dataset):
    return os.path.join('saved_models', model_config.get('save_name'), dataset.config.get('save_name'))


def prepare(args, mode='train'):
    model_config = read_yaml(args.model_config)
    data_config = read_yaml(args.data_config)

    # model_config = read_yaml('config/model/llama_lora_1b.yaml')
    # data_config = read_yaml('config/data/cnn_dataset.yaml')

    model = get_model(model_config, mode)
    tokenizer = AutoTokenizer.from_pretrained(model_config.get('base_model_name'))
    dataset = get_dataset(tokenizer, data_config)
    return model_config, data_config, model, dataset




def train(args):
    def get_trainer(model, dataset: BaseDataSet, model_config):
        if args.output_path:
            output_path = args.output_path
        else:
            output_path = get_default_model_path(model_config, dataset)

        training_args = TrainingArguments(
            output_dir=output_path,
            per_device_train_batch_size=model_config.get('batch_size', 1),
            per_device_eval_batch_size=model_config.get('batch_size', 1),
            gradient_accumulation_steps=model_config.get('gradient_accumulation_step', 1),
            learning_rate=float(model_config.get('learning_rate', 1e-4)),
            num_train_epochs=model_config.get('epoch', 1),
            save_steps=1000,
            logging_steps=100,
            save_total_limit=1,
            fp16=True,  # hard-coded. Should be modified.
            optim=model_config.get('optimizer', 'paged_adamw_8bit'),
            eval_strategy='epoch'
        )

        return Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset.train_set.map(dataset.training_data_process,
                                                remove_columns=dataset.train_set.column_names),
            eval_dataset=dataset.valid_set.map(dataset.training_data_process,
                                               remove_columns=dataset.train_set.column_names),
            data_collator=dataset.get_collator(),
        )

    model_config, data_config, model, dataset = prepare(args)

    trainer = get_trainer(model, dataset, model_config)
    trainer.train()


# generate summaries and dump generated result.
def generate(args):
    def generation_metadata_for_logging():
        meta = {}
        meta['base_model'] = model_config.get('base_model_name')
        if is_loaded:
            meta['loaded_checkpoint'] = checkpoint_path
        meta.update(generate_config)
        return meta


    model_config, data_config, model, dataset = prepare(args, 'generation')
    generate_config = read_yaml(args.generate_config)
    checkpoint_path = args.pretrained_model if args.pretrained_model else get_default_model_path(model_config, dataset)

    model, is_loaded = maybe_load_lora_adapter(model, checkpoint_path)

    summaries = generate_summary(model, dataset, generate_config)
    sources = get_sources(dataset, generate_config)
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = os.path.join(model_config.get('save_name'),
                                   data_config.get('save_name'),
                                   generate_config.get('save_name'))

    save_generation_result(sources, summaries, generation_metadata_for_logging(), output_path)



# evaluate model performance. Default = rouge and bert-score
def evaluate(args):
    model_config, data_config, model, dataset = prepare(args, 'evaluation')
    # load lora adapter
    model, is_loaded = maybe_load_lora_adapter(model, get_default_model_path(model_config, dataset))
    generate_config = read_yaml(args.generate_config)
    metrics = evaluate_metrics(model, dataset, generate_config)
    print(metrics)


if __name__ == '__main__':
    args = get_args()
    if args.mode == 'train':
        train(args)

    elif args.mode == 'evaluate':
        evaluate(args)

    elif args.mode == 'generate':
        generate(args)
