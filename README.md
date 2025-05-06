# Quick Start

LLM fine-tuning on LoRA adapter


## 1. Model Training


`python -m main --mode train -m config/model/llama_lora_1b.yaml -d config/data/cnn_dataset.yaml`

Code above will fine-tune Llama-3.2-1B-Instruct model with LoRA adapter on the CNN Daily Mail dataset.

You can modify `base_model_name` in [Model Config](./config/model/llama_lora_1b.yaml) to any existing model name in the huggingface hub, to train on a different base model.

Also, you can set `dataset_name` field in [Data config](./config/data/cnn_dataset.yaml) to train on a different dataset.

If you want to train on a custom dataset, change `data_type` to `csv`, and set `data_path` to data path. Data should be formatted in CSV. Implement `_load_dataset` in [dataset.py](./src/utils/dataset.py) in order to support other formats.



## 2. Model Evaluation

`python -m main --mode evaluate -m config/model/llama_lora_1b.yaml -d config/data/cnn_dataset.yaml -g config/generation/beam_search.yaml`

Once training is done, you can evaluate model performance. Code above will evaluate performance on already trained example model [Example Model Checkpoint](./saved_models/llama_lora_1b/cnn_dailymail).

## 3. Summary Generation

`python -m main --mode generate -m config/model/llama_lora_1b.yaml -d config/data/custom_txt_dataset.yaml -g config/generation/beam_search.yaml --output-path data/csv_generated_result --pretrained-model saved_models/llama_lora_1b/cnn_dailymail/`

To 