# Quick Start

## 1. Model Training


`python -m main --mode train -m config/model/llama_lora_1b.yaml -d config/data/cnn_dataset.yaml`

Code above will fine-tune Llama-3.2-1B-Instruct model with LORA adapter on the CNN Daily Mail dataset.

Base model and dataset can be adjusted by modifying corresponding config files.

For example, you can modify `base_model_name` in [Model Config](./config/model/llama_lora_1b.yaml) to any existing model name in the huggingface hub.

Also, `dataset_name` field in [Data config](./config/data/cnn_dataset.yaml) can be changed to train on a different dataset.



## 2. Model Evaluation

`python -m main --mode evaluate -m config/model/llama_lora_1b.yaml -d config/data/cnn_dataset.yaml -g config/generation/beam_search.yaml`

Once training is done, you can evaluate model performance. Code above will evaluate performance on already trained example model [Example Model Checkpoint](./saved_models/model/llama_lora_1b/cnn_dailymail).

## 3. Summary Generation

`python -m main --mode generate -m config/model/llama_lora_1b.yaml -d config/data/custom_txt_dataset.yaml -g config/generation/beam_search.yaml --output-path data/csv_generated_result --pretrained-model saved_models/llama_lora_1b/cnn_dailymail/`
