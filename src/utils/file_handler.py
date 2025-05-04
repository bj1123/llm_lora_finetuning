from pathlib import Path
import yaml
from peft import PeftModel
import pandas as pd
import os
import json


def read_yaml(path):
    with open(path) as f:
        res = yaml.safe_load(f)
    return res


def save_generation_result(sources, generated_summaries, meta_data, path):
    os.makedirs(path, exist_ok=True)
    df = pd.DataFrame(zip(sources, generated_summaries), columns=['Source', 'Summary'])
    df.to_csv(os.path.join(path, 'generated_summaries.csv'))
    with open(os.path.join(path, 'config.txt'), "w") as f:
        json.dump(meta_data, f)


def maybe_load_lora_adapter(model, path):
    def get_latest_checkpoint():
        checkpoints = list(Path(path).glob("checkpoint-*"))
        if len(checkpoints) == 0:
            return None
        return sorted(checkpoints)[-1]

    checkpoint = get_latest_checkpoint()
    if checkpoint:
        return PeftModel.from_pretrained(model, checkpoint), True
    print('checkpoint not found, return original model')
    return model, False
