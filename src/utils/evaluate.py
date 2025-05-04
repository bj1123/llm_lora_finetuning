from torch.utils.data import DataLoader
from src.utils.dataset import BaseDataSet
from rouge_score import rouge_scorer
import numpy as np
import bert_score
import datetime


def get_target_set(dataset: BaseDataSet, data_split):
    if data_split == 'test':
        target_set = dataset.test_set
    elif data_split == 'validation':
        target_set = dataset.valid_set
    else:
        raise ValueError
    return target_set


def generate_summary(model, dataset: BaseDataSet, config):
    target_set = get_target_set(dataset, config.get('target_set', 'test'))
    collator = dataset.get_collator()

    # for batch generation, left padding is needed
    padding_side_mem = collator.tokenizer.padding_side
    collator.tokenizer.padding_side = 'left'

    res = []
    dl = DataLoader(target_set.map(dataset.generation_data_process, remove_columns=target_set.column_names),
                    batch_size=config.get('batch_size', 1), shuffle=False, collate_fn=collator)
    for i, batch in enumerate(dl):
        print(i, datetime.datetime.now())
        batch['input_ids'] = batch['input_ids'].to(model.device)
        batch['attention_mask'] = batch['attention_mask'].to(model.device)

        input_len = batch['input_ids'].shape[1]
        outputs = model.generate(**batch,
                                 max_new_tokens=dataset.config.get('max_summary_len', 128),
                                 eos_token_id=dataset.tokenizer.eos_token_id,
                                 pad_token_id=dataset.tokenizer.pad_token_id,
                                 num_beams=config.get('num_beams', 1),
                                 do_sample=config.get('do_sample', False),
                                 temperature=config.get('temperature', 1.),
                                 top_p=config.get('top_p', 1.0),
                                 )
        res.extend(outputs[:, input_len:].tolist())

    # roll back to original padding side
    collator.tokenizer.padding_side = padding_side_mem
    decoded = collator.tokenizer.batch_decode(res)
    return decoded


def get_sources(dataset, config):
    target_set = get_target_set(dataset, config.get('target_set', 'test'))
    return [i[dataset.source_key] for i in target_set]


def evaluate_rouge(ground_truths, generated_texts):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    r1, r2, rl = [], [], []
    for src, tgt in zip(ground_truths, generated_texts):
        score = scorer.score(src, tgt)
        r1.append(score['rouge1'].fmeasure)
        r2.append(score['rouge2'].fmeasure)
        rl.append(score['rougeL'].fmeasure)

    return {'rouge1': np.mean(r1),
            'rouge2': np.mean(r2),
            'rougeL': np.mean(rl)}


def evaluate_bert_score(ground_truths, generated_texts, lang='en', model_type='roberta-large'):
    _, _, bert_score_f1 = bert_score.score(generated_texts, ground_truths, lang=lang, model_type=model_type)
    return {'bert_score': np.mean(bert_score_f1.tolist())}


def evaluate_metrics(model, dataset, config, metrics=['rouge', 'bert_score']):
    def get_ground_truths():
        summary_key = dataset.summary_key
        target_set = get_target_set(dataset, config.get('target_set', 'test'))
        return [i[summary_key] for i in target_set]

    summaries = generate_summary(model, dataset, config)
    ground_truths = get_ground_truths()
    scores = {}
    for metric in metrics:
        if metric == 'rouge':
            scores.update(evaluate_rouge(ground_truths, summaries))
        elif metric == 'bert_score':
            scores.update(evaluate_bert_score(ground_truths, summaries))

    return scores


