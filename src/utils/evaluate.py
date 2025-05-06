from torch.utils.data import DataLoader
from src.utils.dataset import BaseDataSet
from rouge_score import rouge_scorer
import bert_score
import datetime
from nltk.translate.bleu_score import modified_precision
import numpy as np



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


def get_summaries(dataset, config):
    target_set = get_target_set(dataset, config.get('target_set', 'test'))
    return [i[dataset.summary_key] for i in target_set]


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


def ngram_overlaps(sources, summaries, n=4):
    overlaps = [[] for _ in range(n)]
    for i in range(1, n + 1):
        for source, summary in zip(sources, summaries):
            source_tokens = source.split()
            summary_tokens = summary.split()

            frac = modified_precision([source_tokens], summary_tokens, i)

            overlaps[i - 1].append(frac.numerator / frac.denominator)

    return {f'{i+1}-gram overlaps': np.mean(overlaps[i]) for i in range(n)}


def compression_rate(sources, summaries):
    res = []
    for source, summary in zip(sources, summaries):
        source_tokens = source.split()
        summary_tokens = summary.split()
        res.append(len(summary_tokens) / len(source_tokens))
    return {'compression_rate': np.mean(res)}


def evaluate_metrics(model, dataset, config, metrics=['rouge', 'bert_score', 'compression_rate', 'ngram_overlaps']):
    # To-Do: metrics argument should be passed when it's called
    summaries = generate_summary(model, dataset, config)
    ground_truths = get_summaries(dataset, config)
    sources = None
    if 'compression_rate' in metrics or 'ngram_overlaps' in metrics:
        sources = get_sources(dataset, config)

    scores = {}
    for metric in metrics:
        if metric == 'rouge':
            scores.update(evaluate_rouge(ground_truths, summaries))
        elif metric == 'bert_score':
            scores.update(evaluate_bert_score(ground_truths, summaries))
        elif metric == 'compression_rate':
            scores.update(compression_rate(sources, summaries))
        elif metric == 'ngram_overlaps':
            scores.update(ngram_overlaps(sources, summaries))

    return scores


