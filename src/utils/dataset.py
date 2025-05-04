from abc import ABC, abstractmethod
from datasets import Dataset, load_dataset, ReadInstruction
from transformers import DataCollatorForSeq2Seq


class BaseDataSet(ABC):
    DEFAULT_SOURCE_KEY = 'source'
    DEFAULT_SUMMARY_KEY = 'summary'

    def __init__(self, config, tokenizer):
        self.config = config
        self.source_key = config.get('source_key', BaseDataSet.DEFAULT_SOURCE_KEY)
        self.summary_key = config.get('summary_key', BaseDataSet.DEFAULT_SUMMARY_KEY)
        self.source_max_len = config.get('max_source_len', 0)
        self.summary_max_len = config.get('max_summary_len', 0)

        self.tokenizer = tokenizer
        self.messages_template = config.get('prompts', self._get_default_message())
        self.train_set, self.valid_set, self.test_set = self._load_dataset()

        # to handle missing pad token for llama tokenizer
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})

    @abstractmethod
    def _load_dataset(self):
        pass

    @staticmethod
    def _get_default_message():
        messages = [{'role': 'user', 'content': 'Summary this article : ', 'text_type': 'source'}]
        return messages

    def get_collator(self, ):
        return DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True
        )

    def _get_split_ratios(self):
        train_ratio = self.config.get('train_ratio', 90)
        val_ratio = train_ratio + self.config.get('valid_ratio', 5)
        test_ratio = val_ratio + self.config.get('test_ratio', 5)
        return train_ratio, val_ratio, test_ratio

    def _truncate_text(self, text, max_tokens):
        if max_tokens == 0:
            return text
        tokens = self.tokenizer.tokenize(text)
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.convert_tokens_to_string(truncated_tokens)

    def _encode_source_text(self, data):
        source = data.get(self.source_key, "")
        source = self._truncate_text(source, self.source_max_len)
        messages = []
        for message in self.messages_template:
            message = message.copy()
            if message.get('text_type', '') == 'source':
                message['content'] = message['content'] + source
            messages.append(message)

        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    def training_data_process(self, data):
        summary = data.get(self.summary_key, "")
        summary = self._truncate_text(summary, self.summary_max_len)

        source_encoded = self._encode_source_text(data)
        summary_encoded = self.tokenizer.encode(summary, add_special_tokens=False) + [self.tokenizer.eos_token_id]

        # mask out source text for labels
        labels = [-100] * len(source_encoded) + summary_encoded

        return {
            "input_ids": source_encoded + summary_encoded,
            "labels": labels
        }

    def generation_data_process(self, data):
        return {"input_ids": self._encode_source_text(data)}


class HuggingFaceHubDataset(BaseDataSet):
    def _load_dataset(self):
        train_ratio, val_ratio, test_ratio = self._get_split_ratios()

        return load_dataset(self.config.get('dataset_name'), self.config.get('dataset_version'),
                            split=[f"train[:{train_ratio}%]",
                                   f"train[{train_ratio}%:{val_ratio}%]",
                                   f"train[{val_ratio}%:{test_ratio}]"])


class CSVDataset(BaseDataSet):
    def _load_dataset(self):
        return None, None, Dataset.from_csv(self.config.get('data_path'), column_names=[self.source_key, self.summary_key])


class TxtDataset(BaseDataSet):
    # txt dataset only supports generation
    # The whole text file will be treated as a single source text.
    def _load_dataset(self):
        with open(self.config.get('data_path')) as f:
            temp = f.readlines()
        data = [{'source': '\n'.join(temp)}]
        return None, None, Dataset.from_list(data)


def get_dataset(tokenizer, config):
    # simple factory
    dataset_map = {'csv': CSVDataset,
                   'txt': TxtDataset,
                   'huggingface_hub': HuggingFaceHubDataset}

    data_type = config.get('data_type')
    if data_type not in dataset_map:
        raise ValueError(f'{data_type} dataset is invalid')

    return dataset_map[data_type](config, tokenizer)
