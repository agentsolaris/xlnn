
import json
import logging
import sys

import numpy as np
import torch
from task_config import SuperGLUE_LABEL_MAPPING

from snorkel.mtl.data import MultitaskDataset

sys.path.append("..")  # Adds higher directory to python modules path.


logger = logging.getLogger(__name__)

TASK_NAME = "SST"


def parse(jsonl_path, tokenizer, max_data_samples, max_sequence_length):
    logger.info(f"Loading data from {jsonl_path}.")
    rows = [json.loads(row) for row in open(jsonl_path, encoding="utf-8")]
    for i in range(2):
        logger.info(f"Sample {i}: {rows[i]}")

    # Truncate to max_data_samples
    if max_data_samples:
        rows = rows[:max_data_samples]
        logger.info(f"Truncating to {max_data_samples} samples.")

    # sentence1 text
    sentence1s = []
    # sentence2 text
    sentence2s = []
    # label
    labels = []

    xlnet_token_ids = []
    xlnet_token_masks = []
    xlnet_token_segments = []

    # Check the maximum token length
    max_len = -1

    for row in rows:
        #index = row["idx"]
        sentence1 = row["sentence"]
        #sentence2 = row["#2 String"]
        label = row["label"] if "label" in row else 1

        sentence1s.append(sentence1)
        #sentence2s.append(sentence2)
        labels.append(SuperGLUE_LABEL_MAPPING[TASK_NAME][label])

        # Tokenize sentences
        sent1_tokens = tokenizer.tokenize(sentence1)
        #sent2_tokens = tokenizer.tokenize(sentence2)

        if len(sent1_tokens) > max_len:
            max_len = len(sent1_tokens)

        while True:
            total_length = len(sent1_tokens) 
            # Account for [CLS], [SEP], [SEP] with "- 3"
            if total_length <= max_sequence_length - 3:
                break

        # Convert to XLNET manner
        tokens = sent1_tokens + ["[SEP]"] + ["[CLS]"]
        token_segments = [0] * len(tokens)

        #tokens += sent2_tokens + ["[SEP]"]
        #token_segments += [1] * (len(sent2_tokens) + 1)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Generate mask where 1 for real tokens and 0 for padding tokens
        token_masks = [1] * len(token_ids)

        xlnet_token_ids.append(torch.LongTensor(token_ids))
        xlnet_token_masks.append(torch.LongTensor(token_masks))
        xlnet_token_segments.append(torch.LongTensor(token_segments))

    labels = torch.from_numpy(np.array(labels))

    logger.info(f"Max token len {max_len}")

    return MultitaskDataset(
        name="SuperGLUE",
        X_dict={
            "sentence": sentence1s,
            #"#2 String": sentence2s,
            "token_ids": xlnet_token_ids,
            "token_masks": xlnet_token_masks,
            "token_segments": xlnet_token_segments,
        },
        Y_dict={"labels": labels},
    )

