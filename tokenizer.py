import logging

from pytorch_transformers import *

logger = logging.getLogger(__name__)


def get_tokenizer(tokenizer_name):
    logger.info(f"Loading Tokenizer {tokenizer_name}")

    if tokenizer_name.startswith("xlnet"):
        do_lower_case = "uncased" in tokenizer_name
        tokenizer = XLNetTokenizer.from_pretrained(
            'xlnet-large-cased'
        )

    return tokenizer
