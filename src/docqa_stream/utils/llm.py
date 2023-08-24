import os

import transformers
from langchain import HuggingFacePipeline
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering


def get_model():
    model_config = AutoConfig.from_pretrained(
        os.getenv("QA_MODEL_ID"),
        cache_dir=os.getenv("HF_CACHE")
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        os.getenv("QA_MODEL_ID"),
        config=model_config,
        cache_dir=os.getenv("HF_CACHE")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        os.getenv("QA_MODEL_ID"),
        cache_dir=os.getenv("HF_CACHE"),
    )

    model.eval()

    pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task='question-answering',
        temperature=1,
        max_new_tokens=1024,
        repetition_penalty=1.1
    )

    llm = HuggingFacePipeline(pipeline=pipeline)
    return llm
