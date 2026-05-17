"""
mT5 fine-tuning utilities (Experiments 2-3). Generative seq2seq fine-tuning
approach, included for comparison against the retrieval-based approach in
retrieval.py, which performed better on this task (see EXPERIMENT_LOG.md).
"""
import re
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,
)
from datasets import Dataset


def build_prompt(question: str, language: str = None) -> str:
    """Build the input prompt for the model. Currently a passthrough -
    language conditioning was tested but not found necessary for this setup."""
    return str(question).strip()


def load_model(model_name: str = 'google/mt5-small', device: str = None):
    """Load tokenizer and model, moved to the appropriate device."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(device)
    return tokenizer, model, device


def make_hf_dataset(df, tokenizer, question_col, answer_col,
                     max_input_len=128, max_target_len=256):
    """Convert a DataFrame to a tokenized HuggingFace Dataset for training."""
    records = [
        {'prompt': build_prompt(str(r[question_col])), 'answer': str(r[answer_col])}
        for _, r in df.iterrows()
    ]
    raw_ds = Dataset.from_list(records)

    def preprocess(examples):
        model_inputs = tokenizer(
            examples['prompt'], max_length=max_input_len, truncation=True, padding=False
        )
        labels = tokenizer(
            text_target=examples['answer'], max_length=max_target_len,
            truncation=True, padding=False
        )
        model_inputs['labels'] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in seq]
            for seq in labels['input_ids']
        ]
        return model_inputs

    return raw_ds.map(preprocess, batched=True, remove_columns=['prompt', 'answer'])


def get_training_args(output_dir: str, epochs: int = 3, batch_size: int = 8,
                       learning_rate: float = 5e-4, device: str = 'cuda',
                       max_target_len: int = 256):
    """
    Training arguments tuned for single-GPU, memory-constrained fine-tuning.

    Key choices (see EXPERIMENT_LOG.md and report Section 5 for rationale):
    - optim='adafactor': avoids Adam's extra optimizer-state memory overhead
    - generation_num_beams=1: greedy decoding, far less memory-hungry than
      beam search during evaluation
    """
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        eval_accumulation_steps=1,
        generation_num_beams=1,
        learning_rate=learning_rate,
        optim="adafactor",
        predict_with_generate=True,
        bf16=(device == 'cuda' and torch.cuda.is_bf16_supported()),
        fp16=(device == 'cuda' and not torch.cuda.is_bf16_supported()),
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        logging_steps=200,
        generation_max_length=max_target_len,
        report_to='none',
    )


def generate_answers_batch(model, tokenizer, questions, languages=None,
                            device='cuda', batch_size=16,
                            max_input_len=128, max_output_len=256):
    """Generate answers for a list of questions using greedy decoding."""
    if languages is None:
        languages = [None] * len(questions)

    all_answers = []
    n_batches = (len(questions) + batch_size - 1) // batch_size

    for b in range(n_batches):
        start, end = b * batch_size, min((b + 1) * batch_size, len(questions))
        prompts = [build_prompt(q, l)
                   for q, l in zip(questions[start:end], languages[start:end])]

        inputs = tokenizer(
            prompts, return_tensors='pt', padding=True,
            truncation=True, max_length=max_input_len
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_output_len,
                num_beams=1, no_repeat_ngram_size=3
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        cleaned = [re.sub(r'<extra_id_\d+>', '', a).strip() for a in decoded]
        all_answers.extend(cleaned)

    return all_answers
