# 선행 설치 패키지 with python=3.10
# pip install transformers konlpy optuna
# conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia

from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
from konlpy.tag import Okt
import os
import json
import multiprocessing as mp
import optuna
import requests

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

log_dirs = [
    '/mnt/c/workspace/kobart_model/logs',
    '/mnt/c/workspace/kobart_model/logs/logs.1',
    '/mnt/c/workspace/kobart_model/logs/logs.2',
    '/mnt/c/workspace/kobart_model/logs/results.1',
    '/mnt/c/workspace/kobart_model/logs/results.2'
]
for log_dir in log_dirs:
    os.makedirs(log_dir, exist_ok=True)

def load_stop_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stop_words = file.read().splitlines()
    return set(stop_words)

def load_keywords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        keywords = file.read().splitlines()
    return set(keywords)

def remove_stop_words(text, stop_words, keywords):
    okt = Okt()
    words = okt.morphs(text)
    filtered_words = [word for word in words if word not in stop_words or word in keywords]
    return ' '.join(filtered_words)

def chunk_text(text, max_length, tokenizer):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i + max_length]
        chunk = tokenizer.convert_tokens_to_string(chunk)
        chunks.append(chunk)
    return chunks

def preprocess_document(document, max_length, tokenizer, stop_words, keywords):
    try:
        full_text = ' '.join(sentence['sentence'] for part in document['text'] for sentence in part)
        summary = ' '.join(document.get('abstractive', []))

        if not summary.strip():
            return [], []

        full_text = remove_stop_words(full_text, stop_words, keywords)
        summary = remove_stop_words(summary, stop_words, keywords)

        text_chunks = chunk_text(full_text, max_length, tokenizer)

        texts = text_chunks
        summaries = [summary] * len(text_chunks)
        return texts, summaries
    except KeyError:
        return [], []

def load_and_preprocess_data(file_path, max_length, tokenizer, stop_words, keywords):
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)['documents']
    
    # 프로세스 풀 생성하여 멀티 프로세스 사용
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(preprocess_document, [(doc, max_length, tokenizer, stop_words, keywords) for doc in dataset])
    
    texts, summaries = [], []
    for doc_texts, doc_summaries in results:
        texts.extend(doc_texts)
        summaries.extend(doc_summaries)
    return texts, summaries

tokenizer = PreTrainedTokenizerFast.from_pretrained('hyunwoongko/kobart')
max_token_limit = 512

stop_words = load_stop_words('/mnt/c/workspace/kobart_model/불용어_신문기사.txt')
keywords = load_keywords('/mnt/c/workspace/kobart_model/키워드_신문기사.txt')

train_texts, train_summaries = load_and_preprocess_data('/mnt/c/workspace/train_v2.json', max_token_limit, tokenizer, stop_words, keywords)
validation_texts, validation_summaries = load_and_preprocess_data('/mnt/c/workspace/valid_v2.json', max_token_limit, tokenizer, stop_words, keywords)

def tokenize_data(texts, summaries):
    inputs = tokenizer(texts, max_length=max_token_limit, truncation=True, padding="max_length", return_tensors='pt')
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(summaries, max_length=128, truncation=True, padding="max_length", return_tensors='pt')
        
    return inputs, labels

train_inputs, train_labels = tokenize_data(train_texts, train_summaries)
validation_inputs, validation_labels = tokenize_data(validation_texts, validation_summaries)

class SummarizationDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs.input_ids)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        item['labels'] = self.labels.input_ids[idx]
        return item

train_dataset = SummarizationDataset(train_inputs, train_labels)
validation_dataset = SummarizationDataset(validation_inputs, validation_labels)

def model_init():
    return BartForConditionalGeneration.from_pretrained('hyunwoongko/kobart')

def objective(trial):
    training_args = TrainingArguments(
        output_dir='/mnt/c/workspace/kobart_model/logs/results.1',
        evaluation_strategy='epoch',
        learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 5e-4),
        per_device_train_batch_size=trial.suggest_categorical('per_device_train_batch_size', [4, 8, 16]),
        per_device_eval_batch_size=trial.suggest_categorical('per_device_eval_batch_size', [4, 8, 16]),
        num_train_epochs=trial.suggest_int('num_train_epochs', 1, 5),
        weight_decay=trial.suggest_loguniform('weight_decay', 1e-5, 1e-1),
        logging_dir='/mnt/c/workspace/kobart_model/logs/logs.1',
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    
    return eval_results['eval_loss']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print(f"Best hyperparameters: {study.best_params}")

best_params = study.best_params

training_args = TrainingArguments(
    output_dir='/mnt/c/workspace/kobart_model/logs/results.2',
    evaluation_strategy='epoch',
    learning_rate=best_params['learning_rate'],
    per_device_train_batch_size=best_params['per_device_train_batch_size'],
    per_device_eval_batch_size=best_params['per_device_eval_batch_size'],
    num_train_epochs=best_params['num_train_epochs'],
    weight_decay=best_params['weight_decay'],
    logging_dir='/mnt/c/workspace/kobart_model/logs/logs.2',
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
)

trainer = Trainer(
    model=model_init(),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Check_Sinmunkisa evaluation results: {eval_results}")

trainer.model.save_pretrained('./check_Sinmunkisa')
tokenizer.save_pretrained('./check_Sinmunkisa')
