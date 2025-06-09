# -*- coding: utf-8 -*-
"""
data_utils.py

Utilities for loading, preprocessing, and partitioning the MIMIC-IV-Ext-CDM dataset
for federated fine-tuning.

Note: This script assumes the necessary MIMIC-IV-Ext-CDM CSV files
(e.g., history_of_present_illness.csv, physical_examination.csv) are located
in the specified data directory. Access to MIMIC-IV requires credentials and
completion of training via PhysioNet.
"""

import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np
import logging
import pickle
from config import DATA_DIR
import torch
from torch.utils.data import random_split
from functools import lru_cache
import concurrent.futures
from typing import Dict, List, Set, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
DEFAULT_DATA_DIR = "/data"
# Define which CSV files and columns contain the text data for fine-tuning
# Adjust based on the specific fine-tuning task (e.g., diagnosis prediction, text generation)
# Example: Using History of Present Illness and Physical Examination
DATA_FILES_COLS = {
    "history_of_present_illness.csv": "hpi",
    "physical_examination.csv": "pe",
    "discharge_diagnosis.csv": "discharge_diagnosis",
    "discharge_procedures.csv": "discharge_procedure",
    "icd_diagnosis.csv": "icd_diagnosis",
    "icd_procedures.csv": "icd_code, icd_title, icd_version",
    "lab_test_mapping.csv": "label,fluid,category,count,corresponding_ids",
    "laboratory_tests.csv": "litemid,valuestr,ref_range_lower,ref_range_upper",
    "microbiology.csv": "test_itemid,valuestr,spec_itemid",
    "radiology_reports.csv": "note_id,modality,region,exam_name,text",
    "pathology.json": "pathology",
}
# Define the primary key column common across files (usually related to hospital admission)
ID_COLUMN = "hadm_id"

# Tokenizer settings (using GPT-2 tokenizer as a placeholder, adjust if using a different base model like Llama)
TOKENIZER_NAME = "llama3-8b-4096"
MAX_LENGTH = 4096 # Max sequence length for tokenizer, adjust based on model and memory

class DataCache:
    """메모리 캐시를 위한 클래스"""
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataCache, cls).__new__(cls)
            cls._instance.lab_map_cache = {}
        return cls._instance

    @lru_cache(maxsize=1024)
    def get_lab_label(self, litemid: str) -> str:
        return self.lab_map_cache.get(str(litemid))

    def build_lab_map_cache(self, lab_map_df: pd.DataFrame) -> None:
        """lab mapping 캐시 구축"""
        self.lab_map_cache.clear()
        for _, row in lab_map_df.iterrows():
            ids = str(row['corresponding_ids']).split(';')
            for id_ in ids:
                self.lab_map_cache[id_] = row['label']

def load_mimic_data(data_dir='data', use_cache=True) -> pd.DataFrame:
    """최적화된 MIMIC 데이터 로딩"""
    cache_file = os.path.join(data_dir, 'processed_mimic_cache.pkl')
    
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # 파일 경로 설정
    paths = {
        'hpi': os.path.join(data_dir, 'history_of_present_illness.csv'),
        'lab_map': os.path.join(data_dir, 'lab_test_mapping.csv'),
        'lab': os.path.join(data_dir, 'laboratory_tests.csv'),
        'micro': os.path.join(data_dir, 'microbiology.csv'),
        'radio': os.path.join(data_dir, 'radiology_reports.csv'),
        'diag': os.path.join(data_dir, 'discharge_diagnosis.csv'),
        'proc': os.path.join(data_dir, 'discharge_procedures.csv')
    }

    # 병렬 데이터 로딩
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {name: executor.submit(pd.read_csv, path) for name, path in paths.items()}
        dfs = {name: future.result() for name, future in futures.items()}

    # 캐시 초기화 및 구축
    cache = DataCache()
    cache.build_lab_map_cache(dfs['lab_map'])

    # Lab 데이터에 레이블 추가 (벡터화된 연산)
    dfs['lab']['label'] = dfs['lab']['litemid'].astype(str).map(cache.get_lab_label)

    # hadm_id 집합 연산 최적화
    hadm_ids = set(dfs['hpi']['hadm_id'])
    for df_name in ['lab', 'micro', 'radio']:
        hadm_ids &= set(dfs[df_name]['hadm_id'])
    hadm_ids &= set(dfs['diag']['hadm_id'])
    hadm_ids &= set(dfs['proc']['hadm_id'])

    # 데이터 처리 및 텍스트 생성 최적화
    full_texts = []
    for hadm_id in sorted(hadm_ids):
        text_parts = []
        
        # HPI
        hpi = dfs['hpi'].loc[dfs['hpi']['hadm_id'] == hadm_id, 'hpi'].iloc[0] if not dfs['hpi'].empty else ''
        text_parts.append(f"HPI: {hpi}")
        
        # Lab results
        lab_tests = dfs['lab'][dfs['lab']['hadm_id'] == hadm_id]
        if not lab_tests.empty:
            lab_summary = '; '.join(f"{row['label']}: {row['valuestr']}" for _, row in lab_tests.iterrows() if row['label'])
            text_parts.append(f"Laboratory Test: {lab_summary}")
        
        # Microbiology
        micro = dfs['micro'][dfs['micro']['hadm_id'] == hadm_id]
        if not micro.empty:
            micro_summary = '; '.join(f"{row['test_itemid']}: {row['valuestr']}" for _, row in micro.iterrows())
            text_parts.append(f"Microbiology: {micro_summary}")
        
        # Radiology
        radio = dfs['radio'][dfs['radio']['hadm_id'] == hadm_id]
        if not radio.empty:
            radio_summary = '; '.join(str(row['text']) for _, row in radio.iterrows())
            text_parts.append(f"Radiology: {radio_summary}")
        
        # Diagnosis & Procedures
        diag = dfs['diag'].loc[dfs['diag']['hadm_id'] == hadm_id, 'discharge_diagnosis'].iloc[0] if not dfs['diag'].empty else ''
        proc = dfs['proc'].loc[dfs['proc']['hadm_id'] == hadm_id, 'discharge_procedure'].iloc[0] if not dfs['proc'].empty else ''
        
        text = (
            f"In the case of {hadm_id} of the patient:\n" +
            '\n'.join(text_parts) +
            f'\nThe patient was diagnosed with "{diag}" and underwent "{proc}".\n'
        )
        
        full_texts.append({'hadm_id': hadm_id, 'text': text})

    mimic_df = pd.DataFrame(full_texts)
    
    # 캐시 저장
    if use_cache:
        with open(cache_file, 'wb') as f:
            pickle.dump(mimic_df, f)
    
    return mimic_df

def preprocess_and_tokenize(data_df: pd.DataFrame, tokenizer_name: str = TOKENIZER_NAME, max_length: int = MAX_LENGTH) -> Dataset:
    """토크나이징 최적화"""
    logging.info(f"Starting preprocessing and tokenization using {tokenizer_name}")
    
    # 토크나이저 로딩 최적화
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 배치 처리를 위한 함수
    def tokenize_batch(texts: List[str]) -> Dict:
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    # 데이터셋 변환 및 토크나이징
    dataset = Dataset.from_pandas(data_df)
    
    # 배치 크기 최적화
    batch_size = 32
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_batch(examples["text"]),
        batched=True,
        batch_size=batch_size,
        remove_columns=["text", ID_COLUMN, "__index_level_0__"]
    )
    
    tokenized_dataset.set_format("torch")
    logging.info("Tokenization complete.")
    return tokenized_dataset

def partition_data(dataset: Dataset, num_clients: int = 16, alpha: float = 0.5) -> List[Dataset]:
    """데이터 파티셔닝 최적화"""
    logging.info(f"Partitioning data for {num_clients} clients using Dirichlet (alpha={alpha})")
    
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    # Dirichlet 분포를 사용한 샘플 수 계산
    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
    client_sample_counts = (proportions * num_samples).astype(int)
    
    # 샘플 수 조정
    diff = num_samples - client_sample_counts.sum()
    client_sample_counts[:diff] += 1
    
    # 클라이언트별 데이터셋 생성
    client_datasets = []
    start_idx = 0
    for count in client_sample_counts:
        client_indices = indices[start_idx:start_idx + count]
        client_datasets.append(dataset.select(client_indices))
        start_idx += count
        
    return client_datasets

def create_full_texts_from_dataset(dataset_dir=DATA_DIR, output_file='data/full_texts.csv'):
    """
    Combine multiple csv files to create full texts for each hadm_id and save them.
    """
    # File paths
    hpi_path = os.path.join(dataset_dir, 'history_of_present_illness.csv')
    lab_map_path = os.path.join(dataset_dir, 'lab_test_mapping.csv')
    lab_path = os.path.join(dataset_dir, 'laboratory_tests.csv')
    micro_path = os.path.join(dataset_dir, 'microbiology.csv')
    radio_path = os.path.join(dataset_dir, 'radiology_reports.csv')
    diag_path = os.path.join(dataset_dir, 'discharge_diagnosis.csv')
    proc_path = os.path.join(dataset_dir, 'discharge_procedures.csv')

    # Load data
    hpi_df = pd.read_csv(hpi_path)
    lab_map_df = pd.read_csv(lab_map_path)
    lab_df = pd.read_csv(lab_path)
    micro_df = pd.read_csv(micro_path)
    radio_df = pd.read_csv(radio_path)
    diag_df = pd.read_csv(diag_path)
    proc_df = pd.read_csv(proc_path)

    # laboratory_tests에 label 붙이기
    def get_lab_label(litemid):
        for _, row in lab_map_df.iterrows():
            ids = str(row['corresponding_ids']).split(';')
            if str(litemid) in ids:
                return row['label']
        return None

    lab_df['label'] = lab_df['litemid'].apply(get_lab_label)

    # hadm_id 목록
    hadm_ids = set(hpi_df['hadm_id'])
    hadm_ids |= set(lab_df['hadm_id'])
    hadm_ids |= set(micro_df['hadm_id'])
    hadm_ids |= set(radio_df['hadm_id'])
    hadm_ids &= set(diag_df['hadm_id'])
    hadm_ids &= set(proc_df['hadm_id'])

    full_texts = []
    for hadm_id in sorted(hadm_ids):
        # HPI
        hpi_row = hpi_df[hpi_df['hadm_id'] == hadm_id]
        hpi = hpi_row['hpi'].iloc[0] if not hpi_row.empty else ''
        # Laboratory Test
        lab_tests = lab_df[lab_df['hadm_id'] == hadm_id]
        lab_summary = '; '.join([f"{row['label']}: {row['valuestr']}" for _, row in lab_tests.iterrows() if row['label']])
        # Microbiology
        micro = micro_df[micro_df['hadm_id'] == hadm_id]
        micro_summary = '; '.join([f"{row['test_itemid']}: {row['valuestr']}" for _, row in micro.iterrows()])
        # Radiology
        radio = radio_df[radio_df['hadm_id'] == hadm_id]
        radio_summary = '; '.join([str(row['text']) for _, row in radio.iterrows()])
        # Discharge Diagnosis
        diag_row = diag_df[diag_df['hadm_id'] == hadm_id]
        diag = diag_row['discharge_diagnosis'].iloc[0] if not diag_row.empty else ''
        # Discharge Procedures
        proc_row = proc_df[proc_df['hadm_id'] == hadm_id]
        proc = proc_row['discharge_procedure'].iloc[0] if not proc_row.empty else ''

        text = (
            f"The patient has the following information:\n"
            f"has HPI: {hpi}\n"
            f"Lab results:\n"
            f"- Laboratory Test: {lab_summary}\n"
            f"- Microbiology: {micro_summary}\n"
            f"- Radiology: {radio_summary}\n"
            f'The patient was diagnosed with "{diag}" and underwent "{proc}".\n'
        )

        full_texts.append({'hadm_id': hadm_id, 'text': text, 'diagnosis': diag})

    result_df = pd.DataFrame(full_texts)
    result_df.to_csv(output_file, index=False)

def load_full_texts_csv(csv_path='dataset/full_texts.csv'):
    """
    Load full_texts.csv file and return a DataFrame.
    """
    df = pd.read_csv(csv_path)
    return df

# Example usage flow (can be called from the main FL script)
if __name__ == '__main__':
    logging.info("--- Running Data Utils Example --- ")
    # 1. Load data
    mimic_df = load_mimic_data()

    if mimic_df is not None and not mimic_df.empty:
        logging.info(f"Loaded DataFrame shape: {mimic_df.shape}")
        logging.info(f"DataFrame columns: {mimic_df.columns.tolist()}")
        logging.info(f"Sample text entry:\n{mimic_df['text'].iloc[0][:500]}...")

        # 2. Preprocess and tokenize
        tokenized_data = preprocess_and_tokenize(mimic_df)
        logging.info(f"Tokenized dataset features: {tokenized_data.features}")
        logging.info(f"Sample tokenized entry: {tokenized_data[0]}")

        # 3. Partition data
        num_federated_clients = 16
        client_datasets = partition_data(tokenized_data, num_clients=num_federated_clients, alpha=0.5)
        logging.info(f"Created {len(client_datasets)} client datasets.")
        logging.info(f"Size of first client dataset: {len(client_datasets[0])}")

        # 4. Split client datasets into train, validation, and test sets
        for i, client_dataset in enumerate(client_datasets):
            n = len(client_dataset)
            train_size = int(0.8 * n)
            val_size = int(0.1 * n)
            test_size = n - train_size - val_size
            train_ds, val_ds, test_ds = torch.utils.data.random_split(
                client_dataset, [train_size, val_size, test_size]
            )
            logging.info(f"Client {i}: Train size: {len(train_ds)}, Validation size: {len(val_ds)}, Test size: {len(test_ds)}")
    else:
        logging.error("Failed to load or process data. Exiting example.")
    logging.info("--- Data Utils Example Finished --- ")

