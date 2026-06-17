"""
Data loading and preprocessing utilities for the Multilingual Health QA project.
"""
from pathlib import Path
import pandas as pd

QUESTION_COL = 'input'
ANSWER_COL = 'output'
LANG_COL = 'subset'
ID_COL = 'ID'

SUBSET_TO_LANGUAGE = {
    'Eng': 'English',
    'Aka': 'Akan',
    'Lug': 'Luganda',
    'Swa': 'Swahili',
    'Amh': 'Amharic',
}


def subset_to_language_name(subset_code: str) -> str:
    """Extract the full language name from a subset code such as 'Amh_Eth'."""
    if not subset_code or not isinstance(subset_code, str):
        return 'English'
    lang_prefix = subset_code.split('_')[0]
    return SUBSET_TO_LANGUAGE.get(lang_prefix, subset_code)


def clean_text(x) -> str:
    """Strip whitespace and handle null values."""
    if pd.isna(x):
        return ''
    return str(x).strip()


def load_data(data_dir: str | Path):
    """
    Load and clean the Train/Test/Val CSVs.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing Train.csv, Test.csv, Val.csv, SampleSubmission.csv

    Returns
    -------
    tuple of (train, test, val) DataFrames, cleaned and with empty rows removed.
    """
    data_dir = Path(data_dir)

    train = pd.read_csv(data_dir / 'Train.csv')
    test = pd.read_csv(data_dir / 'Test.csv')
    val = pd.read_csv(data_dir / 'Val.csv')

    for df, has_answer in [(train, True), (val, True), (test, False)]:
        df[QUESTION_COL] = df[QUESTION_COL].map(clean_text)
        if has_answer:
            df[ANSWER_COL] = df[ANSWER_COL].map(clean_text)

    train = train[(train[QUESTION_COL] != '') & (train[ANSWER_COL] != '')].reset_index(drop=True)
    val = val[(val[QUESTION_COL] != '') & (val[ANSWER_COL] != '')].reset_index(drop=True)
    test = test[test[QUESTION_COL] != ''].reset_index(drop=True)

    return train, test, val
