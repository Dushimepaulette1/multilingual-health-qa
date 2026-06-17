"""
Submission file generation, matching the required Zindi format:
ID, TargetRLF1, TargetR1F1, TargetLLM (all three target columns identical).
"""
import re
from pathlib import Path
import pandas as pd


def make_submission(ids, predictions, output_path, expected_length: int = None):
    """
    Build and save a valid Zindi submission file.

    Parameters
    ----------
    ids : array-like
        Row IDs matching the test set.
    predictions : list[str]
        Generated or retrieved answers.
    output_path : str or Path
    expected_length : int, optional
        If provided, asserts the submission has this many rows.

    Returns
    -------
    pd.DataFrame
        The submission DataFrame that was saved.
    """
    clean_preds = [re.sub(r'<extra_id_\d+>', '', str(p)).strip() for p in predictions]

    sub = pd.DataFrame({
        'ID': ids,
        'TargetRLF1': clean_preds,
        'TargetR1F1': clean_preds,
        'TargetLLM': clean_preds,
    })

    required_cols = ['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']
    assert list(sub.columns) == required_cols, f'Column mismatch: {list(sub.columns)}'
    if expected_length is not None:
        assert len(sub) == expected_length, \
            f'Row count mismatch: {len(sub)} vs {expected_length}'
    assert sub[['TargetRLF1', 'TargetR1F1', 'TargetLLM']].notna().all().all(), \
        'Missing values found'
    assert (sub['TargetRLF1'] == sub['TargetR1F1']).all(), 'TargetRLF1/TargetR1F1 differ'
    assert (sub['TargetRLF1'] == sub['TargetLLM']).all(), 'TargetRLF1/TargetLLM differ'

    output_path = Path(output_path)
    sub.to_csv(output_path, index=False, encoding='utf-8')
    print(f'Saved to {output_path} — shape {sub.shape}')

    return sub
