"""
Semantic retrieval utilities - the best-performing approach in this project
(Experiment 5/8). Embeds questions with a multilingual sentence-embedding
model and retrieves the most similar training question's answer.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    """
    Top-1 semantic retrieval restricted to matching language subset.

    Usage
    -----
    retriever = SemanticRetriever(model_name='sentence-transformers/LaBSE')
    retriever.fit(train_df, question_col='input', answer_col='output', lang_col='subset')
    answer = retriever.retrieve(query_embedding, subset='Amh_Eth')
    """

    def __init__(self, model_name: str = 'sentence-transformers/LaBSE', device: str = None):
        import torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name, device=self.device)
        self.questions = None
        self.answers = None
        self.embeddings = None
        self.subset_indices = {}

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of texts into normalized embeddings."""
        return self.encoder.encode(
            texts, batch_size=batch_size,
            show_progress_bar=True, normalize_embeddings=True
        )

    def fit(self, df, question_col: str, answer_col: str, lang_col: str):
        """Build the retrieval index from a DataFrame of question/answer pairs."""
        self.questions = df[question_col].tolist()
        self.answers = df[answer_col].reset_index(drop=True)
        self.embeddings = self.embed(self.questions)

        for subset_code in df[lang_col].unique():
            idx = np.where(df[lang_col].values == subset_code)[0]
            self.subset_indices[subset_code] = idx

        return self

    def retrieve(self, query_embedding: np.ndarray, subset: str):
        """Return the answer for the most similar question within the given subset."""
        idx = self.subset_indices.get(subset)
        if idx is None or len(idx) == 0:
            idx = np.arange(len(self.questions))

        candidate_embeddings = self.embeddings[idx]
        sims = cosine_similarity(
            query_embedding.reshape(1, -1), candidate_embeddings
        ).flatten()
        best_local = int(np.argmax(sims))
        best_global = int(idx[best_local])
        return self.answers.iloc[best_global]

    def retrieve_batch(self, queries: list[str], subsets: list[str], batch_size: int = 64):
        """Retrieve answers for a batch of queries."""
        query_embeddings = self.embed(queries, batch_size=batch_size)
        return [
            self.retrieve(query_embeddings[i], subsets[i])
            for i in range(len(queries))
        ]
