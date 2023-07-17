import gzip

from sentence_transformers import SentenceTransformer, util

from config import SBertConfig


class GzipSemanticSearch():
    
    def compute_distance(s1: str, s2: str) -> float:
        cs1 = len(gzip.compress(s1.encode()))
        cs2 = len(gzip.compress(s2.encode()))
        s1s2 = " ".join([s1, s2])
        cs1s2 = len(gzip.compress(s1s2.encode()))
        ncd = (cs1s2 - min(cs1, cs2)) / max(cs1, cs2)
        return ncd


class SBert():

    def __init__(self) -> None:
        self.model = SentenceTransformer(SBertConfig.model_name)

    def compute_distance(self, s1: str, s2: str):
        embeddings1 = self.model.encode(s1, convert_to_tensor=True)
        embeddings2 = self.model.encode(s2, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        return cosine_scores


