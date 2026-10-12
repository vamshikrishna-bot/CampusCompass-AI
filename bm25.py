from collections import Counter
import math
import re
from typing import Generic, Protocol, Sequence, TypeVar


class TextDocument(Protocol):
    page_content: str


DocumentT = TypeVar("DocumentT", bound=TextDocument)


class BM25Retriever(Generic[DocumentT]):
    """Retrieve document chunks by lexical relevance without loading an ML model."""

    def __init__(self, documents: Sequence[DocumentT], k: int = 8) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")

        self._documents = list(documents)
        self._k = k
        self._tokens = [self._tokenize(doc.page_content) for doc in self._documents]
        self._frequencies = [Counter(tokens) for tokens in self._tokens]
        self._document_frequency = Counter(
            term for frequencies in self._frequencies for term in frequencies
        )
        self._average_length = (
            sum(map(len, self._tokens)) / len(self._tokens) if self._tokens else 0.0
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def invoke(self, query: str) -> list[DocumentT]:
        query_terms = set(self._tokenize(query))
        if not query_terms or not self._documents or self._average_length == 0:
            return []

        document_count = len(self._documents)
        scored_documents = []
        for index, (tokens, frequencies) in enumerate(
            zip(self._tokens, self._frequencies)
        ):
            length = len(tokens)
            score = 0.0
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue

                document_frequency = self._document_frequency[term]
                inverse_frequency = math.log(
                    1
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = term_frequency + 1.5 * (
                    1 - 0.75 + 0.75 * length / self._average_length
                )
                score += inverse_frequency * term_frequency * 2.5 / denominator

            if score > 0:
                scored_documents.append((score, index, self._documents[index]))

        scored_documents.sort(key=lambda item: (-item[0], item[1]))
        return [document for _, _, document in scored_documents[: self._k]]
