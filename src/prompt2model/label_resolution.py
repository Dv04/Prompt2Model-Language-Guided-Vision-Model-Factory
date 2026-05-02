from __future__ import annotations

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import torch

from prompt2model.config import RequestedLabel, ResolvedLabel

logger = logging.getLogger(__name__)


@dataclass
class _ClipCache:
    tokenizer: Any
    model: Any


@dataclass
class SimilarityMatrix:
    """Full cosine similarity matrix between user labels and dataset labels."""

    user_labels: list[str]
    dataset_labels: list[str]
    scores: list[list[float]]
    assignments: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_labels": self.user_labels,
            "dataset_labels": self.dataset_labels,
            "scores": self.scores,
            "assignments": self.assignments,
        }


@dataclass
class ResolutionResult:
    """Combined resolution output containing both resolved labels and the similarity matrix."""

    resolved_labels: list[ResolvedLabel]
    similarity_matrix: SimilarityMatrix | None


class LabelResolver:
    def __init__(
        self,
        threshold: float = 0.25,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        enable_clip: bool = True,
    ) -> None:
        self.threshold = threshold
        self.clip_model_name = clip_model_name
        self.enable_clip = enable_clip
        self._cache: _ClipCache | None = None

    def resolve(
        self,
        requested_labels: list[RequestedLabel],
        dataset_labels: list[str],
    ) -> list[ResolvedLabel]:
        """Resolve labels using a three-tier fallback chain: CLIP → lexical → substring."""
        if not dataset_labels:
            return [
                ResolvedLabel(
                    requested_label=request.name,
                    dataset_label=request.name,
                    score=1.0,
                    method="identity",
                )
                for request in requested_labels
            ]

        if self.enable_clip:
            try:
                resolved = self._resolve_semantic(requested_labels, dataset_labels)
                logger.info("Label resolution completed via CLIP semantic matching")
                return resolved
            except Exception as exc:
                logger.info("CLIP unavailable (%s), falling back to lexical", exc)

        lexical = self._resolve_lexical(requested_labels, dataset_labels)
        # Check if any labels have very low lexical scores and try substring fallback
        final: list[ResolvedLabel] = []
        for resolved_label in lexical:
            if resolved_label.score < self.threshold:
                logger.warning(
                    "Lexical match for '%s' → '%s' below threshold (%.3f < %.3f), trying substring",
                    resolved_label.requested_label,
                    resolved_label.dataset_label,
                    resolved_label.score,
                    self.threshold,
                )
                # Find the original RequestedLabel
                request = next(
                    (r for r in requested_labels if r.name == resolved_label.requested_label),
                    RequestedLabel(name=resolved_label.requested_label),
                )
                substring_result = self._resolve_substring([request], dataset_labels)
                if substring_result and substring_result[0].score > resolved_label.score:
                    final.append(substring_result[0])
                    continue
                logger.warning(
                    "No strong match found for '%s' in dataset labels %s",
                    resolved_label.requested_label,
                    dataset_labels,
                )
            final.append(resolved_label)
        return final

    def resolve_with_matrix(
        self,
        requested_labels: list[RequestedLabel],
        dataset_labels: list[str],
    ) -> ResolutionResult:
        """Resolve labels and return the full cosine similarity matrix artifact."""
        resolved = self.resolve(requested_labels, dataset_labels)
        matrix = self.compute_similarity_matrix(requested_labels, dataset_labels)
        return ResolutionResult(resolved_labels=resolved, similarity_matrix=matrix)

    def compute_similarity_matrix(
        self,
        requested_labels: list[RequestedLabel],
        dataset_labels: list[str],
    ) -> SimilarityMatrix:
        """Compute the full user×dataset cosine similarity matrix.

        When CLIP is available, produces semantic similarity scores.
        Otherwise falls back to lexical SequenceMatcher ratios.
        """
        user_names = [req.name for req in requested_labels]
        if not dataset_labels:
            return SimilarityMatrix(
                user_labels=user_names,
                dataset_labels=[],
                scores=[],
                assignments={name: name for name in user_names},
            )

        if self.enable_clip:
            try:
                return self._compute_clip_matrix(requested_labels, dataset_labels, user_names)
            except Exception:
                logger.info("CLIP unavailable for matrix computation, falling back to lexical")

        return self._compute_lexical_matrix(requested_labels, dataset_labels, user_names)

    def _compute_clip_matrix(
        self,
        requested_labels: list[RequestedLabel],
        dataset_labels: list[str],
        user_names: list[str],
    ) -> SimilarityMatrix:
        dataset_embeddings = self._embed_texts(dataset_labels)
        scores: list[list[float]] = []
        assignments: dict[str, str] = {}

        for request, user_name in zip(requested_labels, user_names):
            phrases = [request.name, *request.synonyms]
            request_embedding = self._embed_texts(phrases).mean(dim=0, keepdim=True)
            request_embedding = torch.nn.functional.normalize(request_embedding, dim=-1)
            similarities = (request_embedding @ dataset_embeddings.T).squeeze(0)
            row = [float(s.item()) for s in similarities]
            scores.append(row)
            best_index = int(torch.argmax(similarities).item())
            assignments[user_name] = dataset_labels[best_index]

        return SimilarityMatrix(
            user_labels=user_names,
            dataset_labels=list(dataset_labels),
            scores=scores,
            assignments=assignments,
        )

    def _compute_lexical_matrix(
        self,
        requested_labels: list[RequestedLabel],
        dataset_labels: list[str],
        user_names: list[str],
    ) -> SimilarityMatrix:
        scores: list[list[float]] = []
        assignments: dict[str, str] = {}

        for request, user_name in zip(requested_labels, user_names):
            row: list[float] = []
            best_score = -1.0
            best_label = dataset_labels[0]
            phrases = [request.name, *request.synonyms]
            for dataset_label in dataset_labels:
                phrase_scores = [
                    SequenceMatcher(None, phrase.lower(), dataset_label.lower()).ratio()
                    for phrase in phrases
                ]
                cell_score = max(phrase_scores)
                row.append(cell_score)
                if cell_score > best_score:
                    best_score = cell_score
                    best_label = dataset_label
            scores.append(row)
            assignments[user_name] = best_label

        return SimilarityMatrix(
            user_labels=user_names,
            dataset_labels=list(dataset_labels),
            scores=scores,
            assignments=assignments,
        )

    def _resolve_lexical(
        self,
        requested_labels: list[RequestedLabel],
        dataset_labels: list[str],
    ) -> list[ResolvedLabel]:
        resolved: list[ResolvedLabel] = []
        for request in requested_labels:
            best_label = ""
            best_score = -1.0
            phrases = [request.name, *request.synonyms]
            for phrase in phrases:
                for dataset_label in dataset_labels:
                    score = SequenceMatcher(None, phrase.lower(), dataset_label.lower()).ratio()
                    if score > best_score:
                        best_score = score
                        best_label = dataset_label
            resolved.append(
                ResolvedLabel(
                    requested_label=request.name,
                    dataset_label=best_label or dataset_labels[0],
                    score=float(best_score),
                    method="lexical",
                )
            )
        return resolved

    def _resolve_substring(
        self,
        requested_labels: list[RequestedLabel],
        dataset_labels: list[str],
    ) -> list[ResolvedLabel]:
        """Last-resort fallback: case-insensitive substring containment matching."""
        resolved: list[ResolvedLabel] = []
        for request in requested_labels:
            best_label = ""
            best_score = 0.0
            phrases = [request.name.lower(), *(s.lower() for s in request.synonyms)]
            for phrase in phrases:
                for dataset_label in dataset_labels:
                    dl = dataset_label.lower()
                    if phrase in dl or dl in phrase:
                        # Score based on overlap ratio, but boosted if one is entirely within the other
                        overlap_len = min(len(phrase), len(dl))
                        max_len = max(len(phrase), len(dl), 1)
                        # Boost containment matches to be higher than weak lexical matches
                        score = 0.5 + 0.5 * (overlap_len / max_len)
                        if score > best_score:
                            best_score = score
                            best_label = dataset_label
            if best_label:
                resolved.append(
                    ResolvedLabel(
                        requested_label=request.name,
                        dataset_label=best_label,
                        score=float(best_score),
                        method="substring",
                    )
                )
            else:
                # No substring match at all — yield the first dataset label with zero score
                resolved.append(
                    ResolvedLabel(
                        requested_label=request.name,
                        dataset_label=dataset_labels[0],
                        score=0.0,
                        method="substring_fail",
                    )
                )
        return resolved

    def _load_clip(self) -> _ClipCache:
        if self._cache is not None:
            return self._cache
        from transformers import AutoTokenizer, CLIPModel

        tokenizer = AutoTokenizer.from_pretrained(self.clip_model_name)
        model = CLIPModel.from_pretrained(self.clip_model_name)
        model.eval()
        self._cache = _ClipCache(tokenizer=tokenizer, model=model)
        return self._cache

    def _embed_texts(self, texts: list[str]) -> torch.Tensor:
        cache = self._load_clip()
        inputs = cache.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            features = cache.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(features, dim=-1)

    def _resolve_semantic(
        self,
        requested_labels: list[RequestedLabel],
        dataset_labels: list[str],
    ) -> list[ResolvedLabel]:
        dataset_embeddings = self._embed_texts(dataset_labels)
        resolved: list[ResolvedLabel] = []

        for request in requested_labels:
            phrases = [request.name, *request.synonyms]
            request_embedding = self._embed_texts(phrases).mean(dim=0, keepdim=True)
            request_embedding = torch.nn.functional.normalize(request_embedding, dim=-1)
            similarities = (request_embedding @ dataset_embeddings.T).squeeze(0)
            best_index = int(torch.argmax(similarities).item())
            best_score = float(similarities[best_index].item())
            if best_score < self.threshold:
                lexical = self._resolve_lexical([request], dataset_labels)[0]
                resolved.append(lexical)
                continue
            resolved.append(
                ResolvedLabel(
                    requested_label=request.name,
                    dataset_label=dataset_labels[best_index],
                    score=best_score,
                    method="clip",
                )
            )

        return resolved

