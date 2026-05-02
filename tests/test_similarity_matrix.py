"""Week 5 — Dev: Tests for the cosine similarity matrix computation."""

from prompt2model.config import RequestedLabel
from prompt2model.label_resolution import LabelResolver, SimilarityMatrix


def test_lexical_matrix_dimensions() -> None:
    resolver = LabelResolver(enable_clip=False)
    matrix = resolver.compute_similarity_matrix(
        requested_labels=[
            RequestedLabel(name="motorcycle", synonyms=["two wheeler"]),
            RequestedLabel(name="car"),
        ],
        dataset_labels=["cat", "dog", "motorbike"],
    )
    assert isinstance(matrix, SimilarityMatrix)
    assert len(matrix.user_labels) == 2
    assert len(matrix.dataset_labels) == 3
    assert len(matrix.scores) == 2
    assert all(len(row) == 3 for row in matrix.scores)


def test_lexical_matrix_assignment_correctness() -> None:
    resolver = LabelResolver(enable_clip=False)
    matrix = resolver.compute_similarity_matrix(
        requested_labels=[
            RequestedLabel(name="motorcycle", synonyms=["two wheeler", "motorbike"]),
        ],
        dataset_labels=["cat", "dog", "motorbike"],
    )
    assert matrix.assignments["motorcycle"] == "motorbike"
    # Score for motorbike column should be highest
    motorbike_idx = matrix.dataset_labels.index("motorbike")
    assert matrix.scores[0][motorbike_idx] == max(matrix.scores[0])


def test_lexical_matrix_to_dict() -> None:
    resolver = LabelResolver(enable_clip=False)
    matrix = resolver.compute_similarity_matrix(
        requested_labels=[RequestedLabel(name="airplane")],
        dataset_labels=["plane", "jet"],
    )
    d = matrix.to_dict()
    assert "user_labels" in d
    assert "dataset_labels" in d
    assert "scores" in d
    assert "assignments" in d
    assert d["user_labels"] == ["airplane"]


def test_resolve_with_matrix_returns_both() -> None:
    resolver = LabelResolver(enable_clip=False)
    result = resolver.resolve_with_matrix(
        requested_labels=[
            RequestedLabel(name="motorcycle", synonyms=["motorbike"]),
        ],
        dataset_labels=["cat", "dog", "motorbike"],
    )
    assert len(result.resolved_labels) == 1
    assert result.resolved_labels[0].dataset_label == "motorbike"
    assert result.similarity_matrix is not None
    assert result.similarity_matrix.assignments["motorcycle"] == "motorbike"


def test_empty_dataset_labels_matrix() -> None:
    resolver = LabelResolver(enable_clip=False)
    matrix = resolver.compute_similarity_matrix(
        requested_labels=[RequestedLabel(name="cat")],
        dataset_labels=[],
    )
    assert matrix.scores == []
    assert matrix.assignments["cat"] == "cat"
