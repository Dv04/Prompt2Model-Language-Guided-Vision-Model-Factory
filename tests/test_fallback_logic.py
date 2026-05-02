"""Week 6 — Dev: Tests for fallback label resolution logic."""

import logging
from prompt2model.config import RequestedLabel
from prompt2model.label_resolution import LabelResolver

def test_substring_fallback_logic() -> None:
    # Test that substring matching handles cases where lexical matching is weak
    resolver = LabelResolver(threshold=0.7, enable_clip=False)
    
    # "car" vs ["racecar", "dog"]
    # lexical "car" vs "racecar" might be relatively low depending on implementation
    # but substring "car" in "racecar" is a strong signal.
    
    resolved = resolver.resolve(
        requested_labels=[RequestedLabel(name="car")],
        dataset_labels=["racecar", "dog"]
    )
    
    assert resolved[0].dataset_label == "racecar"
    assert resolved[0].method == "substring"

def test_substring_overlap_scoring() -> None:
    resolver = LabelResolver(enable_clip=False)
    
    # "moto" should match "motorcycle" better than "boat"
    resolved = resolver._resolve_substring(
        requested_labels=[RequestedLabel(name="moto")],
        dataset_labels=["motorcycle", "boat"]
    )
    assert resolved[0].dataset_label == "motorcycle"
    assert resolved[0].score > 0

def test_substring_fail_fallback() -> None:
    resolver = LabelResolver(threshold=0.1, enable_clip=False)
    
    # No match at all
    resolved = resolver.resolve(
        requested_labels=[RequestedLabel(name="xyzabc")],
        dataset_labels=["cat", "dog"]
    )
    # Lexical or substring_fail? 
    # If lexical is very low, it tries substring. 
    # If substring also fails, it returns substring_fail.
    assert resolved[0].method in ("lexical", "substring_fail")

def test_case_insensitive_substring() -> None:
    resolver = LabelResolver(enable_clip=False)
    resolved = resolver._resolve_substring(
        requested_labels=[RequestedLabel(name="CAT")],
        dataset_labels=["Category", "Dog"]
    )
    assert resolved[0].dataset_label == "Category"
    assert resolved[0].method == "substring"
