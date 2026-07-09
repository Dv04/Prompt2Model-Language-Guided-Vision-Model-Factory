"""Weeks 9-10 - Dev: Tests for the Language-to-Config ablation suite."""

from prompt2model.ablation import AblationSuite


def test_ablation_accuracy() -> None:
    suite = AblationSuite()
    results = suite.run()
    
    accuracy = suite.compute_accuracy(results)
    
    # Requirement: parsing accuracy should be > 90% (27/30 pass)
    # Our regex-based parser is currently very consistent with the rubric.
    assert len(results) == 30
    assert accuracy >= 0.90
    
    # Print failed cases if any for debugging
    for result in results:
        if not result.passed:
            print(f"FAILED: {result.case_name} - {result.details}")
