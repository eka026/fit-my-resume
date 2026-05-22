import pandas as pd

from src.create_teacher_pairs import REQUIRED_COLUMNS, create_pairs_for_split, parse_splits


def _resumes():
    return pd.DataFrame(
        [
            {
                "resume_id": "resume_1",
                "category": "Data Science",
                "resume_text": "Python machine learning pandas SQL model evaluation",
            },
            {
                "resume_id": "resume_2",
                "category": "HR",
                "resume_text": "Recruiting onboarding employee relations payroll benefits",
            },
        ]
    )


def _jobs():
    return pd.DataFrame(
        [
            {
                "job_id": "job_1",
                "position_title": "Machine Learning Engineer",
                "job_description": "Build Python machine learning models with SQL and evaluation",
            },
            {
                "job_id": "job_2",
                "position_title": "Human Resources Generalist",
                "job_description": "Support recruiting onboarding payroll and employee relations",
            },
            {
                "job_id": "job_3",
                "position_title": "Warehouse Associate",
                "job_description": "Package inventory shipments and operate warehouse equipment",
            },
            {
                "job_id": "job_4",
                "position_title": "Frontend Developer",
                "job_description": "Build user interfaces with React JavaScript CSS and accessibility",
            },
        ]
    )


def test_create_pairs_has_required_columns_and_strategies():
    pairs = create_pairs_for_split(
        resumes=_resumes(),
        jobs=_jobs(),
        split="train",
        pairs_per_resume=3,
        seed=42,
    )

    assert list(pairs.columns) == REQUIRED_COLUMNS
    assert set(pairs["pairing_strategy"]) == {
        "strong_hybrid",
        "medium_tfidf",
        "weak_random",
    }


def test_create_pairs_does_not_duplicate_resume_job_pairs():
    pairs = create_pairs_for_split(
        resumes=_resumes(),
        jobs=_jobs(),
        split="train",
        pairs_per_resume=3,
        seed=42,
    )

    duplicate_count = pairs.duplicated(subset=["resume_id", "job_id"]).sum()
    assert duplicate_count == 0


def test_create_pairs_is_deterministic_for_same_seed():
    first = create_pairs_for_split(
        resumes=_resumes(),
        jobs=_jobs(),
        split="train",
        pairs_per_resume=3,
        seed=42,
    )
    second = create_pairs_for_split(
        resumes=_resumes(),
        jobs=_jobs(),
        split="train",
        pairs_per_resume=3,
        seed=42,
    )

    pd.testing.assert_frame_equal(first, second)


def test_parse_splits_uses_defaults_only_when_no_split_is_provided():
    assert parse_splits(None) == ["train", "validation"]
    assert parse_splits(["train"]) == ["train"]
    assert parse_splits(["train,validation"]) == ["train", "validation"]
