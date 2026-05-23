import argparse
import random
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED_COLUMNS = [
    "pair_id",
    "split",
    "resume_id",
    "job_id",
    "pairing_strategy",
    "similarity_score",
    "resume_category",
    "job_position_title",
    "resume_text",
    "job_description",
]

CATEGORY_KEYWORDS = {
    "data science": ["data", "machine learning", "ml", "python", "analytics", "scientist"],
    "hr": ["hr", "human resources", "recruiting", "recruiter", "onboarding", "payroll"],
    "information technology": ["it", "information technology", "systems", "network", "support"],
    "engineering": ["engineer", "engineering", "mechanical", "electrical", "design"],
    "accountant": ["accounting", "accountant", "finance", "financial", "tax", "audit"],
    "sales": ["sales", "account executive", "business development", "customer"],
    "healthcare": ["healthcare", "clinical", "patient", "medical", "nurse"],
}


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def compute_similarity(resume_text: str, job_descriptions: pd.Series) -> list[float]:
    documents = [resume_text] + job_descriptions.fillna("").astype(str).tolist()
    matrix = TfidfVectorizer(stop_words="english").fit_transform(documents)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    return [float(score) for score in scores]


def compute_similarity_matrix(resumes: pd.Series, jobs: pd.Series):
    resume_texts = resumes.fillna("").astype(str).tolist()
    job_texts = jobs.fillna("").astype(str).tolist()
    documents = resume_texts + job_texts
    matrix = TfidfVectorizer(stop_words="english").fit_transform(documents)
    resume_matrix = matrix[: len(resume_texts)]
    job_matrix = matrix[len(resume_texts) :]
    return cosine_similarity(resume_matrix, job_matrix)


def category_keywords(category: str) -> list[str]:
    normalized = normalize_text(category)
    keywords = [normalized] if normalized else []
    keywords.extend(CATEGORY_KEYWORDS.get(normalized, []))
    return [keyword for keyword in dict.fromkeys(keywords) if keyword]


def related_job_indices(category: str, jobs: pd.DataFrame) -> set[int]:
    keywords = category_keywords(category)
    related = set()
    for index, job in jobs.iterrows():
        haystack = normalize_text(
            f"{job.get('position_title', '')} {job.get('job_description', '')}"
        )
        if any(keyword in haystack for keyword in keywords):
            related.add(index)
    return related


def related_job_indices_by_category(
    categories: pd.Series,
    jobs: pd.DataFrame,
) -> dict[str, set[int]]:
    return {
        str(category): related_job_indices(str(category), jobs)
        for category in categories.fillna("").drop_duplicates()
    }


def choose_index(candidates: list[int], rng: random.Random, used: set[int]) -> int | None:
    available = [candidate for candidate in candidates if candidate not in used]
    if not available:
        return None
    return rng.choice(available)


def create_pairs_for_split(
    resumes: pd.DataFrame,
    jobs: pd.DataFrame,
    split: str,
    pairs_per_resume: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    rng = random.Random(seed)
    resumes = resumes.reset_index(drop=True).copy()
    jobs = jobs.reset_index(drop=True).copy()
    similarity_scores = compute_similarity_matrix(
        resumes["resume_text"],
        jobs["job_description"],
    )
    related_by_category = related_job_indices_by_category(resumes["category"], jobs)

    for resume_index, resume in resumes.iterrows():
        scores = similarity_scores[resume_index]
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        related = related_by_category.get(str(resume.get("category", "")), set())
        used: set[int] = set()

        top_count = max(1, min(10, len(ranked)))
        bottom_start = max(0, int(len(ranked) * 0.8))
        medium_start = max(0, int(len(ranked) * 0.4))
        medium_end = max(medium_start + 1, int(len(ranked) * 0.6))

        top_related = [index for index in ranked[:top_count] if index in related]
        weak_candidates = [index for index in ranked[bottom_start:] if index not in related]
        strategies = [
            ("strong_hybrid", top_related or ranked[:top_count]),
            ("medium_tfidf", ranked[medium_start:medium_end]),
            ("weak_random", weak_candidates or ranked[bottom_start:]),
        ]

        for strategy, candidates in strategies[:pairs_per_resume]:
            job_index = choose_index(candidates, rng, used)
            if job_index is None:
                continue
            used.add(job_index)
            job = jobs.iloc[job_index]
            pair_id = f"{split}_{resume['resume_id']}_{job['job_id']}_{strategy}"
            rows.append(
                {
                    "pair_id": pair_id,
                    "split": split,
                    "resume_id": resume["resume_id"],
                    "job_id": job["job_id"],
                    "pairing_strategy": strategy,
                    "similarity_score": round(float(scores[job_index]), 6),
                    "resume_category": resume.get("category", ""),
                    "job_position_title": job.get("position_title", ""),
                    "resume_text": resume["resume_text"],
                    "job_description": job["job_description"],
                }
            )

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def parse_splits(values: list[str] | None) -> list[str]:
    if values is None:
        return ["train", "validation"]

    splits = []
    for value in values:
        splits.extend(part.strip() for part in value.split(",") if part.strip())
    return splits or ["train", "validation"]


def write_pairs_for_split(
    processed_dir: Path,
    split: str,
    pairs_per_resume: int,
    max_resumes: int | None,
    seed: int,
) -> Path:
    resumes = pd.read_csv(processed_dir / f"resumes_{split}.csv")
    jobs = pd.read_csv(processed_dir / f"jobs_{split}.csv")
    if max_resumes is not None:
        resumes = resumes.head(max_resumes)

    pairs = create_pairs_for_split(
        resumes=resumes,
        jobs=jobs,
        split=split,
        pairs_per_resume=pairs_per_resume,
        seed=seed,
    )
    output_path = processed_dir / f"teacher_pairs_{split}.csv"
    pairs.to_csv(output_path, index=False)
    return output_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create teacher-labeling resume-job pairs.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", action="append")
    parser.add_argument("--pairs-per-resume", type=int, default=3)
    parser.add_argument("--max-resumes", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.pairs_per_resume < 1 or args.pairs_per_resume > 3:
        raise ValueError("--pairs-per-resume must be between 1 and 3")
    if args.max_resumes is not None and args.max_resumes < 1:
        raise ValueError("--max-resumes must be at least 1")

    for split in parse_splits(args.split):
        output_path = write_pairs_for_split(
            processed_dir=args.processed_dir,
            split=split,
            pairs_per_resume=args.pairs_per_resume,
            max_resumes=args.max_resumes,
            seed=args.seed,
        )
        print(f"wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
