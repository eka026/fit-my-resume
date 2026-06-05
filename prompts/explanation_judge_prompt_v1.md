# FitMyResume Explanation Judge Prompt v1

You are evaluating a resume-job matching system. Use only the provided resume, job description, teacher reference, and candidate model output. Do not reward claims that are not supported by the resume.

Return only valid JSON with this schema:

```json
{
  "correctness": 1,
  "specificity": 1,
  "matched_qualification_coverage": 1,
  "missing_qualification_coverage": 1,
  "fabrication_flag": false,
  "short_rationale": "One or two concise sentences."
}
```

Scoring scale for each numeric field:

- 1 = poor
- 2 = weak
- 3 = acceptable
- 4 = strong
- 5 = excellent

Rubric:

- `correctness`: The explanation accurately describes the resume-job fit and does not contradict the resume or job description.
- `specificity`: The explanation uses concrete job requirements and resume evidence rather than generic comments.
- `matched_qualification_coverage`: The explanation covers important qualifications that the resume satisfies.
- `missing_qualification_coverage`: The explanation covers important job requirements that are missing or weak in the resume.
- `fabrication_flag`: Set to true if the explanation claims the candidate has experience, skills, education, credentials, employers, or accomplishments not present in the resume.

Input:

RESUME:
{resume}

JOB_DESCRIPTION:
{job_description}

TEACHER_REFERENCE_JSON:
{teacher_reference_json}

CANDIDATE_MODEL_JSON:
{candidate_model_json}
