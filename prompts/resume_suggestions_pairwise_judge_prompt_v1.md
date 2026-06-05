# FitMyResume Resume Suggestions Pairwise Judge Prompt v1

You are comparing two sets of resume suggestions for the same resume and job description. These are suggestions only, not full rewritten resumes. Use only the provided resume and job description. Do not reward suggestions that tell the candidate to add experience, skills, education, credentials, employers, or accomplishments unless the suggestion clearly says to add it only if true.

Return only valid JSON with this schema:

```json
{
  "preferred_system": "A",
  "relevance_a": 1,
  "relevance_b": 1,
  "faithfulness_a": 1,
  "faithfulness_b": 1,
  "usefulness_a": 1,
  "usefulness_b": 1,
  "fabrication_flag_a": false,
  "fabrication_flag_b": false,
  "short_rationale": "One or two concise sentences."
}
```

Use `"A"`, `"B"`, or `"tie"` for `preferred_system`.

Scoring scale for each numeric field:

- 1 = poor
- 2 = weak
- 3 = acceptable
- 4 = strong
- 5 = excellent

Rubric:

- `relevance`: Suggestions target the job's important requirements.
- `faithfulness`: Suggestions are grounded in the original resume and avoid unsupported claims.
- `usefulness`: Suggestions are concrete enough for a candidate to apply.
- `fabrication_flag_*`: Set to true if the suggestions encourage unsupported additions without a clear "if true" condition.

Input:

RESUME:
{resume}

JOB_DESCRIPTION:
{job_description}

SYSTEM_A_SUGGESTIONS_JSON:
{system_a_suggestions_json}

SYSTEM_B_SUGGESTIONS_JSON:
{system_b_suggestions_json}
