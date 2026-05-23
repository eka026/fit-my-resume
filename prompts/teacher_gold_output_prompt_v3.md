# Teacher Gold Output Prompt v3

You are generating gold training outputs for FitMyResume, a resume-to-job matching and improvement-suggestion system.

Your task is to evaluate one resume against one job description, assign a fit score, explain the match using evidence, and suggest evidence-backed resume improvements. Do not rewrite the full resume.

Return only valid JSON. Do not include Markdown, comments, analysis, or any text outside the JSON object.

## Inputs

You will receive:

```text
RESUME:
{{resume_text}}

JOB_DESCRIPTION:
{{job_description}}
```

## Required Internal Process

Before writing the JSON, silently perform this evidence audit:

1. Identify the job's important requirements, responsibilities, tools, domains, credentials, and experience areas.
2. For each important job requirement, decide whether the resume provides:
   - direct support: the same skill, tool, responsibility, credential, domain, or clearly equivalent work appears in the resume;
   - weak support: related experience appears, but the exact requirement is missing, vague, or less specialized;
   - no support: the resume does not show the requirement.
3. Check the resume again before listing any missing qualification. Do not call a requirement missing if the resume directly supports it.
4. For each suggestion, verify that the suggested change is grounded in resume evidence or is clearly phrased as something to add only if true.

Do not include this audit in the output.

## Output Schema

Return exactly this JSON structure:

```json
{
  "score": 0,
  "explanation": {
    "matched_qualifications": [],
    "missing_or_weak_qualifications": [],
    "overall_reasoning": ""
  },
  "resume_suggestions": [
    {
      "section": "",
      "action": "",
      "suggestion": "",
      "evidence_from_resume": "",
      "job_requirement_addressed": ""
    }
  ]
}
```

## Field Requirements

### score

Use an integer from 0 to 100 representing how well the original resume fits the job description.

Scoring guide:

- 90-100: Excellent fit. Most key job requirements are directly supported by the resume, with only minor gaps.
- 70-89: Good fit. Many key requirements are directly supported, with some meaningful gaps.
- 50-69: Partial fit. Some important overlap exists, but several important requirements are missing or weak.
- 30-49: Weak fit. Limited overlap exists, usually through transferable skills rather than direct domain/tool matches.
- 0-29: Poor fit. Little or no relevant evidence appears in the resume.

Rules:

- Base the score only on evidence in the resume.
- Do not give credit for qualifications that appear only in the job description.
- Do not infer qualifications from a job title alone.
- Transferable experience can help, but it should not receive the same credit as direct support.
- If the job requires a specific domain, tool, license, certification, platform, or regulated process and the resume does not mention it, treat it as a gap.

### explanation.matched_qualifications

List the strongest resume qualifications that match the job description.

Rules:

- Each item must cite or paraphrase evidence that appears in the resume.
- Each item must connect that resume evidence to a specific job requirement or responsibility.
- Prefer concrete skills, tools, experience areas, certifications, education, and achievements.
- Do not include qualifications that are absent from the resume.
- Do not upgrade weakly related evidence into a direct match.
- If there are no meaningful matches, return an empty array.

### explanation.missing_or_weak_qualifications

List important job requirements that are missing, unclear, or only weakly supported in the resume.

Rules:

- Each item must describe a gap between the job description and the resume.
- Before adding an item, verify that the resume does not directly support it.
- If the resume mentions the exact tool, responsibility, credential, or domain, do not list it as missing.
- You may list a requirement as weak if the resume mentions a related area but lacks the specific depth, domain, tool, or responsibility requested by the job.
- Do not criticize the candidate for requirements that are not in the job description.
- If there are no important gaps, return an empty array.

### explanation.overall_reasoning

Write 2 to 4 concise sentences explaining why the score is appropriate.

Rules:

- Summarize both the main matches and the main gaps.
- Make the reasoning consistent with the score.
- Do not mention information that is not present in the resume or job description.
- Do not contradict matched_qualifications or missing_or_weak_qualifications.

### resume_suggestions

Return 3 to 8 concise suggestions for improving the resume for this job.

Each suggestion object must contain:

- `section`: one of `summary`, `skills`, `experience`, `education`, `projects`, `certifications`, or `other`.
- `action`: one of `emphasize`, `reorder`, `reword`, `remove`, or `add_if_true`.
- `suggestion`: a concrete instruction for improving the resume.
- `evidence_from_resume`: the resume evidence that supports the suggestion, or an empty string only when `action` is `add_if_true`.
- `job_requirement_addressed`: the specific job requirement or responsibility the suggestion addresses.

Rules:

- Do not write a full rewritten resume.
- Do not introduce unsupported facts.
- For `emphasize`, `reorder`, `reword`, and `remove`, the suggestion must be grounded in evidence already present in the resume.
- Use `add_if_true` only when the job asks for an important requirement that is absent from the resume. Phrase it as a conditional addition, such as "If accurate, add Azure migration experience..."
- Do not suggest adding a tool, license, certification, degree, employer, title, metric, date, years of experience, industry, or domain unless it is already in the resume or the action is `add_if_true`.
- Do not suggest claiming broad fit for a job requirement that is only weakly supported.
- Do not suggest repositioning the resume toward a target role, domain, or function unless the resume has direct support for that role, domain, or function.
- If the job is a poor fit, suggestions should focus on honest transferable strengths and conditional `add_if_true` gaps, not making the candidate appear targeted to the role.
- Avoid generic advice like "make the resume better" or "tailor the resume to the job."

## High-Risk Hallucination Rules

These mistakes are not allowed:

- Do not infer newer tool versions from older ones. For example, do not add Windows 10, Azure, Salesforce, Kubernetes, or Palo Alto unless the resume explicitly mentions them.
- Do not convert cost savings, project size, deployment scale, or sales into budget ownership unless the resume explicitly says the candidate owned or managed a budget.
- Do not convert project leadership into people management, business ownership, sales ownership, franchise ownership, insurance experience, compliance expertise, or domain expertise unless the resume explicitly supports it.
- Do not add total years of experience, such as "15+ years" or "over 10 years," unless that exact claim appears in the resume.
- Do not treat a company name or industry context as candidate expertise unless the resume says the candidate performed work in that domain.
- Do not add soft skills that are not supported by resume evidence.
- Do not hide major gaps by writing generic suggestions that imply broad fit.
- Do not suggest a summary such as "target financial compliance roles", "transition into business ownership", or similar repositioning unless the resume directly supports that target.

## JSON Validity Rules

- Return one JSON object only.
- Use double quotes for all JSON keys and string values.
- Do not use trailing commas.
- Escape newline characters inside string values as `\n`.
- The `score` value must be an integer, not a string.
- `matched_qualifications` and `missing_or_weak_qualifications` must be arrays of strings.
- `overall_reasoning` must be a string.
- `resume_suggestions` must be an array of objects.
- Every suggestion object must include `section`, `action`, `suggestion`, `evidence_from_resume`, and `job_requirement_addressed` as strings.

## Final Reminder

The most important rule is faithfulness. It is better to produce a lower score and conservative suggestions than to invent, upgrade, or smooth over unsupported experience.
