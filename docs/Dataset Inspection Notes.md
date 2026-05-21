# Dataset Inspection Notes

Date inspected: 2026-05-21

Raw dataset location:

```text
data/raw/kaggle/
```

Files inspected:

- `Resume.csv`
- `training_data.csv`

## Resume.csv

Row count: 2,484

Columns:

| Column | Type | Nulls | Unique values | Notes |
|---|---:|---:|---:|---|
| `ID` | integer | 0 | 2,484 | Stable resume identifier. Useful for traceability. |
| `Resume_str` | text | 0 | 2,482 | Plain-text resume content. Main resume text field. |
| `Resume_html` | text/html | 0 | 2,482 | HTML-formatted resume content. Optional unless formatting is needed. |
| `Category` | text | 0 | 24 | Resume category label, such as `HR`. |

Confirmed fields:

- Resume text: `Resume_str`
- Resume category label: `Category`
- Resume ID / metadata: `ID`
- Optional resume HTML: `Resume_html`

Notes:

- `ID` is unique for all rows.
- `Resume_str` and `Resume_html` each contain 2 duplicate values.

## training_data.csv

Row count: 853

Columns:

| Column | Type | Nulls | Unique values | Notes |
|---|---:|---:|---:|---|
| `company_name` | text | 0 | 853 | Company metadata. |
| `job_description` | text | 0 | 853 | Main job description text field. |
| `position_title` | text | 0 | 725 | Job title metadata. May be useful for weak pairing or analysis. |
| `description_length` | integer | 0 | 803 | Numeric metadata for job description length. |
| `model_response` | text/json-like | 0 | 844 | Structured generated job analysis, not a resume-job match label. |

Confirmed fields:

- Job description text: `job_description`
- Job metadata: `company_name`, `position_title`, `description_length`
- Generated job analysis: `model_response`

## Match Labels and Joins

No direct resume-job match labels were found.

No shared join key exists between `Resume.csv` and `training_data.csv`.

This means the raw dataset contains separate resume records and job description records, but it does not directly provide labeled resume-job pairs. For the matching task, the project will need to create pairs through a separate strategy, such as weak category/title matching, synthetic pairing, teacher-generated scoring, or manual labeling.
