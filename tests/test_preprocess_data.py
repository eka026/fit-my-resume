import pandas as pd

from src.preprocess_data import clean_text, mask_resume_pii, split_dataframe


def test_clean_text_normalizes_whitespace_control_chars_and_punctuation():
    raw = "Hello\u0000   world\u2014with\u201ccurly\u201d quotes\n\nNext\tline"

    assert clean_text(raw) == 'Hello world-with"curly" quotes Next line'


def test_mask_resume_pii_replaces_common_contact_fields():
    raw = (
        "Jane Doe\n"
        "jane.doe@example.com\n"
        "+1 (555) 123-4567\n"
        "https://janedoe.dev\n"
        "123 Main Street, Boston, MA 02110\n"
        "Experienced analyst"
    )

    masked = mask_resume_pii(raw)

    assert "[EMAIL]" in masked
    assert "[PHONE]" in masked
    assert "[URL]" in masked
    assert "[ADDRESS]" in masked
    assert "jane.doe@example.com" not in masked
    assert "555" not in masked
    assert "janedoe.dev" not in masked
    assert "Experienced analyst" in masked


def test_split_dataframe_is_seeded_and_uses_stable_ids():
    df = pd.DataFrame(
        {
            "resume_id": [5, 1, 3, 2, 4],
            "resume_text": ["five", "one", "three", "two", "four"],
        }
    )

    first = split_dataframe(df, id_column="resume_id", seed=7)
    second = split_dataframe(df, id_column="resume_id", seed=7)

    assert [len(first["train"]), len(first["validation"]), len(first["test"])] == [4, 0, 1]
    assert first["train"]["resume_id"].tolist() == second["train"]["resume_id"].tolist()
    assert first["test"]["resume_id"].tolist() == second["test"]["resume_id"].tolist()
