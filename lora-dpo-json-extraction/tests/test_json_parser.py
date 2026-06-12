from lora_dpo_json_extraction.utils import extract_first_json_object


def test_extract_first_json_object() -> None:
    text = 'prefix {"intent":"bug_report","priority":"high"} suffix'
    parsed = extract_first_json_object(text)
    assert parsed == '{"intent":"bug_report","priority":"high"}'


def test_extract_none_on_no_object() -> None:
    assert extract_first_json_object("no json here") is None
