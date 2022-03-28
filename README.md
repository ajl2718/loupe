# Loupe - Automated detection of Personally Identifiable Information

Detect and redact Personally Identifiable Information (PII).

## PII Types
The table below summarises the types of PII that can currently be detected by PrivateEye:

| PII Type | Label | Method |
| -------- | ----- | ------ |
| Person Name | PERSON_NAME | NLP |
| Place Name | GPE | NLP |
| Postcode | POSTCODE | Rule-based | 
| Email Address | EMAIL | Rule-based |
| URL | URL | Rule-based |
| Phone Number | PHONE_AU | Rule-based |
| Date | DATE | NLP |
| Medicare Number | MEDICARE_NUMBER | Rule-based |
| Tax File Number | TAX_FILE_NUMBER | Rule-based |
| Victorian Car Registration number | VIC_CAR_REGO | Rule-based |
| Credit card number | CREDIT_CARD_NUMBER | Rule-based |
| Australian Business Number | AUSTRALIAN_BUSINESS_NUMBER | Rule-based |
| Nationality, Religion and Political affiliation | NORP | NLP |
| Language | LANGUAGE | NLP |

Loupe currently processes the following data
- Pandas Dataframes
- Python Lists
- Microsoft Word (.docx) Documents
- Machine-generated PDF files (not scanned images yet)

## Requirements
- Poetry
- Python 3.6+
- spaCy's en_core_web_lg language model

## Installation
Install poetry
```
pip install poetry
poetry install
```

Once that has installed you'll need to load in the en_core_web_lg language model:

```
poetry shell
python -m spacy download en_core_web_lg
```

## Usage
Create an instance of a Inspector object (at the moment there is only the TextInspector):
```
from loupe import TextInspector
spec = TextInspector() 
```

To inspect a list of strings named inputs:
```
pii = spec.inspect(inputs)
```

To redact PII elements:
```
output_redacted = spec.redact(inputs)
```

The API can be run locally using (once in the poetry shell):

```
uvicorn main:app --reload
```

## How it works
PrivateEye uses spaCy's (https://www.spacy.io) Rule-based Matcher along with its pretrained Named Entity Recognition models to detect different types of PII. 