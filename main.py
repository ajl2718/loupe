from typing import Optional
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from deid import TextInspector
from io import StringIO, BytesIO
from docx import Document
import pdfplumber
import pandas as pd


def time_pipe(texts, nlp):
    t1 = time()
    for doc in nlp.pipe(texts):
        a = 1
    t2 = time()
    return t2 - t1


class Item(BaseModel):
    input_text: list = None
    pii_types: list = [
        "PERSON_NAME",
        "PHONE_AU",
        "URL",
        "EMAIL",
        "POSTCODE",
    ]  # default PII types
    output: str = "overall_counts"


app = FastAPI()
inspector = TextInspector()

# post request with a list of strings
@app.post("/inspect_list")
async def check_list(item: Item):
    item_dict = item.dict()
    texts = item_dict["input_text"]
    pii_types = item_dict["pii_types"]
    output = item_dict["output"]
    if output == "PII_ITEMS":
        results = inspector.inspect_list(texts, pii_types)
    else:
        results = inspector.inspect_list(texts, pii_types)
    return results


@app.post("/redact_list")
async def inspect_list(item: Item):
    item_dict = item.dict()
    texts = item_dict["input_text"]
    pii_types = item_dict["pii_types"]
    output = item_dict["output"]
    if output == "PII_ITEMS":
        results = inspector.redact_list(texts, pii_types)
    else:
        results = inspector.redact_list(texts, pii_types)
    return results


# post request with a file
@app.post("/inspect_file")
async def inspect_file(file: UploadFile = File(...)):
    pii_types = [
        "PERSON_NAME",
        "PHONE_AU",
        "URL",
        "EMAIL",
        "MEDICARE_NUMBER",
        "TAX_FILE_NUMBER",
        "DATE",
        "DATE",
        "CREDIT_CARD_NUMBER",
        "CAR_REG_VIC",
        "ORG",
        "GPE",
        "MONEY",
        "AUSTRALIAN_BUSINESS_NUMBER",
    ]

    contents = await file.read()
    content_type = file.content_type
    print(content_type)

    # if it is a word docx doc... (does it work with .doc files?)
    if (
        content_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        doc = Document(BytesIO(contents))
        results = inspector.inspect_doc(doc, pii_types)
    elif content_type == "application/pdf":
        pdf = pdfplumber.open(BytesIO(contents))
        results = inspector.inspect_pdf(pdf, pii_types)
    else:
        s = contents.decode("utf-8")
        data = StringIO(s)
        df = pd.read_csv(data)
        results = inspector.inspect_dataframe(df)
    return results


# This currently does not work
@app.post("/redact_file")
async def redact_file(file: UploadFile = File(...)):
    pii_types = [
        "PERSON_NAME",
        "PHONE_AU",
        "URL",
        "EMAIL",
        "MEDICARE_NUMBER",
        "TAX_FILE_NUMBER",
        "DATE",
        "DATE",
        "CREDIT_CARD_NUMBER",
        "CAR_REG_VIC",
    ]

    contents = await file.read()
    content_type = file.content_type
    print(content_type)

    # if it is a word docx doc... (does it work with .doc files?)
    if (
        content_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        doc = Document(BytesIO(contents))
        counts = inspector.inspect_doc(doc, pii_types)
    elif content_type == "application/pdf":
        pdf = pdfplumber.open(BytesIO(contents))
        counts = inspector.inspect_pdf(pdf, pii_types)
    else:
        s = contents.decode("utf-8")
        data = StringIO(s)
        df = pd.read_csv(data)
        counts = inspector.inspect_dataframe(df)
    return {key: str(value) for key, value in counts.items()}
