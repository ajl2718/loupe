from .BaseInspector import BaseInspector

from .text_processing import calculate_overall_counts, calculate_cell_counts
from .pii_utils import redact_texts, create_matcher, pii_from_texts
from .text_processing import process_text, process_df
from .text_extraction import get_text_from_worddoc, get_text_from_pdf

import pandas as pd
import numpy as np
from docx import Document
from pdfplumber import PDF

import spacy

default_pii_types = [
    "PERSON_NAME",
    "EMAIL",
    "URL",
    "PHONE_AU",
    "MEDICARE_NUMBER",
    "TAX_FILE_NUMBER",
    "CAR_REG_VIC",
]


class TextInspector(BaseInspector):
    """
    TextInspector: inspect and redact strs and list of strs, PDF files, Word Documents,
    anything involving text.
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.matcher = create_matcher(self.nlp, "./loupe/patterns.jsonl")

    def inspect(self, document, pii_types=default_pii_types):
        # check the data type of document
        inspect_document = self._inspect_type(document)
        results = inspect_document(document, pii_types)
        return results

    def redact(self, document, pii_types=default_pii_types):
        # check the data type of document
        redact_document = self._redact_type(document)
        results = redact_document(document, pii_types)
        return results

    def _inspect_type(self, document):
        """
        Determine what type of object the document is. Options are:
        - pdfplumber pdf object
        - docx doc object
        - list of strs
        - str

        Return: the appropriate inspect method
        """
        if isinstance(document, str) or isinstance(document, list):
            return self.inspect_list
        if isinstance(document, np.ndarray) or isinstance(document, pd.Series):
            return self.inspect_list
        if isinstance(document, pd.DataFrame):
            return self.inspect_dataframe
        if isinstance(document, PDF):
            return self.inspect_pdf
        if isinstance(document, Document):
            return self.inspect_doc

    def _redact_type(self, document):
        """
        Determine what type of object the document is. Options are:
        - pdfplumber pdf object
        - docx doc object
        - list of strs
        - str

        Return: the appropriate redact method
        """
        if isinstance(document, str) or isinstance(document, list):
            return self.redact_list
        if isinstance(document, np.ndarray) or isinstance(document, pd.Series):
            return self.redact_list
        if isinstance(document, pd.DataFrame):
            return self.redact_dataframe
        if isinstance(document, PDF):
            return self.redact_pdf
        if isinstance(document, Document):
            return self.redact_doc

    def inspect_list(self, texts, pii_types=default_pii_types):
        """
        Given a list of strs check for PII

        Args:
        texts: list with the data we are inspecting

        Return:
        df: dictionary with the counts of each pii type
        """

        if isinstance(texts, str):
            texts = [texts]

        texts = [process_text(text) for text in texts]
        pii_results = pii_from_texts(texts, self.matcher, self.nlp, pii_types)
        return pii_results

    def redact_list(self, texts, pii_types=default_pii_types):
        """
        Given a list of strs, remove all ocurrences of specified PII

        Args:
        texts: list of strs
        pii_types: the specified PII types to redact

        Return:
        results: list of strs with each ocurrence of PII replaced by '[REDACTED]'
        """

        if isinstance(texts, str):
            texts = [texts]

        texts = [process_text(text) for text in texts]
        texts_redacted = redact_texts(texts, self.matcher, self.nlp, pii_types)
        return texts_redacted

    def inspect_doc(self, worddoc, pii_types=default_pii_types):
        """
        Given a Microsoft Word (.docx) document check for PII

        Args:
        worddoc: the docx object representing the word document
        pii_types: PII types to check for

        Return:
        df: dictionary with the counts off each pii type
        """
        texts = get_text_from_worddoc(worddoc)
        texts = [process_text(text) for text in texts]

        # get the dict of pii_counts for each pii type for each text
        pii_results = self.inspect_list(texts, pii_types)

        results = {pii_type: [] for pii_type in pii_types}

        for pii_result in pii_results:
            for pii_item in pii_result:
                results[pii_item] += pii_result[pii_item]

        return results

    # This currently is not working for redaction
    def redact_doc(self, worddoc, pii_types=default_pii_types):
        """
        Given a Microsoft Word (.docx) document check for PII

        Args:
        worddoc: the docx object representing the word document
        pii_types: PII types to check for

        Return:
        df: dictionary with the counts off each pii type
        """
        texts = get_text_from_worddoc(worddoc)
        texts = [process_text(text) for text in texts]

        # get the dict of pii_counts for each pii type for each text
        pii_results = self.inspect_list(texts, pii_types)
        return pii_results

    def inspect_pdf(self, pdf, pii_types=default_pii_types):
        """
        Given a PDF check for PII

        Args:
        pdf: pdfplumber pdf object
        pii_types: PII types to check for

        Return:
        df: dictionary with the counts off each pii type
        """
        # extract and process text from PDF
        texts = get_text_from_pdf(pdf)
        if texts == None:
            texts = [""]
        else:
            texts = [process_text(text) for text in texts]

        # get the dict of pii_counts for each pii type for each text
        # convert to a single dict for the entire document
        output = dict({pii_type: [] for pii_type in pii_types})
        pii_results = self.inspect_list(texts, pii_types)
        for pii_result in pii_results:
            for pii_type in pii_types:
                output[pii_type] += pii_result[pii_type]
        return output

    # this currently does not work
    def redact_pdf(self, pdf, pii_types=default_pii_types):
        """
        Given a PDF check for PII

        Args:
        pdf: pdfplumber pdf object
        pii_types: PII types to check for

        Return:
        df: dictionary with the counts off each pii type
        """
        # extract and process text from PDF
        texts = get_text_from_pdf(pdf)
        if texts == None:
            texts = [""]
        else:
            texts = [process_text(text) for text in texts]

        # get the dict of pii_counts for each pii type for each text
        pii_results = self.inspect_list(texts, pii_types)
        return pii_results

    # currently broken
    def inspect_dataframe(self, df, pii_types=default_pii_types, sample_size=1024):
        """
        Given a dataframe, scan for PII

        Args:
        df: pandas.DataFrame with the data we are inspecting

        Return:
        results: dictionary with the counts of each pii type
        """
        if sample_size:
            df = df.sample(sample_size)

        df = process_df(df)

        self.pii_results = inspect_for_pii(df, self.matcher, self.nlp)
        return results

    # currently broken
    def redact_dataframe(self, df, pii_types=default_pii_types):
        """
        Given a dataframe remove all ocurrences of the the specified PII types

        Args:
        df: pandas.DataFrame with the data we want to redact

        Return:
        df: pandas.DataFrame with the pii removed
        """
        # first check where the PII is
        # go through and remove each instance of the PII with '[REDACTED]'
        return 0
