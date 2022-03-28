def get_text_from_pdf(pdf):
    """
    Given a pdfplumber PDF object, extract all the text and return a list of strs

    Args:
    pdf: pdfplumber pdf object

    Return:
    output: list of strs with a str corresponding to each page of the PDF
    """
    output = []
    for page in pdf.pages:
        output.append(page.extract_text())

    return output


def get_table_text(table):
    """
    Given a docx Table object, extract all the text from it

    Args:
    table: docx Table object

    Return:
    table_texts: list of strs containing the text in table
    """
    table_texts = []

    for row in table.rows:
        for cell in row.cells:
            # When a newline appears, split into separate texts as we assume these are separate
            # bits of text
            table_texts += cell.text.split("\n")

    return table_texts


def get_text_from_worddoc(doc):
    """
    Extract all the text (from paragraphs, tables, footnotes, headers, titles...)
    from a Word document. Return a list of strings.

    Args:
    doc: a docx Document object

    Return:
    doc_texts: list of strs with the text from worddoc

    """
    # get all the text from paragraphs
    paragraphs = [paragraph.text for paragraph in doc.paragraphs]

    # get all the text from table cells
    cells = [text for table in doc.tables for text in get_table_text(table)]

    # repeat these for the headers and footers
    headers = [section.header for section in doc.sections]
    footers = [section.footer for section in doc.sections]

    header_paragraphs = [
        paragraph.text for header in headers for paragraph in header.paragraphs
    ]
    footer_paragraphs = [
        paragraph.text for footer in footers for paragraph in footer.paragraphs
    ]

    header_cells = [
        text
        for header in headers
        for table in header.tables
        for text in get_table_text(table)
    ]
    footer_cells = [
        text
        for footer in footers
        for table in footer.tables
        for text in get_table_text(table)
    ]

    return (
        paragraphs
        + header_paragraphs
        + footer_paragraphs
        + cells
        + header_cells
        + footer_cells
    )
