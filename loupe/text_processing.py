def process_text(text):
    """
    Process str to remove certain characters so that spacy can work with it
    """
    # remove the apostrophes, new lines, line breaks
    if text == None:
        return ""
    else:
        text = (
            text.replace("'", "")
            .replace("\n", " ")
            .replace("\r", "")
            .replace("\b", " ")
        )
        return text


def process_df(df):
    """
    Given a data frame, fill nan values with 0 or '' depending on datatype
    Ensures that column from dataframe can be parsed by pii detection functions
    """
    dtypes = df.dtypes

    for column in dtypes.index:
        if dtypes[column] == "O":
            df[column] = df[column].astype(str).fillna("")
        else:
            df[column] = df[column].fillna(0)

    return df


def pii_summary(pii):
    """
    Produce summary of the counts of each of the PII types given a dict
    that describes the PII for each of the entries in a list of strs
    """
    summary_counts = dict()

    for row in pii.keys():
        out = pii[row]
        counts = {key: len(value) for key, value in out.items()}
        summary_counts[row] = counts

    return summary_counts


def pii_profile(df_input, matcher):
    """
    Produce a nicely formatted summary dataframe,
    showing the types of PII in each of the columns which have PII
    """
    columns = df_input.columns

    # only the text columns
    text_columns = [column for column in columns if df_input[column].dtype == "O"]

    df_pii_profile = pd.DataFrame()

    for column in text_columns:
        print(column)
        input_list = df_input[column].values
        PII = pii_from_texts(input_list, matcher)
        summary_counts = pd.DataFrame(pii_summary(PII)).T.sum()
        df_pii_profile[column] = summary_counts

    return df_pii_profile


def calculate_overall_counts(pii_results):
    """
    Given a dict describing the pii outputs from a dataframe, calculate the overall
    counts of each PII type.
    """
    columns = pii_results.keys()
    df_pii_summary = pd.DataFrame()

    for column in columns:
        PII = pii_results[column]
        summary_counts = pd.DataFrame(pii_summary(PII)).T.sum()
        df_pii_summary[column] = summary_counts

    return df_pii_summary


def calculate_overall_counts(pii_results):
    """
    Given a dict describing the pii outputs from a dataframe, calculate the overall
    values of each PII type.
    """
    columns = pii_results.keys()
    dfs = []

    for column in columns:
        PII = pii_results[column]
        column_df = pd.DataFrame(PII).apply(lambda row: row.sum(), axis=1)
        dfs.append(column_df)

    # sum up all the dataframes
    df_final = dfs[0]
    for n in range(1, len(dfs)):
        df_final += dfs[n]

    # remove the duplicate values
    df_final = df_final.apply(lambda l: list(set(l)))

    return df_final


def calculate_cell_counts(pii_results):
    """
    Given a dict describing the pii outputs from a dataframe, calculate the
    counts of each PII type within each cell.
    """
    columns = pii_results.keys()
    df_pii_summary = pd.DataFrame()

    for column in columns:
        PII = pii_results[column]
        summary_counts = (
            pd.DataFrame(pii_summary(PII)).applymap(lambda cell: len(cell)).to_dict()
        )
        df_pii_summary[column] = summary_counts

    return df_pii_summary


def inspect_for_pii(df_input, matcher, nlp):
    columns = df_input.columns

    # only the text columns
    text_columns = [column for column in columns if df_input[column].dtype == "O"]

    pii_results = dict()

    for column in text_columns:
        input_list = df_input[column].values  # only need the distinct ones
        PII = pii_from_texts(input_list, matcher, nlp)
        pii_results[column] = PII

    return pii_results


def get_pii_rows(pii_results, pii_type):
    """
    Given a results dict with the pii results in each of the rows
    return the rows indices that contain each pii type

    Return:
    list of row indices where the specified pii type appearss
    """

    row_indices = []

    for column in pii_results.keys():
        for row_index in pii_results[column].keys():
            if len(pii_results[column][row_index][pii_type]) > 0:
                row_indices.append(row_index)

    return list(set(row_indices))
