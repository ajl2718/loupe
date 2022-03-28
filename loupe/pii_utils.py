from spacy.matcher import Matcher
from .check_digits import candidate_medicare_valid, candidate_tfn_valid
from .check_digits import candidate_creditcard_valid, candidate_abn_valid


no_check_digit = lambda x: True

check_digit_algorithms = {
    "PERSON_NAME": no_check_digit,
    "PHONE_AU": no_check_digit,
    "EMAIL": no_check_digit,
    "URL": no_check_digit,
    "MEDICARE_NUMBER": candidate_medicare_valid,
    "TAX_FILE_NUMBER": candidate_tfn_valid,
    "POSTCODE": no_check_digit,
    "DATE": no_check_digit,
    "CAR_REG_VIC": no_check_digit,
    "CREDIT_CARD_NUMBER": candidate_creditcard_valid,
    "AUSTRALIAN_BUSINESS_NUMBER": candidate_abn_valid,
}

# the default types of PII to look for (i.e., all of them)
DEFAULT_PII_TYPES = [
    "PERSON_NAME",
    "PHONE_AU",
    "EMAIL",
    "URL",
    "MEDICARE_NUMBER",
    "TAX_FILE_NUMBER",
    "POSTCODE",
    "DATE",
    "CAR_REG_VIC",
    "CREDIT_CARD_NUMBER",
    "NORP",
    "GPE",
    "LANGUAGE",
    "ORG",
    "FAC",
    "MONEY",
    "AUSTRALIAN_BUSINESS_NUMBER",
]
ENTITY_TYPES = [
    "PERSON",
    "PHONE_AU",
    "EMAIL",
    "URL",
    "MEDICARE_CANDIDATE",
    "TFN_CANDIDATE",
    "POSTCODE",
    "DATE",
    "CAR_REG_VIC",
    "CREDIT_CARD_CANDIDATE",
    "NORP",
    "GPE",
    "LANGUAGE",
    "ORG",
    "FAC",
    "MONEY",
    "ABN_CANDIDATE",
]

ENTITY_PII_MAPPING = dict(zip(DEFAULT_PII_TYPES, ENTITY_TYPES))


def is_nlp_type(pii_type):
    """
    checks a pii_type to see if it requires spaCy language model
    these include:
    - PERSON_NAME
    - GPE
    - ADDRESS
    - FAC
    - LOC
    - DATE
    """
    nlp_types = [
        "PERSON_NAME",
        "GPE",
        "ORG",
        "FAC",
        "LOC",
        "ADDRESS",
        "DATE",
        "NORP",
        "LANGUAGE",
    ]
    if pii_type in nlp_types:
        return True
    else:
        return False


def redact_text(text, pii_spans):
    """
    Given a text and a list of entity spans corresponding to pii items replace each
    of elements of text corresponding to pii with '***'

    Args
    text: str to redact
    pii_spans: list of tuples describing character spans where PII is in text

    Return
    text_pii_removed: str with PII removed from original text in doc
    """
    pii_spans.sort()
    if len(pii_spans) == 0:
        return text
    else:
        gap_start = 0
        gap_end = pii_spans[0][0]
        all_spans = [(gap_start, gap_end)]

        for n in range(0, len(pii_spans) - 1):
            gap_start = pii_spans[n][1]
            gap_end = pii_spans[n + 1][0]
            all_spans.append((pii_spans[n][0], pii_spans[n][1]))
            all_spans.append((gap_start, gap_end))

        all_spans.append((pii_spans[-1][0], pii_spans[-1][1]))
        all_spans.append((pii_spans[-1][1], -1))

        redacted_list = []
        for n, span in enumerate(all_spans):
            if n % 2 == 0:
                redacted_list += [text[span[0] : span[1]]]
            else:
                redacted_list += ["[PII REMOVED]"]
        # include the final character of the text if not in a pii element
        # this could be nicer
        if len(text) != all_spans[-1][0]:
            redacted_list += [text[-1]]
        return "".join(redacted_list)


def pii_from_texts(texts, matcher, nlp, pii_types=DEFAULT_PII_TYPES):
    """
    Extract Personally Identifiable Information from a list of strings
    and return dict describing PII.
    Args
    texts: list of strs to
    matcher: spacy matcher object specifying the patterns to look for
    nlp: spacy language model

    Return:
    list with key, value pair for each str in texts.
    """
    output = []

    use_language_model = False

    nlp_types = [pii_type for pii_type in pii_types if is_nlp_type(pii_type)]
    rule_types = [pii_type for pii_type in pii_types if is_nlp_type(pii_type) == False]

    if len(nlp_types) > 0:
        use_language_model = True

    # run through all the texts and check for PII types
    for doc in nlp.pipe(texts):
        # dict for all pii within a given str
        item_pii = dict()
        for pii_type in pii_types:
            item_pii[pii_type] = []

        matches = matcher(doc)

        # pii types that require nlp language model
        for nlp_type in nlp_types:
            item_pii[nlp_type] = [
                {"text": ent.text, "span": (ent.start_char, ent.end_char)}
                for ent in doc.ents
                if ent.label_ == ENTITY_PII_MAPPING[nlp_type]
            ]

        # pii types defined by rules
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            for pii_type in rule_types:
                # check each of the pii types and
                if string_id == ENTITY_PII_MAPPING[pii_type]:
                    # apply check digit algorithm if necessary
                    check_digit_function = check_digit_algorithms[pii_type]
                    if check_digit_function(doc[start:end]):
                        # append data to appropriate entity type list
                        new_item = {
                            "text": doc[start:end].text,
                            "span": (
                                doc[start:end].start_char,
                                doc[start:end].end_char,
                            ),
                        }
                        item_pii[pii_type].append(new_item)
                else:
                    next
        output.append(item_pii)
    return output


def redact_texts(texts, matcher, nlp, pii_types=DEFAULT_PII_TYPES):
    pii_with_spans = pii_from_texts(texts, matcher, nlp, pii_types)
    redacted_list = []
    for n in range(0, len(texts)):
        text = texts[n]
        spans = []
        pii = pii_with_spans[n]
        for key in pii.keys():
            for pii_item in pii[key]:
                spans.append(pii_item["span"])
        new_text = redact_text(text, spans)
        redacted_list.append(new_text)
    return redacted_list


def load_patterns(file):
    """
    Load up the rules for the patterns from jsonl file, as per spacy documentation.

    Args:
    file: path of jsonl file containing the patterns

    Returns:
    patterns: list of dicts of the form ("match_id": "NAME_OF_ENTITY", "patterns": [{patterns...}])
    """
    with open(file, "r") as f:
        outputs = f.readlines()

    outputs = [eval(output) for output in outputs]

    return outputs


def create_matcher(nlp, pattern_file):
    """
    Create a spacy matcher object containing rules that define the entities we would
    like to detect. These are for well-structured entities such as postcodes, telephone numbers,
    medicare numbers.

    To do: move the rules to another function so that this is neater

    Args:
    nlp: a spacy language model
    pattern_file: str describing path to the .jsonl file with the patterns

    return: spacy.matcher object
    """
    matcher = Matcher(nlp.vocab)

    # load up the patterns from a file and then add each type of pattern
    patterns = load_patterns(pattern_file)
    for pattern in patterns:
        matcher.add(pattern["match_id"], pattern["patterns"])

    return matcher
