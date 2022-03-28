from math import ceil, log


def string_is_integer(s):
    """
    Check if a string s represents an integer
    Returns the integer or False if an error is raised
    """
    try:
        return int(s.replace(" ", ""))
    except ValueError:
        return False


def get_digits(number: int):
    """
    Given an integer, return an array containing each of its digits

    Args:
    number: an integer

    Returns:
    list of ints: each entry is a digit of number
    """
    num_digits = ceil(log(number, 10))
    digits = []
    for n in range(0, num_digits):
        exponent = num_digits - 1 - n
        digit = number // (10 ** exponent) - 10 * (number // (10 ** (exponent + 1)))
        digits.append(digit)
    return digits


def is_creditcard(number: int):
    """
    Given a number, check that it is a valid credit card number
    Based on check digits algorithm in: https://www.gizmodo.com.au/2014/01/how-credit-card-numbers-work/
    Luhn algorithm

    Args:
    number: integer with the credit card number

    Returns:
    True if it is a valid credit card number and False otherwise
    """
    is_16_digits = (number < 10 ** 16) & (number > 10 ** 15)
    # mapping from doubling a digit and then summing the resulting digits if >= 10
    digit_mapping = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 1, 6: 3, 7: 5, 8: 7, 9: 9}
    if is_16_digits == False:
        return False
    else:
        digits = get_digits(number)
    # weighted sum over digits
    weighted_sum = sum([digit_mapping[digit] for digit in digits[0::2]]) + sum(
        digits[1::2]
    )
    return weighted_sum % 10 == 0


def is_tfn(number: int) -> bool:
    """
    Given a number, check that it is a valid Australian Tax File Number
    Based on check digits algorithm in: http://www.mathgen.ch/codes/tfn.html

    Args:
    number: integer with the candidate tax file number

    Returns:
    True if it is a valid TFN and False otherwise
    """
    weights = (1, 4, 3, 7, 5, 8, 6, 9, 10)

    is_9_digits = (number < 10 ** 9) & (number > 10 ** 8)

    if is_9_digits == False:
        return False
    else:
        digits = get_digits(number)

    weighted_sum = sum([weight * digit for weight, digit in zip(weights, digits)])

    return weighted_sum % 11 == 0


def is_abn(number: int) -> bool:
    """
    Given a number, check that it is a valid Australian Business Number
    Based on check digits algorithm in: https://abr.business.gov.au/Help/AbnFormat

    Args:
    number: integer with the candidate ABN

    Returns:
    True if it is a valid ABN and False otherwise
    """
    weights = (10, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19)

    is_11_digits = (number < 10 ** 11) & (number > 10 ** 10)

    if is_11_digits == False:
        return False
    else:
        digits = get_digits(number)

    digits[0] = digits[0] - 1

    weighted_sum = sum([weight * digit for weight, digit in zip(weights, digits)])

    return weighted_sum % 89 == 0


def checksum(number: int, num_digits: int, weights):
    """
    Check that a given number satisfies certain check digit conditions
    """

    correct_num_digits = (number < 10 ** num_digits) & (number > 10 ** (num_digits - 1))

    if correct_num_digits == False:
        return False
    else:
        digits = get_digits(number)

    weighted_sum = sum([weight * digit for weight, digit in zip(weights, digits[:-2])])

    return weighted_sum % 10 == digits[-2]


def is_medicare(number: int):
    """
    Given a number, check that it is a valid Medicare number
    Based on check digits algorithm in: https://www.clearwater.com.au/code/medicare

    Args:
    number: integer with the candidate medicare number

    Returns:
    True if it is a valid medicare number and False otherwise
    """
    weights = (1, 3, 7, 9, 1, 3, 7, 9)
    is_10_digits = (number < 10 ** 10) & (number > 10 ** 9)

    if is_10_digits == False:
        return False
    else:
        digits = get_digits(number)

    weighted_sum = sum([weight * digit for weight, digit in zip(weights, digits[:-2])])

    return weighted_sum % 10 == digits[-2]


def candidate_tfn_valid(tfn_candidate_span):
    """
    Given a spacy span representing a tax file number candidate check that it is
    a valid TFN.

    Args
    tfn_candidate_span: spacy span

    Return
    True if it is a TFN or False otherwise
    """
    tfn_candidate = string_is_integer(tfn_candidate_span.text.replace(" ", ""))
    if tfn_candidate:
        return is_tfn(tfn_candidate)
    else:
        return False


def candidate_medicare_valid(medicare_candidate_span):
    """
    Given a spacy span representing a medicare number candidate check that it is
    a valid medicare card number.

    Args
    medicare_candidate_span: spacy span

    Return
    True if it is a medicare card number or False otherwise
    """
    medicare_candidate = string_is_integer(
        medicare_candidate_span.text.replace(" ", "")
    )
    if medicare_candidate:
        return is_medicare(medicare_candidate)
    else:
        return False


def candidate_creditcard_valid(creditcard_candidate_span):
    """
    Given a spacy span representing a credit carrd number candidate check that it is
    a valid credit card number.

    Args
    creditcard_candidate_span: spacy span

    Return
    True if it is a credit card number or False otherwise
    """
    creditcard_candidate = string_is_integer(
        creditcard_candidate_span.text.replace(" ", "")
    )
    if creditcard_candidate:
        return is_creditcard(creditcard_candidate)
    else:
        return False


def candidate_abn_valid(abn_candidate_span):
    """
    Given a spacy span representing ABN candidate check that it is
    a valid ABN.

    Args
    abn_candidate_span: spacy span

    Return
    True if it is a credit card number or False otherwise
    """
    abn_candidate = string_is_integer(abn_candidate_span.text.replace(" ", ""))
    if abn_candidate:
        return is_abn(abn_candidate)
    else:
        return False
