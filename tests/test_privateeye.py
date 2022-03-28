from deid import __version__
from deid.utils.check_digits import is_medicare, is_tfn
from deid.utils.pii_utils import pii_from_texts, create_matcher
import spacy
import os

texts = ["Last seen leaving supermarket with Jean Claude whose number is 0402234123 and tfn is 304 876 149",
"Then went to the park with his friends peter and ahmed. Peter's phone number is 0432 331 444",
"Here is another example (ph: 0402234123, TFN: 304876149) and 04464555353 not a phone number",
"The cars rego was CYI-855 in otherwords CYI 855",
"A valid credit card number is 2347 6234 2348 6510",
"28th of June 1981 is somebody's birthday.",
"Phones: (03) 8234 1234, 0400 234 234, 90234993, +61 423 234 231", "DOJ23433", "VCD2341123"]

pattern_file = 'privateeye/patterns.jsonl'

def test_version():
    assert __version__ == '0.1.0'

# check that a number is a medicare number
def test_medicare1():
    assert is_medicare(2434576182) == True

# check that a number is not a medicare number
def test_medicare2():
    assert is_medicare(2434576128) == False

# check that a number is a tax file number
def test_tfn1():
    assert is_tfn(304876149) == True

# check that a number is not a tax file number
def test_tfn2():
    assert is_tfn(304876194) == False

# test that PII types are being detected in example texts
def test_tax_file_number():
    nlp = spacy.load('en_core_web_lg')
    matcher = create_matcher(nlp, pattern_file)
    results = pii_from_texts(texts, matcher, nlp)
    assert results[0]['TAX_FILE_NUMBER'][0] == '304 876 149'
    assert results[2]['TAX_FILE_NUMBER'][0] == '304876149'

def test_car_rego_vic():
    nlp = spacy.load('en_core_web_lg')
    matcher = create_matcher(nlp, pattern_file)
    results = pii_from_texts(texts, matcher, nlp)    
    assert results[3]['CAR_REG_VIC'] == ['CYI-855', 'CYI 855']
    assert results[7]['CAR_REG_VIC'] == []
    assert results[8]['CAR_REG_VIC'] == []

def test_credit_card_number():
    nlp = spacy.load('en_core_web_lg')
    matcher = create_matcher(nlp, pattern_file)
    results = pii_from_texts(texts, matcher, nlp)    
    assert results[4]['CREDIT_CARD_NUMBER'][0] == '2347 6234 2348 6510'    

def test_person_name():
    nlp = spacy.load('en_core_web_lg')
    matcher = create_matcher(nlp, pattern_file)
    results = pii_from_texts(texts, matcher, nlp)  
    assert results[0]['PERSON_NAME'][0] == 'Jean Claude' 

def test_date():
    nlp = spacy.load('en_core_web_lg')
    matcher = create_matcher(nlp, pattern_file)
    results = pii_from_texts(texts, matcher, nlp)
    assert results[5]['DATE'][0] == '28th of June 1981'

def test_phone_au():
    nlp = spacy.load('en_core_web_lg')
    matcher = create_matcher(nlp, pattern_file)
    results = pii_from_texts(texts, matcher, nlp)
    assert results[6]['PHONE_AU'][0] == '(03) 8234 1234'
    assert results[6]['PHONE_AU'][1] == '8234 1234'
    assert results[6]['PHONE_AU'][2] == '0400 234 234'
    assert results[6]['PHONE_AU'][3] == '90234993'
    assert results[6]['PHONE_AU'][4] == '+61 423 234 231'

    