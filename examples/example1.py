import pandas as pd
from loupe import TextInspector
from time import time
from docx import Document
import numpy as np
import pdfplumber

# load up some data
df = pd.read_csv('datasets/dataset.csv')

# initialize text inspector object
inspector = TextInspector()

# calculate the counts of pii in the dataset
counts = inspector.inspect_dataframe(df)

# calculate the counts of pii in the word doc
counts_doc = inspector.inspect_doc(doc)

# get the rows that each of the pii types occurs in
rows = inspector.pii_rows()

# convert the results of the pii detection to a dataframe
df = pd.read_csv('datasets/movie_lines.csv')

inputs = df.sample(1024)]['text'].astype(str).values

pii = inspector.inspect(inputs, pii_types=['PERSON_NAME', 'ORG', 'DATE', 'NORP', 'GPE', 'LANGUAGE', 'FAC'])

df = pd.DataFrame(pii)
df['text'] = inputs
columns = ['text'] + list(df.columns)[:-1]
df[columns].to_csv('outputs/movie_lines_pii.csv')

texts = list(df['text'].fillna('').values)
# load up the movie lines and time how long it takes
num_texts = 100000
t1 = time()
results = inspector.inspect_list(texts[:num_texts])
t2 = time()
print(f'Took {np.round(t2-t1,2)}s to inspect {num_texts} texts.')