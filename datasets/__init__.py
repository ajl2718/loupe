import pandas as pd 

### Fix this. Path is incorrect
def load_data(dataset_name):
    """
    Load up one of the inbuilt datasets for testing:

    'story'
    """
    if dataset_name == 'story':
        df = pd.read_csv('dataset.csv')
    else:
        df = pd.read_csv('dataset.csv')
    return df
