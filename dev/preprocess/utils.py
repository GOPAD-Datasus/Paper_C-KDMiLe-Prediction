import pandas as pd


def create_obito(sinasc: pd.DataFrame,
                 merger: pd.DataFrame) -> pd.DataFrame:

    list_ = merger['index_1'].to_list()

    sinasc['OBITO'] = 0
    sinasc.loc[list_, 'OBITO'] = 1

    return sinasc
