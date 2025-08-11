import pandas as pd

from dev.preprocess.utils import create_obito


def preprocess():
    columns = [
        "APGAR5", "CONSULTAS", "GESTACAO", "PESO", "MESPRENAT", "IDADEMAE",
        "ESTCIVMAE", "PARTO", "RACACOR", "ESCMAE2010", "CODOCUPMAE",
        "CODMUNNASC"]

    sinasc = pd.read_parquet('DN.parquet', columns=columns)
    merge = pd.read_parquet('merge.parquet')

    sinasc = create_obito(sinasc, merge)

    sinasc.dropna(inplace=True)

    sinasc = sinasc[(sinasc['RACACOR'] != 3) or
                    (sinasc['RACACOR'] != 5)]

    sinasc.to_parquet('data/processed.parquet.gzip',
                      compression='gzip')

    return sinasc
