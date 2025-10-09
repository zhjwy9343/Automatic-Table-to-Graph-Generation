""" A helper script to generate the metadata and dataframe information

    This script is used to analyze given tables by converting each of them into a Pandas
    Dataframe, computing statistics of all columns, and save table's metadata into a 
    `metadata.yaml` file, tables' statistics into a `information.txt` file.
    
    Note: Users need to mannually edit the `meteadata.yaml` later to add primary key and
          foreign key information and task fields, which this script cannot automatically extract.

"""

import os
import argparse
import yaml
import pandas as pd
import numpy as np

from models.llm.gconstruct import analyze_dataframes

def categorize_dtypes(df: pd.DataFrame) -> dict:
    colums = []
    for col in df.columns:
        result = {}
        result['name'] = col

        if pd.api.types.is_numeric_dtype(df[col]):
            result['dtype'] = 'float'
        elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
            result['dtype'] = 'text'
        elif pd.api.types.is_bool_dtype(df[col]):
            result['dtype'] = 'category'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            result['dtype'] = 'datetime'
        elif pd.api.types.is_categorical_dtype(df[col]):
            result['dtype'] = 'category'
        else:
            result['dtype'] = 'other'

        colums.append(result)
    return columns


def load_table(table_path: str, table_format:str) -> str, pd.DataFrame:
    """ """
    if table_format == 'csv':
        df = pd.read_csv(table_path)
    elif table_format == 'parquet':
        df = pd.read_parquet(table_path)
    elif table_format == 'numpy':
        np_array = np.load(table_path)
        np_dict = dict(np_array)
        df = pd.DataFrame(np_dict)
    else:
        raise ValueError("Not support data format: {table_format}")

    file_name = os.path.basename(table_path)
    table_name = os.path.splitext(file_name)

    return table_name, df


def analyze_columns(df: pd.DataFrame, table_name: str, data_format: str) -> dict:
    """ get table and column metadata like:

    tables:
    - columns:
      - dtype: float
          name: feat
      - dtype: category
          name: label
      - dtype: primary_key
          name: paperID
      - dtype: category
          name: year
      format: numpy
      name: Paper
      source: data/paper.npz
    """
    table_dict = {}
    table_dict['format'] = data_format
    table_dict['name'] = table_name

    col_types = categorize_dtypes(df)
    table_dict['columns'] = col_types
    
    return table_dict
    

def main(args):
    """ The offline generation procedure """
    # extract all files, not directories
    files = os.path.listdir(args.data_path)
    file_paths = [os.path.join(args.data_path, file) for file in files]
    val_file_paths = [file_path for file_path in file_paths if os.path.isfile(file_path)]
    print(f'Will explore files:\n {val_file_paths}')

    # load in tables
    table_dfs = {}
    table_meta_dict = {}
    table_meta_dict["tables"] = []
    for val_file_path in val_file_paths:
        table_name, table_df = load_table(val_file_path, args.data_format)
        table_dfs[table_name] = table_df

        # analyze columns
        table_meta = analyze_columns(table_df, table_name, args.data_format)
        table_meta["source"] = os.path.basename(val_file_path)
        table_meta_dict["tables"].append(table_meta)

    print(f'Metadata of tables: {table_meta_dict}')

    # Analyze tables
    information = analyze_dataframes(table_dfs, k=5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A Table metadata generator")
    parser.add_argument("--data-path", type=str, required=True,
                        help=("The path of table files to be analyzed. By default, this script "
                              "will explore all files under this path, and use the file name as "
                              "table name."))
    parser.add_argument("--data-format", type=str, choices=['csv', 'parquet', 'numpy'],
                        default='csv', help=("The table file format. Options are 'csv', "
                                             "'parquet', 'numpy'"))
    parser.add_argument("--target-column", type=str, required=True,
                        help="The column name of the target table for prediction. The format "
                             "should be table_name:column_name, using ':'split the table name "
                             "and column name.")
    parser.add_argument("--task-type", type=str, required=True,
                        choices=['classification', 'regression'],
                        help="The prediction task types. Options are 'classification', and "
                        "'regression'")
    parser.add_argument("--output-path", type=str,
                        help=("The path of output for metadata.yaml and information.txt. "
                              "If not given, will use the --data-path by default."))
    parser.add_argument()
    args = parser.parse_args()
    print(f'Arguments: {args}')

    main(args)
    