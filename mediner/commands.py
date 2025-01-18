import logging
import pandas
import json

from mediner import transformations


logger = logging.getLogger(__name__)


def convert_csv_to_label_studio(
        input_filename: str,
        text_column: str,
        output_filename: str = None) -> list:
    """Convert a CSV file to a label studio format.

    Output file should be .json format for importing into label studio

    returns list of dictionaries, if output_filename defined saves to disk

    :param str input_filename:
    :param str text_column:
    :param str output_filename:
    :return list:
    """
    if not input_filename.lower().endswith('.csv'):
        raise ValueError("Input Filename must be a CSV file")

    if output_filename and not output_filename.lower().endswith('.json'):
        raise ValueError("Output Filename must be a JSON file")

    if not text_column:
        raise ValueError("Text column cannot be empty")

    df = pandas.read_csv(input_filename)

    logger.debug(df)

    templates = [
        {
            "data": {
                "text": row[text_column]
            },
            "meta_info":
            {
                "md5": transformations.text_to_md5(row[text_column])
            }
        }
        for _, row in df.fillna('').iterrows()
        if row[text_column]
    ]

    logger.info(f"Exported {len(templates)} templates for label studio")

    if output_filename:
        logger.info(f"Writing to file {output_filename}")
        with open(output_filename, 'w') as jf:
            json.dump(templates, jf, indent=2)

    return templates

def train(
        input_filenames: list[str],
        output_filename: str = None) -> str:
    """Train a NER from the source annotation filenames
    
    :returns str: path of trained model binary
    """
    logger.info(f"Training from {len(input_filenames)} filenames")
    annotations = transformations.files_to_annotations(input_filenames)
    docbin = transformations.annotations_to_docbin(annotations)
    pass
