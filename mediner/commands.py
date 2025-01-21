import logging
import pandas
import json
import spacy
import pickle
import datetime

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
        output_filename: str = None,
        output_path: str = 'mediner_model',
        train_spacy_filename: str = 'train.spacy',
        dev_spacy_filename: str = 'dev.spacy',
        config_filename: str = 'config.cfg') -> str:
    """Train a NER from the source annotation filenames
    
    :returns str: path of trained model binary
    """
    logger.info(f"Training from {len(input_filenames)} filenames")
    annotations = transformations.files_to_annotations(input_filenames)
    dev_annotations, train_annotations = transformations.split_dev_train(annotations)
    logger.info(f"Split annotations, dev {len(dev_annotations)}, train {len(train_annotations)}")
    dev_docbin = transformations.annotations_to_docbin(dev_annotations)
    train_docbin = transformations.annotations_to_docbin(train_annotations)
    dev_docbin.to_disk('dev.spacy')
    train_docbin.to_disk('train.spacy')
    output_path = 'mediner_model'
    model_choice = 'model-best'
    spacy.cli.train.train(
        config_filename,
        output_path=output_path,
        overrides={
            "paths.train": train_spacy_filename,
            "paths.dev": dev_spacy_filename
        }
    )
    nlp = spacy.load(
        "{}/{}".format(output_path, model_choice)
    )
    # https://spacy.io/usage/saving-loading#pipeline
    serialized_model = {
        'config': nlp.config.to_str(),
        'nlp': nlp.to_bytes()
    }
    now = datetime.datetime.now()
    if output_filename is None:
        output_filename = f"mediner-model-{now.year}-{now.month}-{now.day}-{now.timestamp()}.pkl"
    logger.info(f"Saving model; {output_filename}")
    with open(output_filename, 'wb') as f:
        pickle.dump(serialized_model, f)
    
    return output_filename


def load(filename: str):
    logger.info(f"Loading model from {filename}")
    with open(filename, 'rb') as f:
        serialized_model = pickle.load(f)
    config = spacy.util.load_config_from_str(
        serialized_model['config']
    )
    lang_cls = spacy.util.get_lang_class(
        config["nlp"]["lang"]
    )
    nlp = lang_cls.from_config(config)
    nlp.from_bytes(
        serialized_model['nlp']
    )
    logger.info(f"Model loaded into nlp object; {nlp}")
    return nlp