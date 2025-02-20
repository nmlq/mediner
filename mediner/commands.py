import logging
import pandas
import json
import spacy
import pickle
import datetime
import tqdm
import os

from mediner import transformations
from mediner import types


logger = logging.getLogger(__name__)


def convert_csv_to_label_studio(
        input_filename: str,
        text_column: str,
        output_filename: str = None,
        predict: bool = False,
        model_filename: str = None) -> list[types.Task]:
    """Convert a CSV file to a label studio format.

    Output file should be .json format for importing into label studio

    returns list of dictionaries, if output_filename defined saves to disk

    :param str input_filename:
    :param str text_column:
    :param str output_filename:
    :return list: list of Task
    """
    if not input_filename.lower().endswith('.csv'):
        raise ValueError("Input Filename must be a CSV file")

    if output_filename and not output_filename.lower().endswith('.json'):
        raise ValueError("Output Filename must be a JSON file")

    if not text_column:
        raise ValueError("Text column cannot be empty")

    df = pandas.read_csv(input_filename)

    logger.info(f"Converting {len(df)} rows to label studio format")
    logger.debug(df)

    tasks = [
        types.Task(
            data=types.Data(
                text=row[text_column]
            )
        )
        for _, row in df.fillna('').iterrows()
        if row[text_column]
    ]

    if predict and model_filename:
        logger.info(
            f"Predictions enabled, predicting on all inputs {len(tasks)}"
        )
        nlp = load(model_filename)
        total_ents = 0
        for task in tqdm.tqdm(tasks):
            doc = nlp(task.data.text)
            total_ents += len(doc.ents)
            task.predictions = [
                types.Prediction(
                    model_version=model_filename,
                    score=1.0,
                    result=[
                        types.EntityResult(
                            id=ent.name,
                            value=[
                                types.SpanValue(
                                    start=ent.start_char,
                                    end=ent.end_char,
                                    score=1.0,
                                    text=ent.text,
                                    labels=[ent.label_]
                                )
                            ]
                        )
                    ]
                )
                for ent in doc.ents
            ]
        logger.info(
            f"Added total {total_ents} entities to {len(tasks)} tasks"
        )

    logger.info(f"Created {len(tasks)} tasks for label studio")

    if output_filename:
        logger.info(f"Writing to file {output_filename}")
        with open(output_filename, 'w') as jf:
            json.dump(
                [task.dict() for task in tasks],
                jf,
                indent=2
            )

    return tasks


def train(
        input_filenames: list[str],
        output_filename: str = None,
        output_path: str = 'mediner_model',
        train_spacy_filename: str = 'train.spacy',
        dev_spacy_filename: str = 'dev.spacy',
        config_filename: str = None) -> str:
    """Train a NER from the source annotation filenames

    :returns str: path of trained model binary
    """
    if not config_filename:
        dirname = os.path.dirname(os.path.abspath(__file__))
        config_filename = f"{dirname}/config.cfg"

    logger.info(f"Training from {len(input_filenames)} filenames")
    tasks = transformations.files_to_tasks(input_filenames)
    dev_tasks, train_tasks = transformations.split_dev_train(tasks)
    logger.info(
        f"Split; dev {len(dev_tasks)}, train {len(train_tasks)}"
    )
    dev_docbin = transformations.tasks_to_docbin(dev_tasks)
    train_docbin = transformations.tasks_to_docbin(train_tasks)
    dev_docbin.to_disk(dev_spacy_filename)
    train_docbin.to_disk(train_spacy_filename)
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
    suffix = f"{now.year}-{now.month}-{now.day}-{now.timestamp()}"
    if output_filename is None:
        output_filename = f"mediner-model-{suffix}.pkl"
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
