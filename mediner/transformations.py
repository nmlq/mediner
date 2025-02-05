import hashlib
import json
import datetime
import logging
import spacy
from mediner import types
from spacy.tokens import DocBin


logger = logging.getLogger(__name__)


def text_to_md5(text: str) -> str:
    """Convert a text input into a md5 string

    :returns str: md5 hex string
    """
    md5 = hashlib.md5()
    md5.update(text.encode())
    return md5.hexdigest()


def files_to_tasks(filenames: list[str]) -> list[dict]:
    """Take a list of task project files from label studio
    and deduplicate objects from the filenames.

    :return list: list of annotation dicts from label studio
    """
    dictionary = dict()
    for filename in filenames:
        logger.info(f"Reading annotations from {filename}")
        with open(filename) as jf:
            tasks = [
                types.Task.from_dict(d)
                for d in json.load(jf)
            ]
        for task in tasks:
            md5 = text_to_md5(task.data.text)
            if md5 not in dictionary:
                dictionary[md5] = task
            else:
                logger.info(f"Duplicate annotation; {md5}")
                current = task.updated_at
                previous = dictionary[md5].updated_at
                if current and previous and current > previous:
                    logger.info(
                        f"Replacing older with newer annotation; {md5}"
                    )
                    dictionary[md5] = task
    return list(dictionary.values())


def merge_overlaps(
        entity_spans: list,
        annotations: list) -> list:
    """Merge overlapping annotations given entity spans for a doc.

    Go through each entity span, and if the current one is starts before
    the ending of the previous, fix the last annotations char end offset.

    If there are no overlaps the merged annotations is the same
    list we started with.

    :return list:
    """
    if len(annotations) <= 1:
        return annotations

    # convert to list of lists for inline assignment
    annotations = list(map(list, annotations))

    # annotations must be in order
    annotations = sorted(annotations)

    prev_entity_span = entity_spans[0]
    merged_annotations = [annotations[0]]

    for curr_entity_span, (start, end, label) in zip(
            entity_spans[1:],
            annotations[1:]):
        # overlaps
        if curr_entity_span.start <= prev_entity_span.end:
            # fix the last annotations `end` char
            # to be the current `end` char
            merged_annotations[-1][1] = end
        # doesn't overlap, just keep it
        else:
            merged_annotations.append([start, end, label])

        prev_entity_span = curr_entity_span

    # convert back to list of tuples
    merged_annotations = list(map(tuple, merged_annotations))

    return merged_annotations


def tasks_to_docbin(tasks: list[types.Task]) -> DocBin:
    """Convert the label studio tasks to the spacy format.

    :return None:
    """
    nlp = spacy.blank("en")
    docbin = DocBin()
    skipped = 0
    total = 0
    for task in tasks:
        text = task.data.text
        doc = nlp(text)
        # Annotations are lists of [start, end, label]
        span_labels = [
            [
                entity_result.value.start,
                entity_result.value.end,
                entity_result.value.labels[0]
            ]
            for annotation in task.annotations
            for entity_result in annotation.result
        ]
        entity_spans = [
            doc.char_span(start, end, label=label, alignment_mode='expand')
            for start, end, label in span_labels
        ]
        merged_span_labels = merge_overlaps(entity_spans, span_labels)
        merged_spans = [
            doc.char_span(start, end, label=label, alignment_mode='expand')
            for start, end, label in merged_span_labels
        ]

        ents = []
        for ent in merged_spans:
            if ent.text != ent.text.strip():
                skipped += 1
                logger.debug(f"Entity with whitespace, skipping; '{ent.text}'")
                continue

            ents.append(ent)
        doc.ents = ents
        total += len(ents)

        docbin.add(doc)
    logger.info(f"Skipped {skipped} entities with whitespace")
    logger.info(f"Gathered {total} entities from {len(tasks)} inputs")
    return docbin


def split_dev_train(
        annotations: list,
        amount: float = 0.2) -> tuple:
    """Split the annotations into a dev/train tuple.

    :return tuple: (dev, train) split
    """
    index = int(len(annotations) * amount)
    dev, train = annotations[:index], annotations[index:]
    return dev, train
