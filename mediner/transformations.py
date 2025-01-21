import hashlib
import json
import datetime
import logging
import spacy
from spacy.tokens import DocBin


logger = logging.getLogger(__name__)


def text_to_md5(text: str) -> str:
    """Convert a text input into a md5 string

    :returns str: md5 hex string
    """
    md5 = hashlib.md5()
    md5.update(text.encode())
    return md5.hexdigest()


def files_to_annotations(filenames: list[str]) -> list[dict]:
    """Take a list of annotation files from label studio
    and deduplicate annotation objects from filenames.

    :return list: list of annotation dicts from label studio
    """
    dictionary = dict()
    for filename in filenames:
        logger.info(f"Reading annotations from {filename}")
        with open(filename) as jf:
            annotations = json.load(jf)
        for annotation in annotations:
            md5 = text_to_md5(annotation['data']['text'])
            if md5 not in dictionary:
                dictionary[md5] = annotation
            else:
                logger.info(f"Duplicate annotation; {md5}")
                current = datetime.datetime.fromisoformat(
                    annotation['updated_at']
                )
                previous = datetime.datetime.fromisoformat(
                    dictionary[md5]['updated_at']
                )
                if current > previous:
                    logger.info(
                        f"Replacing older with newer annotation; {md5}"
                    )
                    dictionary[md5] = annotation
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


def annotations_to_docbin(annotations: list[dict]) -> DocBin:
    """Convert the label studio annotations to the spacy format.

    :return None:
    """
    nlp = spacy.blank("en")
    docbin = DocBin()
    skipped = 0
    total = 0
    for annotation in annotations:
        text = annotation['data']['text']
        doc = nlp(text)
        # Annotations are lists of [start, end, label]
        span_labels = [
            [
                res['value']['start'],
                res['value']['end'],
                res['value']['labels'][0]
            ]
            for ann in annotation['annotations']
            for res in ann['result']
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
    logger.info(f"Gathered {total} entities from {len(annotations)} inputs")
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
