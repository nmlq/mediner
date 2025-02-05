import datetime
import dateutil
from dataclasses import dataclass, asdict, field


@dataclass
class Data:
    text: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(**dictionary)


@dataclass
class Meta:
    md5: str = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(**dictionary)


@dataclass
class SpanValue:
    start: int
    end: int
    text: str
    labels: list[str]
    score: float = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(**dictionary)


@dataclass
class EntityResult:
    id: str
    value: SpanValue
    from_name: str = "label"
    to_name: str = "text"
    type: str = "labels"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(
            id=dictionary['id'],
            value=SpanValue.from_dict(dictionary['value']),
            from_name=dictionary['from_name'],
            to_name=dictionary['to_name'],
            type=dictionary['type'],
        )


@dataclass
class Annotation:
    result: list[EntityResult]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(
            result=[
                EntityResult.from_dict(entity_result)
                for entity_result in dictionary['result']
            ]
        )


@dataclass
class Prediction:
    result: list[EntityResult]
    model_version: str
    score: float

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(
            model_version=dictionary['model_version'],
            score=dictionary['score'],
            result=[
                EntityResult.from_dict(entity_result)
                for entity_result in dictionary['result']
            ]
        )


@dataclass
class Task:
    data: Data
    meta: Meta
    updated_at: datetime.datetime = None
    predictions: list[Prediction] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)

    def to_dict(self):
        dictionary = asdict(self)
        if self.updated_at is not None:
            dictionary['updated_at'] = self.updated_at.isoformat().replace(
                "+00:00",
                "Z"
            )
        return dictionary

    @classmethod
    def from_dict(cls, dictionary: dict):
        updated_at = None
        if 'updated_at' in dictionary:
            updated_at = dateutil.parser.parse(
                dictionary['updated_at']
            )

        return cls(
            data=Data.from_dict(dictionary['data']),
            meta=Meta.from_dict(dictionary['meta']),
            updated_at=updated_at,
            predictions=[
                Prediction.from_dict(prediction)
                for prediction in dictionary.get("predictions", [])
            ],
            annotations=[
                Annotation.from_dict(annotation)
                for annotation in dictionary.get("annotations", [])
            ]
        )
