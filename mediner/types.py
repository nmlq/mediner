import typing
from dataclasses import dataclass, asdict, field
from label_studio_sdk.types import task, annotation, prediction




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


class Task(task.Task):
    annotations: typing.Optional[typing.List[annotation.Annotation]]
    predictions: typing.Optional[typing.List[prediction.Prediction]]
