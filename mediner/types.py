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
    md5: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(**dictionary)


@dataclass
class SpanValue:
    start: int
    end: int
    score: float
    text: str
    labels: list[str]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(**dictionary)


@dataclass
class EntityResult:
    id: str
    value: list[SpanValue]
    from_name: str = "label"
    to_name: str = "text"
    type: str = "labels"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(
            id=dictionary['id'],
            value=[
                SpanValue.from_dict(span_value)
                for span_value in dictionary['value']
            ],
            from_name=dictionary['from_name'],
            to_name=dictionary['to_name'],
            type=dictionary['type'],
        )


@dataclass
class Prediction:
    model_version: str
    score: float
    result: list[EntityResult]

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
    predictions: list[Prediction] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dictionary: dict):
        return cls(
            data=Data.from_dict(dictionary['data']),
            meta=Meta.from_dict(dictionary['meta']),
            predictions=[
                Prediction.from_dict(prediction)
                for prediction in dictionary.get("predictions", [])
            ]
        )
