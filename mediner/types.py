import typing
import datetime as dt
from label_studio_sdk.core.datetime_utils import serialize_datetime
from label_studio_sdk.types import annotation, prediction, base_task
from label_studio_sdk.core.pydantic_utilities import pydantic_v1


class SpanValue(pydantic_v1.BaseModel):
    start: int
    end: int
    text: typing.Optional[str] = pydantic_v1.Field(default=None)
    labels: typing.Optional[typing.List] = pydantic_v1.Field(default=None)
    score: typing.Optional[float] = pydantic_v1.Field(default=None)


class EntityResult(pydantic_v1.BaseModel):
    id: typing.Optional[str] = None
    value: SpanValue
    origin: str = pydantic_v1.Field(default="manual")
    from_name: str = pydantic_v1.Field(default="label")
    to_name: str = pydantic_v1.Field(default="text")
    type: str = pydantic_v1.Field(default="labels")


class Annotation(annotation.Annotation):
    result: typing.Optional[typing.List[EntityResult]]
    completed_by: typing.Optional[int | typing.Dict]

    def dict(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
        dictionary = super().dict(**kwargs)
        if 'updated_at' in dictionary:
            dictionary['updated_at'] = serialize_datetime(
                dictionary['updated_at']
            )
        if 'created_at' in dictionary:
            dictionary['created_at'] = serialize_datetime(
                dictionary['created_at']
            )
        if 'draft_created_at' in dictionary and isinstance(
                dictionary['draft_created_at'], dt.datetime):
            dictionary['draft_created_at'] = serialize_datetime(
                dictionary['draft_created_at']
            )
        return dictionary


class Prediction(prediction.Prediction):
    result: typing.Optional[typing.List[EntityResult]]

    def dict(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
        dictionary = super().dict(**kwargs)
        if 'updated_at' in dictionary:
            dictionary['updated_at'] = serialize_datetime(
                dictionary['updated_at']
            )
        if 'created_at' in dictionary:
            dictionary['created_at'] = serialize_datetime(
                dictionary['created_at']
            )
        return dictionary


class Data(pydantic_v1.BaseModel):
    text: typing.Optional[str]


class Task(base_task.BaseTask):
    annotations: typing.Optional[typing.List[Annotation]]
    predictions: typing.Optional[typing.List[Prediction]]
    data: typing.Optional[Data]

    def dict(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
        dictionary = super().dict(**kwargs)
        if 'updated_at' in dictionary:
            dictionary['updated_at'] = serialize_datetime(
                dictionary['updated_at']
            )
        if 'created_at' in dictionary:
            dictionary['created_at'] = serialize_datetime(
                dictionary['created_at']
            )
        return dictionary
