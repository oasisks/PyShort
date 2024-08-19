from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List
from enum import StrEnum
from datetime import datetime


class Status(StrEnum):
    PENDING: str = "pending"
    COMPLETE: str = "complete"
    FAILED: str = "failed"


class GenericResponse(BaseModel):
    content: bytes = Field(default=None, title="content", description="The response content.")
    json_response: Optional[Dict] = Field(default=None, title="json", description="The json response.")


class AsyncResponse(GenericResponse):
    status_link: Optional[str] = Field(default=None, title="status_link", description="The status link for async.")


class PartitionResponse(BaseModel):
    dataset_name: str = Field(default=None, title="dataset_name", description="The name of the dataset.")
    dataset_group: str = Field(default=None, title="dataset_group", description="The group of the dataset.")
    partition_fields: List[str] = Field(default=None, title="partition_fields", description="Names of the partition "
                                                                                            "fields.")
    partitions: List[Partition] = Field(default=None, title="partitions", description="A list of partitions.")


class Partition(BaseModel):
    values: List[str] = Field(default=None, title="value", description="A parallel list of values to partition_fields.")
