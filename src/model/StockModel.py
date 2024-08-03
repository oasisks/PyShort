from pydantic import BaseModel, Field


class Stock(BaseModel):
    name: str = Field(default=None, title="name")

