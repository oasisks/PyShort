from pydantic import BaseModel, Field
from typing import Any, Dict, Optional


class GenericResponse(BaseModel):
    content: bytes = Field(default=None, title="content", description="The response content.")
    json_response: Optional[Dict] = Field(default=None, title="json", description="The json response.")
    status_link: Optional[str] = Field(default=None, title="status_link", description="The status link for async")
