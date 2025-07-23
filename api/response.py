from fastapi import status
from fastapi.responses import JSONResponse, Response
from typing import Union, List, Any, TypeVar, Generic
from pydantic import BaseModel

T = TypeVar('T')

class ApiResponse(BaseModel):
    code: int
    msg: str
    data: Any = None

    def __call__(self, headers: dict = None) -> Response:
        return JSONResponse(
            content=self.model_dump(),
            headers=headers
        )

class OkResponse(ApiResponse, Generic[T]):
    code: int = 200
    msg: str = "ok"
    data: T | List[T] | None = None

class ServerErrorResponse(ApiResponse):
    code: int = 500
    msg: str = "server error"
    data: Union[list, dict, str, None] = None

class BadRequestResponse(ApiResponse):
    code: int = 400
    msg: str = "bad request"
    data: Union[list, dict, str, None] = None
