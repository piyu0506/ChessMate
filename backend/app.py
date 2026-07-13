from fastapi import FastAPI
from pydantic import BaseModel

from engine import ChessEngine

app = FastAPI(title="Chess Engine API")

engine = ChessEngine()


class MoveRequest(BaseModel):
    fen: str
    move: str


@app.get("/")
def root():
    return {
        "message": "Chess Engine API is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/move")
def make_move(request: MoveRequest):
    return engine.get_ai_response(
        request.fen,
        request.move
    )