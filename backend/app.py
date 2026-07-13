from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from engine import ChessEngine

app = FastAPI(title="Chess Engine API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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