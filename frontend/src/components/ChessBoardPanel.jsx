import { useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import axios from "axios";

const API_URL = "https://chess-engine-api-3q92.onrender.com";

export default function ChessBoardPanel() {
  const [game, setGame] = useState(new Chess());
  const [thinking, setThinking] = useState(false);

  async function onDrop(sourceSquare, targetSquare) {
    if (thinking) return false;

    const gameCopy = new Chess(game.fen());

    let move;

    try {
      move = gameCopy.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: "q",
      });

      if (!move) return false;
    } catch {
      return false;
    }

    setGame(new Chess(gameCopy.fen()));
    setThinking(true);

    try {
      const response = await axios.post(
        `${API_URL}/move`,
        {
          fen: game.fen(),
          move: move.from + move.to,
        }
      );

      if (response.data.success) {
        setGame(new Chess(response.data.fen));
      } else {
        alert(response.data.message);
      }
    } catch (err) {
      console.error(err);
      alert("Unable to connect to Chess Engine.");
    }

    setThinking(false);

    return true;
  }

  function resetGame() {
    setGame(new Chess());
    setThinking(false);
  }

  function statusText() {
    if (game.isCheckmate()) return "♚ Checkmate";
    if (game.isDraw()) return "Draw";
    if (game.isStalemate()) return "Stalemate";
    if (thinking) return "AI Thinking...";
    if (game.inCheck()) return "Check";
    return "♙ Your Turn";
  }

  return (
    <section className="board-card">

      <div className="status">
        {statusText()}
      </div>

      <Chessboard
  id="ChessBoard"
  position={game.fen()}
  onPieceDrop={onDrop}
  arePiecesDraggable={!thinking && !game.isGameOver()}
  boardWidth={560}
  customDarkSquareStyle={{
    backgroundColor: "#4B5563",
  }}
  customLightSquareStyle={{
    backgroundColor: "#F8FAFC",
  }}
/>

      <button
        className="new-game-btn"
        onClick={resetGame}
      >
        New Game
      </button>

    </section>
  );
}