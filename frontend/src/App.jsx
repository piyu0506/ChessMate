import { useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import axios from "axios";

function App() {
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

    // Show player's move immediately
    setGame(new Chess(gameCopy.fen()));
    setThinking(true);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/move",
        {
          fen: game.fen(),
          move: move.from + move.to,
        }
      );

      if (response.data.success) {
        const updatedGame = new Chess(response.data.fen);
        setGame(updatedGame);
      } else {
        alert(response.data.message);
      }
    } catch (error) {
      console.error(error);
      alert("Could not connect to backend.");
    }

    setThinking(false);

    return true;
  }

  return (
    <div
      style={{
        width: "100%",
        minHeight: "100vh",
        backgroundColor: "#1e1e1e",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <div style={{ width: "600px" }}>
        <Chessboard
          id="ChessBoard"
          position={game.fen()}
          onPieceDrop={onDrop}
          arePiecesDraggable={!thinking}
        />

        <h2
          style={{
            color: "white",
            textAlign: "center",
            marginTop: "20px",
          }}
        >
          {thinking ? "AI Thinking..." : "Your Turn"}
        </h2>
      </div>
    </div>
  );
}

export default App;