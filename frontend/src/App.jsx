import { useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import axios from "axios";

const API_URL = "https://chess-engine-api-3q92.onrender.com";

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
        alert(response.data.message || "Move failed.");
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