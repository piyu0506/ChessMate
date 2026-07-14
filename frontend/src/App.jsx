import "./styles.css";
import ChessBoardPanel from "./components/ChessBoardPanel";

export default function App() {
  return (
    <div className="app">

      <header className="hero">

        <h1> ChessMate</h1>

        <p>
          
        </p>

      </header>

      <ChessBoardPanel />

      <footer className="footer">

        <p>
          Powered by React • FastAPI • PyTorch • MCTS
        </p>

      </footer>

    </div>
  );
}