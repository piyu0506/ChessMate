# ♟️ ChessMate — Neural Network Chess Engine

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-latest-646CFF?logo=vite)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**ChessMate** is a full-stack AI chess application where a neural-network-guided Monte Carlo Tree Search (MCTS) engine plays against a human opponent through a responsive React interface. The backend is a FastAPI inference server deployed on Render; the frontend is a Vite-powered React app deployed on Vercel.

🌐 **Live Demo:** [chess-engine-sepia.vercel.app](https://chess-engine-sepia.vercel.app)
🔌 **Backend API:** `https://chess-engine-api-3q92.onrender.com/`

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [Architecture](#-architecture)
4. [Project Structure](#-project-structure)
5. [AI Model and MCTS](#-ai-model-and-mcts)
6. [Technology Stack](#-technology-stack)
7. [Installation and Local Setup](#-installation-and-local-setup)
8. [Running the Backend](#-running-the-backend)
9. [Running the Frontend](#-running-the-frontend)
10. [Deployment](#-deployment)
11. [API Endpoints](#-api-endpoints)
12. [Learning Outcomes](#-learning-outcomes)
13. [Contributing](#-contributing)
14. [Acknowledgements](#-acknowledgements)

---

## 🧩 Overview

ChessMate is an end-to-end AI chess engine application built from scratch. At its core is a residual convolutional neural network trained on high-Elo grandmaster games that evaluates board positions and outputs move probabilities. At inference time, a Monte Carlo Tree Search algorithm uses those probabilities to explore the game tree and select the strongest move.

The application is designed as a three-tier system:

- **Training pipeline** (`training/`) — data preparation, model architecture, and offline training scripts written in PyTorch.
- **Inference backend** (`backend/`) — a FastAPI server that loads the trained model and exposes a REST API for move requests.
- **Interactive frontend** (`frontend/`) — a React application where users play against the AI engine directly in their browser.

The project is fully deployed: the frontend runs on Vercel (CDN-distributed) and the backend runs on Render (containerised Python runtime).

---

## ✨ Key Features

- **Neural network board evaluation** — a residual CNN with a dual policy/value head produces move probabilities and a scalar position score for every board state.
- **MCTS move selection** — Monte Carlo Tree Search explores the game tree using the neural network as a heuristic, balancing exploitation and exploration via the PUCT formula.
- **Illegal-move masking** — raw network output is masked against the list of legal moves before softmax normalisation, guaranteeing the engine never suggests an illegal move.
- **REST API backend** — a FastAPI server accepts board states (FEN strings) and returns best moves, enabling clean separation between AI logic and the UI.
- **React frontend** — an interactive chessboard built in React with full move validation and animated piece movement.
- **CUDA / CPU auto-detection** — inference automatically uses a GPU when available and falls back to CPU.
- **Standalone training scripts** — the full training pipeline (data loading, model definition, training loop) is included in `training/` so the model can be retrained or fine-tuned.
- **Production deployment** — the frontend and backend are independently deployed and communicate over HTTPS.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Browser (Vercel CDN)                    │
│                                                             │
│   ┌──────────────────────────────────────────────────┐      │
│   │             React Frontend (Vite)                │      │
│   │  Interactive chessboard  ──►  User move input    │      │
│   │                                    │             │      │
│   │               POST /move (FEN)     │             │      │
│   └────────────────────────────────────┼─────────────┘      │
└────────────────────────────────────────┼────────────────────┘
                                         │  HTTPS
                                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend (Render)                    │
│                                                             │
│   ┌──────────────────────────────────────────────────┐      │
│   │  app.py  ──►  Loads chess_model.pth on startup   │      │
│   │                         │                        │      │
│   │              Calls MCTS.search(board)            │      │
│   │                         │                        │      │
│   │         ┌───────────────▼────────────────┐       │      │
│   │         │  MCTS (Monte Carlo Tree Search) │       │      │
│   │         │  N simulations per request      │       │      │
│   │         │  PUCT node selection            │       │      │
│   │         │  Expand → Evaluate → Backprop   │       │      │
│   │         └───────────────┬────────────────┘       │      │
│   │                         │                        │      │
│   │         ┌───────────────▼────────────────┐       │      │
│   │         │   ChessNet (PyTorch ResNet)     │       │      │
│   │         │   Input:  12×8×8 board tensor  │       │      │
│   │         │   Output: policy (4672 logits) │       │      │
│   │         │           value  (scalar)      │       │      │
│   │         └────────────────────────────────┘       │      │
│   └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│               Offline Training Pipeline                     │
│               training/  (run locally or on GPU cloud)      │
│                                                             │
│   PGN data  ──►  Encoder  ──►  ChessNet  ──►  chess_model.pth │
└─────────────────────────────────────────────────────────────┘
```

**Data flow for an AI move:**

1. User makes a move in the browser; the React app sends a `POST /move` request with the current board as a FEN string.
2. The FastAPI backend parses the FEN into a `chess.Board` object and passes it to the MCTS engine.
3. MCTS runs *N* simulations. In each simulation it selects a node using PUCT, calls `ChessNet.forward()` to get policy and value estimates, and backpropagates the result.
4. After all simulations, MCTS returns the most-visited child as the best move.
5. The backend responds with the chosen move in UCI notation; the frontend animates the piece.

---

## 📁 Project Structure

```
ChessMate/
├── backend/                    # FastAPI inference server
│   ├── app.py                  # API routes and server entry point
│   ├── model.py                # ChessNet architecture (ResNet + dual heads)
│   ├── mcts.py                 # Monte Carlo Tree Search implementation
│   ├── encoder.py              # 73-plane move encoding / decoding
│   ├── chess_model.pth         # Pre-trained model weights
│   └── requirements.txt        # Backend Python dependencies
│
├── frontend/                   # React + Vite web application
│   ├── src/
│   │   ├── components/         # React components (Board, Square, Piece …)
│   │   ├── App.jsx             # Root component and game state
│   │   └── main.jsx            # Vite entry point
│   ├── public/                 # Static assets (piece images, favicon)
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
│
├── training/                   # Offline training pipeline
│   ├── train.py                # Main training loop
│   ├── model.py                # ChessNet definition (same architecture)
│   ├── dataset.py              # PGN data loading and preprocessing
│   ├── encoder.py              # Shared move encoder
│   └── requirements.txt        # Training dependencies (includes PyTorch)
│
├── play.py                     # Root-level launcher (sys.path helper)
├── render.yaml                 # Render deployment configuration
├── .gitignore
└── README.md
```

---

## 🤖 AI Model and MCTS

### Neural Network — ChessNet

ChessNet is a residual convolutional neural network that takes a board position as input and produces two outputs: a **policy** (probability distribution over moves) and a **value** (scalar score from –1 to +1 indicating how favourable the position is for the side to move).

**Board representation:** Each position is encoded as a 12-channel, 8×8 binary tensor — one channel per piece type (pawn, knight, bishop, rook, queen, king) for each colour, giving 12 × 8 × 8 = 768 features.

**Move encoding:** Moves are encoded using a 73-plane spatial scheme (queen-type moves in 56 planes, knight moves in 8 planes, underpromotion moves in 9 planes), resulting in a flat action space of 4,672 possible moves.

**Architecture:**

```
Input (12 × 8 × 8)
    │
    ▼
Initial Conv Block (Conv2d → BatchNorm → ReLU)
    │
    ▼
Residual Tower (N × ResBlock)
  ┌─────────────────────────┐
  │  Conv2d → BN → ReLU     │
  │  Conv2d → BN            │
  │  + skip connection      │
  │  ReLU                   │
  └─────────────────────────┘
    │
    ├──────────────────────────────┐
    ▼                              ▼
Policy Head                   Value Head
Conv2d → BN → ReLU            Conv2d → BN → ReLU
Flatten → Linear(4672)        Flatten → Linear(256) → ReLU
(raw logits)                  Linear(1) → Tanh
                              (scalar in [-1, 1])
```

### Monte Carlo Tree Search

MCTS builds a search tree rooted at the current position. Each node stores:

- **N** — visit count
- **W** — total value accumulated across simulations
- **Q** — mean action value (W / N)
- **P** — prior probability from the policy head

**Node selection** uses the PUCT formula, which balances exploitation (high Q) and exploration (high P, low N):

```
U(s, a) = Q(s, a) + c_puct × P(s, a) × √(ΣN(s, b)) / (1 + N(s, a))
```

**Simulation loop (repeated N times):**

1. **Select** — traverse the tree by always picking the child with the highest PUCT score until a leaf node is reached.
2. **Expand** — call `ChessNet.forward()` on the leaf position; create child nodes for all legal moves, initialised with their policy priors.
3. **Backpropagate** — propagate the value estimate back up the path, flipping sign at each ply (since the value is always from the perspective of the side to move).

After all simulations, the move with the **highest visit count** is returned as the engine's choice — a robust selection criterion that naturally smooths out noise from individual simulations.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| AI / Training | PyTorch | Neural network definition, training loop, GPU acceleration |
| AI / Inference | python-chess | Board representation, move generation, FEN parsing |
| Backend | FastAPI | REST API server, async request handling |
| Backend | Uvicorn | ASGI server (WSGI-compatible, production-grade) |
| Frontend | React 18 | Component-based UI, game state management |
| Frontend | Vite | Build tooling, dev server with HMR |
| Deployment (backend) | Render | Containerised Python web service |
| Deployment (frontend) | Vercel | CDN-distributed static frontend |
| Language | Python 3.8+ | Backend and training |
| Language | JavaScript (ES2022) | Frontend |

---

## 🔧 Installation and Local Setup

### Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | ≥ 3.8 | `python --version` |
| Node.js | ≥ 18 | `node --version` |
| npm | ≥ 9 | Bundled with Node.js |
| CUDA Toolkit | 11.8 or 12.x | Optional — only needed for GPU training |

### 1. Clone the repository

```bash
git clone https://github.com/piyu0506/ChessMate.git
cd ChessMate
```

---

## 🚀 Running the Backend

### 2. Create and activate a virtual environment

```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Start the FastAPI server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.
Interactive docs (Swagger UI) are at `http://localhost:8000/docs`.

> **Note:** On first startup the server loads `chess_model.pth` into memory. If no checkpoint is found, the engine will warn and fall back to a random-move strategy.

---

## 💻 Running the Frontend

### 5. Install frontend dependencies

Open a new terminal from the project root:

```bash
cd frontend
npm install
```

### 6. Configure the backend URL

Create a `.env` file in the `frontend/` directory:

```env
VITE_API_URL=http://localhost:8000
```

For production, this should point to your deployed Render URL.

### 7. Start the development server

```bash
npm run dev
```

The app will be available at `http://localhost:5173`.

### 8. Production build

```bash
npm run build       # outputs to frontend/dist/
npm run preview     # serves the production build locally
```

### Training (optional)

To retrain the model from scratch or fine-tune on new data:

```bash
cd training
pip install -r requirements.txt
python train.py
```

The trained checkpoint will be saved as `chess_model.pth`. Copy it into `backend/` before starting the server.

---

## ☁️ Deployment

### Backend — Render

The backend is configured for Render via `render.yaml` at the repository root:

```yaml
services:
  - type: web
    name: chess-engine-api
    runtime: python
    rootDir: backend
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
```

**Steps to deploy:**

1. Push the repository to GitHub.
2. Create a new **Web Service** on [render.com](https://render.com) and connect the GitHub repo.
3. Render auto-detects `render.yaml` and configures the service.
4. Set the environment variable `PYTHON_VERSION=3.11` if needed.
5. Deploy — Render installs dependencies and starts Uvicorn automatically.

### Frontend — Vercel

**Steps to deploy:**

1. Import the GitHub repository into [vercel.com](https://vercel.com).
2. Set the **Root Directory** to `frontend`.
3. Vercel auto-detects Vite; the default build settings work out of the box.
4. Add the environment variable:

   ```
   VITE_API_URL=https://<your-render-service>.onrender.com
   ```

5. Deploy — Vercel builds and distributes the static assets globally via its CDN.

> **CORS:** Ensure the FastAPI backend's CORS middleware allows the Vercel deployment URL as an origin.

---

## 📡 API Endpoints

Base URL (local): `http://localhost:8000`
Base URL (production): `https://<your-render-service>.onrender.com`

### `GET /`

Health check — confirms the server is running and the model is loaded.

**Response:**
```json
{
  "status": "ok",
  "message": "ChessMate API is running"
}
```

---

### `POST /move`

Request the engine's best move for the given board position.

**Request body:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
}
```

| Field | Type | Description |
|---|---|---|
| `fen` | `string` | Board position in [FEN notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) |

**Response:**
```json
{
  "move": "e7e5",
  "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
}
```

| Field | Type | Description |
|---|---|---|
| `move` | `string` | Engine's chosen move in UCI notation (e.g. `e7e5`) |
| `fen` | `string` | Updated FEN string after the move is applied |

**Error response (invalid FEN or no legal moves):**
```json
{
  "detail": "Invalid FEN string or game already over"
}
```


---
## 📚 Learning Outcomes

This project provided hands-on experience in designing and deploying an AI-powered software system, covering:

- Deep learning with PyTorch
- Monte Carlo Tree Search (MCTS)
- Chess position encoding and move generation
- REST API development using FastAPI
- Frontend development with React and Vite
- Full-stack deployment using Render and Vercel
- Git, GitHub, and CI/CD workflows

---

## 🤝 Contributing

Suggestions, improvements, and feedback are always welcome. Feel free to fork the repository, open an issue, or submit a pull request.

---

## ⭐ Acknowledgements

This project was developed as a personal learning project to explore artificial intelligence, search algorithms, and full-stack application development through the implementation of a neural-network-guided chess engine.