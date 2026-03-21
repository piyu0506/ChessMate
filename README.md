# Chess Engine — AlphaZero-style chess AI powered by PyTorch and MCTS

[![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](/)

---

## Table of contents

1. [Overview](#overview)
2. [Features](#features)
3. [Tech stack](#tech-stack)
4. [Architecture overview](#architecture-overview)
5. [Prerequisites](#prerequisites)
6. [Getting started](#getting-started)
7. [Available scripts](#available-scripts)
8. [Project structure](#project-structure)
9. [Configuration reference](#configuration-reference)
10. [Testing](#testing)
11. [Contributing](#contributing)
12. [License](#license)
13. [Acknowledgements](#acknowledgements)

---

## Overview

Chess Engine is a self-contained, AlphaZero-inspired chess AI built entirely in Python. It combines a residual convolutional neural network (trained on real grandmaster games) with a Monte Carlo Tree Search (MCTS) algorithm to select strong moves at play time. A lightweight Pygame graphical interface lets you play against the engine immediately, with no external chess GUI required. The project is aimed at developers and machine-learning enthusiasts who want to understand — and experiment with — neural-network-guided game-tree search at a practical, code-readable level.

---

## Features

- **ResNet-based dual-head neural network** — a shared residual tower produces both a move-probability *policy* vector and a scalar *value* estimate for each position.
- **AlphaZero-style PUCT MCTS** — 600 simulations per move, using Upper Confidence Bounds for Trees guided by neural-network priors.
- **73-plane move encoding** — spatially encodes all queen-type, knight, and underpromotion moves into a compact 4,672-element action space.
- **Elo-filtered training data** — only games where both players are rated ≥ 2,000 Elo are used; draws are skipped for sharper policy supervision.
- **CUDA / CPU auto-detection** — training and inference automatically use a GPU when available and fall back to CPU.
- **Interactive Pygame GUI** — click to select and move pieces; the AI (Black) responds automatically; illegal moves are silently rejected; auto-queen promotion implemented.
- **Graceful model-not-found handling** — the engine warns and continues (with random play) when no model checkpoint is found.
- **Pre-trained checkpoint included** — `chess_model.pth` (~7.5 MB) ships with the repo so you can play straight away.

---

## Tech stack

| Component | Library / Tool | Version |
|-----------|---------------|---------|
| Language | Python | 3.8+ |
| Neural network | PyTorch | 2.x |
| Chess rules & PGN parsing | python-chess | latest |
| GUI / rendering | Pygame | latest |
| Numerical arrays | NumPy | latest |
| Training progress bars | tqdm | latest |

> **GPU acceleration**: CUDA-capable NVIDIA GPU is optional but strongly recommended for training. Inference runs comfortably on CPU.

---

## Architecture overview

```
┌──────────────────────────────────────────────────--┐
│                    play.py (GUI)                   │
│  Pygame window ──► Human move ──► chess.Board      │
│                                        │           │
│                               AI turn? │           │
│                                        ▼           │
│                              mcts.py (MCTS)        │
│                       600 simulations / move       │
│                       PUCT selection + backprop    │
│                                        │           │
│                               calls    ▼           │
│                              model.py (ChessNet)   │
│                         Input: 12×8×8 board tensor │
│                         Residual tower (6 blocks)  │
│                         ┌─────────┬─────────┐      │
│                         │ Policy  │  Value  │      │
│                         │ 4,672   │  scalar │      │
│                         │ logits  │  tanh   │      │
│                         └─────────┴─────────┘      │
└──────────────────────────────────────────────────--┘
```

- **Board representation**: a 12-channel 8×8 binary tensor — one channel per piece type (P, N, B, R, Q, K for each colour).
- **encoder.py** translates `chess.Move` objects to/from the flat integer index used by the network's policy head.
- **MCTS** masks illegal moves out of the network's raw output before normalising probabilities, preventing the engine from ever suggesting an illegal move.

---

## Prerequisites

| Tool | Required version | Notes |
|------|-----------------|-------|
| Python | ≥ 3.8 | Test with `python --version` |
| pip | ≥ 21 | Comes with Python |
| CUDA Toolkit | 11.8 or 12.x | Optional — only for GPU training |

---

## Getting started

### a. Clone the repo

```bash
git clone https://github.com/piyu0506/Chess_Engine.git
cd Chess_Engine
```

### b. Create and activate a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### c. Install dependencies

```bash
pip install -r python/requirements.txt
```

> Includes: `python-chess`, `pygame`, `numpy`, `torch`, `tqdm`.

For GPU-accelerated PyTorch (replace `cu121` with your CUDA version):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install python-chess pygame numpy tqdm
```

### d. Play against the engine

```bash
python play.py
```

A 640×640 Pygame window opens. **You play as White**; click a piece then click a destination square. The AI (Black) responds automatically after each of your moves.

---

## Available scripts

All scripts are run from `python/src/` unless stated otherwise.

| Script | Command | Purpose |
|--------|---------|---------| 
| Play | `python play.py` | Launch the interactive Pygame GUI |

---

## Project structure

```
Chess_Engine/
├── .gitignore               # Ignores .pgn, .npy, .pth, venv, __pycache__
├── README.md
└── python/
    ├── requirements.txt     # Python dependencies
    ├── assets/              # Chess piece images (PNG, 80×80)
    │   ├── wp.png           # White pieces: wp wn wb wr wq wk
    │   └── bp.png           # Black pieces: bp bn bb br bq bk
    ├── models/
    │   └── chess_model.pth  # Pre-trained checkpoint (~7.5 MB)
    └── src/
        ├── play.py          # Entry point — Pygame GUI + game loop
        ├── model.py         # ChessNet architecture (ResBlock + policy/value heads)
        ├── mcts.py          # MCTS implementation (MCTSNode + MCTS classes)
        └── encoder.py       # 73-plane move encoding / decoding
```

---

## Configuration reference

There are no `.env` files or external config files. All configuration is done via **hardcoded constants** near the top of each script.

### `play.py`

| Constant | Type | Default | Purpose |
|----------|------|---------|---------|
| `WIDTH` | int | `640` | Window width in pixels |
| `HEIGHT` | int | `640` | Window height in pixels |
| `SQ_SIZE` | int | `80` | Pixels per square (`WIDTH // 8`) |
| `ASSET_PATH` | str | `../assets` | Directory containing PNG piece images |
| `MODEL_PATH` | str | `../models/chess_model.pth` | Path to the trained model checkpoint |
| `simulations` | int | `600` | MCTS simulations per AI move |

### `mcts.py`

| Constant | Type | Default | Purpose |
|----------|------|---------|---------|
| `c_puct` | float | `1.0` | Exploration constant in the PUCT formula |
| `simulations` | int | `600` | Passed in from `play.py` |

---

## API reference

### `ChessNet` (`model.py`)

```python
class ChessNet(nn.Module):
    def __init__(self, num_res_blocks: int = 6)
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]
```

- **Input** `x`: shape `(B, 12, 8, 8)` or equivalently `(B, 768)` — 12-channel binary board.
- **Output**:
  - `policy` — shape `(B, 4672)` raw logits over the move space.
  - `value` — shape `(B, 1)` scalar in `[-1, 1]`; positive means current player is winning.

### `MCTS` (`mcts.py`)

```python
class MCTS:
    def __init__(self, model: ChessNet, device: torch.device, simulations: int = 600)
    def search(self, board: chess.Board) -> chess.Move
```

- **`search`**: Runs `simulations` iterations of select → expand → backpropagate, then returns the most-visited root child as the best move.

### `encode_move` / `decode_move` (`encoder.py`)

```python
def encode_move(move: chess.Move, turn: bool) -> int | None
def decode_move(idx: int, board: chess.Board) -> chess.Move | None
```

- Maps moves to/from a flat integer index in `[0, 4671]` using the 73-plane spatial scheme.

---

## Testing

There is no dedicated test suite in the current codebase. Manual verification approaches:

1. **Sanity-check the encoder** — verify that `encode_move` followed by `decode_move` returns the original move for all legal moves in a position:

    ```python
    import chess
    from encoder import encode_move, decode_move

    board = chess.Board()
    for move in board.legal_moves:
        idx = encode_move(move, board.turn)
        recovered = decode_move(idx, board)
        assert recovered == move, f"Round-trip failed for {move}"
    print("All moves encode/decode correctly.")
    ```

2. **Smoke-test the model** — load the checkpoint and run a forward pass:

    ```python
    import torch, chess
    from model import ChessNet
    from mcts import MCTS

    model = ChessNet(num_res_blocks=6)
    model.load_state_dict(torch.load("../models/chess_model.pth", map_location="cpu"))
    model.eval()

    engine = MCTS(model, torch.device("cpu"), simulations=10)
    move = engine.search(chess.Board())
    print("Engine chose:", move)
    ```

3. **Play a full game** — launch `play.py` and play to checkmate / stalemate to confirm the game-over screen renders correctly.

---

## Contributing

1. **Fork** the repository and create a feature branch:

    ```bash
    git checkout -b feature/your-feature-name
    ```

2. **Branch naming**: use `feature/`, `fix/`, or `chore/` prefixes.

3. **Coding style**: follow [PEP 8](https://peps.python.org/pep-0008/). Keep functions focused and add docstrings to public functions and classes.

4. **Commit messages**: write in the imperative mood, e.g. `Add Dirichlet noise to root prior`.

5. **Open a pull request** against `main` with a clear description of what was changed and why.

6. **Issues**: open a GitHub Issue before starting significant work to discuss the approach.

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for the full text.

---

## Acknowledgements

- Inspired by the [DeepMind AlphaZero paper](https://www.science.org/doi/10.1126/science.aar6404) — *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*.
- [python-chess](https://python-chess.readthedocs.io/) — the library handling all chess rules, PGN parsing, and move generation.
- [Lichess open database](https://database.lichess.org/) — the source of the grandmaster PGN training data.
- [PyTorch](https://pytorch.org/) — deep-learning framework powering ChessNet.
- [Pygame](https://www.pygame.org/) — GUI rendering engine.
