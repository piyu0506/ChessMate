#include "board.h"
#include <iostream>

int main() {
    Board chessBoard;
    std::cout << "Grandmaster-AI Initialized." << std::endl;
    std::cout << "Starting Position:" << std::endl;
    chessBoard.printBoard();
    return 0;
}