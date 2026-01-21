#include "board.h"
#include <iostream>

Board::Board() {
    reset();
}

void Board::reset() {
    // Reset all bitboards to zero
    for (int i = 0; i < 12; i++) bitboards[i] = 0ULL;

    // --- WHITE PIECES ---
    bitboards[WHITE * 6 + PAWN]   = 0x000000000000FF00ULL; // Rank 2
    bitboards[WHITE * 6 + KNIGHT] = 0x0000000000000042ULL; // B1, G1
    bitboards[WHITE * 6 + BISHOP] = 0x0000000000000024ULL; // C1, F1
    bitboards[WHITE * 6 + ROOK]   = 0x0000000000000081ULL; // A1, H1
    bitboards[WHITE * 6 + QUEEN]  = 0x0000000000000008ULL; // D1
    bitboards[WHITE * 6 + KING]   = 0x0000000000000010ULL; // E1

    // --- BLACK PIECES ---
    bitboards[BLACK * 6 + PAWN]   = 0x00FF000000000000ULL; // Rank 7
    bitboards[BLACK * 6 + KNIGHT] = 0x4200000000000000ULL; // B8, G8
    bitboards[BLACK * 6 + BISHOP] = 0x2400000000000000ULL; // C8, F8
    bitboards[BLACK * 6 + ROOK]   = 0x8100000000000000ULL; // A8, H8
    bitboards[BLACK * 6 + QUEEN]  = 0x0800000000000000ULL; // D8
    bitboards[BLACK * 6 + KING]   = 0x1000000000000000ULL; // E8

    // Update Occupancy
    occupancy[WHITE] = 0ULL;
    for (int i = 0; i < 6; i++) occupancy[WHITE] |= bitboards[i];
    
    // black pieces index (6-11)
    occupancy[BLACK] = 0ULL;
    for (int i = 6; i < 12; i++) occupancy[BLACK] |= bitboards[i];

    occupancy[BOTH] = occupancy[WHITE] | occupancy[BLACK];
}

char Board::getPieceChar(int square) {
    const char* pieceChars = "PNBRQKpnbrqk";
    U64 bit = (1ULL << square);
    for (int i = 0; i < 12; i++) {
        if (bitboards[i] & bit) return pieceChars[i];
    }
    return '.';
}

void Board::printBoard() {
    std::cout << "\n  a b c d e f g h\n";
    for (int rank = 7; rank >= 0; rank--) {
        std::cout << rank + 1 << " ";
        for (int file = 0; file < 8; file++) {
            int square = rank * 8 + file;
            std::cout << getPieceChar(square) << " ";
        }
        std::cout << rank + 1 << "\n";
    }
    std::cout << "  a b c d e f g h\n\n";
}