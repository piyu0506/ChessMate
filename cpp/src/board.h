#ifndef BOARD_H
#define BOARD_H

#include "types.h"
#include <string>

class Board {
public:
    // 12 bitboards: 6 for White (0-5), 6 for Black (6-11)
    U64 bitboards[12];
    
    // Occupancy bitboards: White, Black, and All pieces
    U64 occupancy[3];

    Board();
    void reset();
    void printBoard();
    void getVector(float *output);
    
private:
    char getPieceChar(int square);
};

#endif