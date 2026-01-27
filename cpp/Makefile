# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -std=c++17 -O3

# Paths
SRC_DIR = src
BIN_DIR = bin
TARGET = $(BIN_DIR)/grandmaster

# Find all .cpp files in src/
SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# Default rule: Compile everything
all: $(TARGET)

$(TARGET): $(SRCS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# Clean rule: Remove the bin folder
clean:
	rm -rf $(BIN_DIR)