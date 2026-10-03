CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -fopenmp -Wall -Wextra
SRC      = src/main.cpp src/convolution.cpp
BIN      = bin/convolve

all: $(BIN)

$(BIN): $(SRC) src/convolution.hpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN)

clean:
	rm -rf bin
