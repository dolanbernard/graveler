
CC=nvcc
PROG=graveler
SOURCE=src/graveler.cu

.PHONY: all test clean
.SILENT: all test clean

all: $(PROG)
	$(CC) $(SOURCE) -o $(PROG)

$(PROG): $(SOURCE)

test: $(PROG)
	./$(PROG)

clean:
	rm -rf $(PROG)
