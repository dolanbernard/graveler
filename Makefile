
CC=nvcc
PROG=target/graveler
SOURCE=src/graveler.cu
CFLAGS=
DFLAGS=-DEN_TIME

.PHONY: all test clean
.SILENT: all test clean

all: $(PROG)
	-mkdir -p target
	$(CC) $(CFLAGS) $(DFLAGS) $(SOURCE) -o $(PROG)

$(PROG): $(SOURCE)

test: $(PROG)
	./$(PROG)

clean:
	-rm -rf $(PROG)
