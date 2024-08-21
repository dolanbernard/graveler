
CC=nvcc
PROG=target/graveler
SOURCE=src/graveler.cu

.PHONY: all test clean
.SILENT: all test clean

all: $(PROG)
	-mkdir -p target
	$(CC) $(SOURCE) -o $(PROG) -DEN_TIME

$(PROG): $(SOURCE)

test: $(PROG)
	./$(PROG)

clean:
	rm -rf $(PROG)
