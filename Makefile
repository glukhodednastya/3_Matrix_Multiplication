CC = gcc-10
FLAGS = -Wall -pedantic -Wextra -Wno-unused-variable
PAR = -fopenmp

all:
	make clean && make compile && make run && make clean

compile: usual block

usual: 3mm.c 3mm.h general_funcs.h
	$(CC) 3mm.c $(PAR) $(FLAGS) -o 3mm

block: 3mm_block.c 3mm.h general_funcs.h
	$(CC) 3mm_block.c $(PAR) $(FLAGS) -o 3mm_block

run:
	make run_usual && make block_run

run_usual: 
	./3mm

block_run:
	./3mm_block

clean:
	rm -rf *.o 3mm 3mm_block