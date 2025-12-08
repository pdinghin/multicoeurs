PROG = stencil
PROG_VEC = stencil_vec
PROG_STARPU = stencil_starpu
PROG_STARPU_VEC = stencil_starpu_vec
CFLAGS = -Wall -g -O3 -mavx2 -mfma 
LDLIBS = -lm
CC = gcc

STARPU_VERSION = starpu-1.3
CFLAGS += $(shell pkg-config --cflags $(STARPU_VERSION))
LDLIBS += $(shell pkg-config --libs $(STARPU_VERSION))


.phony: all clean

all: $(PROG)

$(PROG): stencil.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

stencil_vec: stencil_vec.c
	$(CC) $(CFLAGS) $< -o $(PROG_VEC) $(LDLIBS)

stencil_starpu: stencil_starpu.c
	$(CC) $(CFLAGS) $< -o $(PROG_STARPU) $(LDLIBS)

stencil_starpu_vec: stencil_starpu_vec.c
	$(CC) $(CFLAGS) $< -o $(PROG_STARPU_VEC) $(LDLIBS)

clean:
	rm -fv $(PROG) $(PROG_VEC) $(PROG_STARPU) $(PROG_STARPU_VEC)
