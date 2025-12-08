PROG = stencil
PROG_VEC = stencil_vec
PROG_STARPU = stencil_starpu
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

clean:
	rm -fv $(PROG) $(PROG_VEC) $(PROG_STARPU)
