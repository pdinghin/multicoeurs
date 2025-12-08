PROG = stencil

CFLAGS = -Wall -g -O3 -mavx2 -mfma
LDLIBS = -lm

STARPU_VERSION = starpu-1.3
CFLAGS += $(shell pkg-config --cflags $(STARPU_VERSION))
LDLIBS += $(shell pkg-config --libs $(STARPU_VERSION))


.phony: all clean

all: $(PROG)

clean:
	rm -fv $(PROG)
