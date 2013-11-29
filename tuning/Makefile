
debug		:= 0
include		../Makefile.inc


all: clean Transpose
 
Transpose: Transpose.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/TransposeTuning Transpose.cpp $(UTILS_SRC) $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/TransposeTuning
