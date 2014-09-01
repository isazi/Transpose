
debug		:= 0
include		../Makefile.inc


all: clean Transpose
 
Transpose: Transpose.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/TransposeTuning Transpose.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/TransposeTuning
