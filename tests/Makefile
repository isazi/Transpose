
debug		:= 0
include		../Makefile.inc


all: clean Transpose
 
Transpose: Transpose.cpp
	$(CC) -o $(PROJ_BASE)/bin/TransposeTest Transpose.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/TransposeTest

