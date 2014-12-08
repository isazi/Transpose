
debug		:= 0
include		../Makefile.inc


all: clean TransposeAVX
 
TransposeAVX: TransposeAVX.cpp
	$(CC) -o $(PROJ_BASE)/bin/TransposeAVXTest TransposeAVX.cpp $(CPU_INCLUDES) $(CPU_CFLAGS) $(CPU_LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/TransposeAVXTest

