
debug		:= 0
include		../Makefile.inc


all: clean Transpose TransposeAVX
 
Transpose: Transpose.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/TransposeTest Transpose.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS)

TransposeAVX: TransposeAVX.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/TransposeAVXTest TransposeAVX.cpp $(CPU_INCLUDES) $(CPU_CFLAGS) $(CPU_LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/TransposeTest
	rm -f $(PROJ_BASE)/bin/TransposeAVXTest

