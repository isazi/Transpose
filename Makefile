
# https://github.com/isazi/utils
UTILS := $(HOME)/src/utils
# https://github.com/isazi/OpenCL
OPENCL := $(HOME)/src/OpenCL

INCLUDES := -I"include" -I"$(UTILS)/include"
CL_INCLUDES := $(INCLUDES) -I"$(OPENCL)/include"
CL_LIBS := -L"$(OPENCL_LIB)"

CFLAGS := -std=c++11 -Wall
ifneq ($(debug), 1)
	CFLAGS += -O3 -g0
else
	CFLAGS += -O0 -g3
endif

LDFLAGS := -lm
CL_LDFLAGS := $(LDFLAGS) -lOpenCL

CC := g++

# Dependencies
DEPS := $(UTILS)/bin/ArgumentList.o $(UTILS)/bin/Timer.o $(UTILS)/bin/utils.o bin/Transpose.o
CL_DEPS := $(DEPS) $(OPENCL)/bin/Exceptions.o $(OPENCL)/bin/InitializeOpenCL.o $(OPENCL)/bin/Kernel.o 


all: Transpose.o TransposeTest TransposeTuning printCode

Transpose.o: $(UTILS)/bin/utils.o include/Transpose.hpp src/Transpose.cpp
	$(CC) -o bin/Transpose.o -c src/Transpose.cpp $(CL_INCLUDES) $(CFLAGS)

TransposeTest: $(CL_DEPS) src/TransposeTest.cpp
	$(CC) -o bin/TransposeTest src/TransposeTest.cpp $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

TransposeTuning: $(CL_DEPS) src/TransposeTuning.cpp
	$(CC) -o bin/TransposeTuning src/TransposeTuning.cpp $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

printCode: $(DEPS) src/printCode.cpp
	$(CC) -o bin/printCode src/printCode.cpp $(DEPS) $(INCLUDES) $(LDFLAGS) $(CFLAGS)

clean:
	rm bin/*

