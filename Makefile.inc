
PROJ_BASE	:= $(HOME)/src/Transpose

# https://github.com/isazi/utils
ISA_UTILS_INCLUDE	:= $(HOME)/src/utils/include
# https://github.com/isazi/OpenCL
ISA_OPENCL_INCLUDE	:= $(HOME)/src/OpenCL/include

INCLUDES := -I"$(PROJ_BASE)/include" -I"$(OPENCL_INCLUDE)" -I"$(ISA_UTILS_INCLUDE)" -I"$(ISA_OPENCL_INCLUDE)"
LIBS := -L"$(OPENCL_LIB)"

ifneq ($(debug), 1)
	CFLAGS := -std=c++11 -O3 -g0 -Wall
else
	CFLAGS := -std=c++11 -O0 -g3 -Wall
endif

LDFLAGS := -lm -lOpenCL

CC := g++

