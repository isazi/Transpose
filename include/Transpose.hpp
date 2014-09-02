// Copyright 2014 Alessio Sclocco <a.sclocco@vu.nl>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include <string>
#include <x86intrin.h>

#include <utils.hpp>


#ifndef TRANSPOSE_HPP
#define TRANSPOSE_HPP

namespace isa {
namespace OpenCL {

// Sequential transpose
template< typename T > void transpose(const unsigned int M, const unsigned int N, const unsigned int padding, std::vector< T > &input, std::vector< T > &output);
// OpenCL transpose
std::string * getTransposeOpenCL(const unsigned int nrThreads, const unsigned int M, const unsigned int N, const unsigned int padding, const unsigned int vector, std::string typeName);
// AVX transpose
void transposeAVX(const unsigned int M, const unsigned int N, const float * input, float * output);


// Implementations
template< typename T > void transpose(const unsigned int M, const unsigned int N, const unsigned int padding, std::vector< T > &input, std::vector< T > &output) {
  for ( unsigned int i = 0; i < M; i++ ) {
    for ( unsigned int j = 0; j < N; j++ ) {
      output[(j * isa::utils::pad(M, padding)) + i] = input[(i * isa::utils::pad(N, padding)) + j];
    }
  }
}

std::string * getTransposeOpenCL(const unsigned int nrThreads, const unsigned int M, const unsigned int N, const unsigned int padding, const unsigned int vector, std::string typeName) {
  std::string * code = new std::string();

  // Begin kernel's template
	*code = "__kernel void transpose(__global const " + typeName + " * const restrict input, __global " + typeName + " * const restrict output) {\n"
	"const unsigned int baseM = get_group_id(0) * " + isa::utils::toString(nrThreads) + ";\n"
	"const unsigned int baseN = get_group_id(1) * " + isa::utils::toString(nrThreads) + ";\n"
	"__local "+ typeName + " tempStorage[" + isa::utils::toString(nrThreads * nrThreads) + "];"
	"\n"
	// Load input
	"for ( unsigned int m = 0; m < " + isa::utils::toString(nrThreads) + "; m++ ) {\n"
	"if ( baseN + get_local_id(0) < " + isa::utils::toString(N) + " ) {\n"
	"tempStorage[(m * " + isa::utils::toString(nrThreads) + ") + get_local_id(0)] = input[((baseM + m) * " + isa::utils::toString(isa::utils::pad(N, padding)) + ") + (baseN + get_local_id(0))];\n"
	"}\n"
	"}\n";
	if ( nrThreads > vector ) {
		*code += "barrier(CLK_LOCAL_MEM_FENCE);\n";
	}
	// Local in-place transpose
	*code += "for ( unsigned int i = 1; i <= " + isa::utils::toString(nrThreads) + " / 2; i++ ) {\n"
	"unsigned int localItem = (get_local_id(0) + i) % " + isa::utils::toString(nrThreads) + ";\n"
	+ typeName + " temp = 0;\n";
	if ( nrThreads == vector ) {
		*code += "if ( (i < "+ isa::utils::toString(nrThreads) + ") || (get_local_id(0) < " + isa::utils::toString(nrThreads) + " / 2) ) {\n";
	} else {
		*code += "if ( (i < "+ isa::utils::toString(nrThreads) + " - " + isa::utils::toString(nrThreads / 2) + ") || (get_local_id(0) < " + isa::utils::toString(nrThreads) + " / 2) ) {\n";
	}
	*code += "temp = tempStorage[(get_local_id(0) * " + isa::utils::toString(nrThreads) + ") + localItem];\n"
	"tempStorage[(get_local_id(0) * " + isa::utils::toString(nrThreads) + ") + localItem] = tempStorage[(localItem * " + isa::utils::toString(nrThreads) + ") + get_local_id(0)];\n"
	"tempStorage[(localItem * " + isa::utils::toString(nrThreads) + ") + get_local_id(0)] = temp;\n"
	"}\n"
	"}\n";
	if ( nrThreads > vector ) {
		*code += "barrier(CLK_LOCAL_MEM_FENCE);\n";
	}
	// Store output
	*code += "for ( unsigned int n = 0; n < " + isa::utils::toString(nrThreads) + "; n++ ) {\n"
	"if ( baseN + n < " + isa::utils::toString(N) + " ) {\n"
	"output[((baseN + n) * " + isa::utils::toString(isa::utils::pad(M, padding)) + ") + (baseM + get_local_id(0))] = tempStorage[(n * " + isa::utils::toString(nrThreads) + ") + get_local_id(0)];"
	"}\n"
	"}\n"
	"}\n";
  // End kernel's template

  return code;
}

void transposeAVX(const unsigned int M, const unsigned int N, const float * input, float * output) {
  #pragma omp parallel for schedule(static)
  for ( unsigned int m = 0; m < M; m += 8 ) {
    #pragma omp parallel for schedule(static)
    for ( unsigned int n = 0; n < N; n += 8 ) {
      __m256 buffer[8];

      // Load input
      for ( unsigned int i = 0; i < 8; i++ ) {
        buffer[i] =  _mm256_load_ps(&(input[((m + i) * isa::utils::pad(N, 8)) + n])); 
      }

      // Local in-place AVX transpose: https://stackoverflow.com/questions/16941098/fast-memory-transpose-with-sse-avx-and-openmp
      __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
      __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
      __t0 = _mm256_unpacklo_ps(buffer[0], buffer[1]);
      __t1 = _mm256_unpackhi_ps(buffer[0], buffer[1]);
      __t2 = _mm256_unpacklo_ps(buffer[2], buffer[3]);
      __t3 = _mm256_unpackhi_ps(buffer[2], buffer[3]);
      __t4 = _mm256_unpacklo_ps(buffer[4], buffer[5]);
      __t5 = _mm256_unpackhi_ps(buffer[4], buffer[5]);
      __t6 = _mm256_unpacklo_ps(buffer[6], buffer[7]);
      __t7 = _mm256_unpackhi_ps(buffer[6], buffer[7]);
      __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
      __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
      __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
      __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
      __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
      __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
      __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
      __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
      buffer[0] = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
      buffer[1] = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
      buffer[2] = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
      buffer[3] = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
      buffer[4] = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
      buffer[5] = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
      buffer[6] = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
      buffer[7] = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);

      // Store output
      for ( unsigned int i = 0; i < 8; i++ ) {
        _mm256_store_ps(reinterpret_cast< void * >(&(input[((n + i) * isa::utils::pad(M, 8)) + m])), buffer[i]); 
      }
    }
  }
}

} // OpenCl
} // isa

#endif // TRANSPOSE_HPP
