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

#include <Exceptions.hpp>
#include <utils.hpp>


#ifndef TRANSPOSE_HPP
#define TRANSPOSE_HPP

namespace isa {
namespace OpenCL {

// Sequential transpose
template< typename T > void transpose(std::vector< T > &input, std::vector< T > &output);
// OpenCL transpose
std::string * getTransposeOpenCL(const unsigned int nrThreads, const unsigned int M, const unsigned int N, const unsigned int padding, const unsigned int vector, std::string typeName);


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
	*code = "__kernel void " + this->name + "(__global const " + typeName + " * const restrict input, __global " + typeName + " * const restrict output) {\n"
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

} // OpenCl
} // isa

#endif // TRANSPOSE_HPP
