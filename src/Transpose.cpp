// Copyright 2015 Alessio Sclocco <a.sclocco@vu.nl>
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

#include <Transpose.hpp>

namespace isa {
namespace OpenCL {

transposeConf::transposeConf() {}

transposeConf::~transposeConf() {}

std::string transposeConf::print() const {
  return std::string(isa::utils::toString(nrItemsPerBlock));
}

std::string * getTransposeOpenCL(const transposeConf & conf, const unsigned int M, const unsigned int N, const unsigned int padding, const unsigned int vector, std::string typeName) {
  std::string * code = new std::string();

  // Begin kernel's template
	*code = "__kernel void transpose(__global const " + typeName + " * const restrict input, __global " + typeName + " * const restrict output) {\n"
	"const unsigned int baseM = get_group_id(0) * " + isa::utils::toString(conf.getNrItemsPerBlock()) + ";\n"
	"const unsigned int baseN = get_group_id(1) * " + isa::utils::toString(conf.getNrItemsPerBlock()) + ";\n"
	"__local "+ typeName + " tempStorage[" + isa::utils::toString(conf.getNrItemsPerBlock() * conf.getNrItemsPerBlock()) + "];"
	"\n"
	// Load input
	"for ( unsigned int m = 0; m < " + isa::utils::toString(conf.getNrItemsPerBlock()) + "; m++ ) {\n"
	"if ( baseN + get_local_id(0) < " + isa::utils::toString(N) + " ) {\n"
	"tempStorage[(m * " + isa::utils::toString(conf.getNrItemsPerBlock()) + ") + get_local_id(0)] = input[((baseM + m) * " + isa::utils::toString(isa::utils::pad(N, padding)) + ") + (baseN + get_local_id(0))];\n"
	"}\n"
	"}\n";
	if ( conf.getNrItemsPerBlock() > vector ) {
		*code += "barrier(CLK_LOCAL_MEM_FENCE);\n";
	}
	// Local in-place transpose
	*code += "for ( unsigned int i = 1; i <= " + isa::utils::toString(conf.getNrItemsPerBlock()) + " / 2; i++ ) {\n"
	"unsigned int localItem = (get_local_id(0) + i) % " + isa::utils::toString(conf.getNrItemsPerBlock()) + ";\n"
	+ typeName + " temp = 0;\n";
	if ( conf.getNrItemsPerBlock() == vector ) {
		*code += "if ( (i < "+ isa::utils::toString(conf.getNrItemsPerBlock()) + ") || (get_local_id(0) < " + isa::utils::toString(conf.getNrItemsPerBlock()) + " / 2) ) {\n";
	} else {
		*code += "if ( (i < "+ isa::utils::toString(conf.getNrItemsPerBlock()) + " - " + isa::utils::toString(conf.getNrItemsPerBlock() / 2) + ") || (get_local_id(0) < " + isa::utils::toString(conf.getNrItemsPerBlock()) + " / 2) ) {\n";
	}
	*code += "temp = tempStorage[(get_local_id(0) * " + isa::utils::toString(conf.getNrItemsPerBlock()) + ") + localItem];\n"
	"tempStorage[(get_local_id(0) * " + isa::utils::toString(conf.getNrItemsPerBlock()) + ") + localItem] = tempStorage[(localItem * " + isa::utils::toString(conf.getNrItemsPerBlock()) + ") + get_local_id(0)];\n"
	"tempStorage[(localItem * " + isa::utils::toString(conf.getNrItemsPerBlock()) + ") + get_local_id(0)] = temp;\n"
	"}\n"
	"}\n";
	if ( conf.getNrItemsPerBlock() > vector ) {
		*code += "barrier(CLK_LOCAL_MEM_FENCE);\n";
	}
	// Store output
	*code += "for ( unsigned int n = 0; n < " + isa::utils::toString(conf.getNrItemsPerBlock()) + "; n++ ) {\n"
	"if ( baseN + n < " + isa::utils::toString(N) + " ) {\n"
	"output[((baseN + n) * " + isa::utils::toString(isa::utils::pad(M, padding)) + ") + (baseM + get_local_id(0))] = tempStorage[(n * " + isa::utils::toString(conf.getNrItemsPerBlock()) + ") + get_local_id(0)];"
	"}\n"
	"}\n"
	"}\n";
  // End kernel's template

  return code;
}

} // OpenCL
} // isa

