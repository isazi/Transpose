// Copyright 2013 Alessio Sclocco <a.sclocco@vu.nl>
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

#include <Kernel.hpp>
using isa::OpenCL::Kernel;
#include <CLData.hpp>
using isa::OpenCL::CLData;
#include <Exceptions.hpp>
using isa::Exceptions::OpenCLError;
#include <utils.hpp>
using isa::utils::toStringValue;
using isa::utils::giga;
using isa::utils::pad;


#ifndef TRANSPOSE_HPP
#define TRANSPOSE_HPP

namespace isa {
namespace OpenCL {

// OpenCL transpose
template< typename T > class Transpose : public Kernel< T > {
public:
	Transpose(string name, string dataType);

	void generateCode() throw (OpenCLError);
	void operator()(CLData< T > * input, CLData< T > * output) throw (OpenCLError);

	inline void setNrThreadsPerBlock(unsigned int threads);

	inline void setDimensions(unsigned int M, unsigned int N);
	inline void setPaddingFactor(unsigned int factor);
	inline void setVectorWidth(unsigned int width);

private:
	unsigned int nrThreadsPerBlock;
	cl::NDRange globalSize;
	cl::NDRange localSize;

	unsigned int M;
	unsigned int N;
	unsigned int padding;
	unsigned int vectorWidth;
};


// Implementation
template< typename T > Transpose< T >::Transpose(string name, string dataType) : Kernel< T >(name, dataType), nrThreadsPerBlock(0), globalSize(cl::NDRange(1, 1, 1)), localSize(cl::NDRange(1, 1, 1)), M(0), N(0), padding(0), vectorWidth(0) {}

template< typename T > void Transpose< T >::generateCode() throw (OpenCLError) {
	// Begin kernel's template
	string localElements_s = toStringValue< unsigned int >(nrThreadsPerBlock * nrThreadsPerBlock);
	string nrThreadsPerBlock_s = toStringValue< unsigned int >(nrThreadsPerBlock);
	string paddedM_s = toStringValue< unsigned int >(pad(M, padding));
	string M_s = toStringValue< unsigned int >(M);
	string paddedN_s = toStringValue< unsigned int >(pad(N, padding));
	string N_s = toStringValue< unsigned int >(N);

	delete this->code;
	this->code = new string();
	*(this->code) = "__kernel void " + this->name + "(__global const " + this->dataType + " * const restrict input, __global " + this->dataType + " * const restrict output) {\n"
	"const unsigned int baseM = get_group_id(0) * " + nrThreadsPerBlock_s + ";\n"
	"const unsigned int baseN = get_group_id(1) * " + nrThreadsPerBlock_s + ";\n"
	"__local "+ this->dataType + " tempStorage[" + localElements_s + "];"
	"\n"
	// Load input
	"for ( unsigned int m = 0; m < " + nrThreadsPerBlock_s + "; m++ ) {\n"
	"if ( baseN + get_local_id(0) < " + N_s + " ) {\n"
	"tempStorage[(m * " + nrThreadsPerBlock_s + ") + get_local_id(0)] = input[((baseM + m) * " + paddedN_s + ") + (baseN + get_local_id(0))];\n"
	"}\n"
	"}\n";
	if ( nrThreadsPerBlock > vectorWidth ) {
		*(this->code) += "barrier(CLK_LOCAL_MEM_FENCE);\n";
	}
	// Local in-place transpose
	*(this->code) += "for ( unsigned int i = 1; i <= " + nrThreadsPerBlock_s + " / 2; i++ ) {\n"
	"unsigned int localItem = (get_local_id(0) + i) % " + nrThreadsPerBlock_s + ";\n"
	+ this->dataType + " temp = 0;\n";
	if ( nrThreadsPerBlock == vectorWidth ) {
		*(this->code) += "if ( (i < "+ nrThreadsPerBlock_s + ") || (get_local_id(0) < " + nrThreadsPerBlock_s + " / 2) ) {\n";
	} else {
		*(this->code) += "if ( (i < "+ nrThreadsPerBlock_s + " - " + toStringValue< unsigned int >(nrThreadsPerBlock / 2) + ") || (get_local_id(0) < " + nrThreadsPerBlock_s + " / 2) ) {\n";
	}
	*(this->code) += "temp = tempStorage[(get_local_id(0) * " + nrThreadsPerBlock_s + ") + localItem];\n"
	"tempStorage[(get_local_id(0) * " + nrThreadsPerBlock_s + ") + localItem] = tempStorage[(localItem * " + nrThreadsPerBlock_s + ") + get_local_id(0)];\n"
	"tempStorage[(localItem * " + nrThreadsPerBlock_s + ") + get_local_id(0)] = temp;\n"
	"}\n"
	"}\n";
	if ( nrThreadsPerBlock > vectorWidth ) {
		*(this->code) += "barrier(CLK_LOCAL_MEM_FENCE);\n";
	}
	// Store output
	*(this->code) += "for ( unsigned int n = 0; n < " + nrThreadsPerBlock_s + "; n++ ) {\n"
	"if ( baseN + n < " + N_s + " ) {\n"
	"output[((baseN + n) * " + paddedM_s + ") + (baseM + get_local_id(0))] = tempStorage[(n * " + nrThreadsPerBlock_s + ") + get_local_id(0)];"
	"}\n"
	"}\n"
	"}\n";
	// End kernel's template

	globalSize = cl::NDRange(M, ceil(static_cast< double >(N) / nrThreadsPerBlock));
	localSize = cl::NDRange(nrThreadsPerBlock, 1);

	this->gb = giga(static_cast< long long unsigned int >(M) * N * 2 * sizeof(T));

	this->compile();
}

template< typename T > void Transpose< T >::operator()(CLData< T > * input, CLData< T > * output) throw (OpenCLError) {
	this->setArgument(0, *(input->getDeviceData()));
	this->setArgument(1, *(output->getDeviceData()));

	this->run(globalSize, localSize);
}

template< typename T > inline void Transpose< T >::setNrThreadsPerBlock(unsigned int threads) {
	nrThreadsPerBlock = threads;
}

template< typename T > inline void Transpose< T >::setDimensions(unsigned int M, unsigned int N) {
	this->M = M;
	this->N = N;
}

template< typename T > inline void Transpose< T >::setPaddingFactor(unsigned int factor) {
	padding = factor;
}

template< typename T > inline void Transpose< T >::setVectorWidth(unsigned int width) {
	vectorWidth = width;
}

} // OpenCl
} // isa

#endif // TRANSPOSE_HPP
