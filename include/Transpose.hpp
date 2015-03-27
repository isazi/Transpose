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
#include <map>
#include <fstream>

#include <utils.hpp>


#ifndef TRANSPOSE_HPP
#define TRANSPOSE_HPP

namespace isa {
namespace OpenCL {

class transposeConf {
public:
  transposeConf();
  ~transposeConf();

  // Get
  unsigned int getNrItemsPerBlock() const;
  // Set
  void setNrItemsPerBlock(unsigned int items);
  // utils
  std::string print() const;

private:
  unsigned int nrItemsPerBlock;
};

typedef std::map< std::string, std::map< unsigned int, isa::OpenCL::transposeConf > > tunedTransposeConf;

// Sequential transpose
template< typename T > void transpose(const unsigned int M, const unsigned int N, const unsigned int padding, std::vector< T > & input, std::vector< T > & output);
// OpenCL transpose
std::string * getTransposeOpenCL(const transposeConf & conf, const unsigned int M, const unsigned int N, const unsigned int padding, const unsigned int vector, std::string typeName);
// Read configuration files
void readTunedTransposeConf(tunedTransposeConf & tunedTranspose, const std::string & transposeFilename);


// Implementations

inline unsigned int transposeConf::getNrItemsPerBlock() const {
  return nrItemsPerBlock;
}

inline void transposeConf::setNrItemsPerBlock(unsigned int items) {
  nrItemsPerBlock = items;
}

template< typename T > void transpose(const unsigned int M, const unsigned int N, const unsigned int padding, std::vector< T > & input, std::vector< T > & output) {
  for ( unsigned int i = 0; i < M; i++ ) {
    for ( unsigned int j = 0; j < N; j++ ) {
      output[(j * isa::utils::pad(M, padding)) + i] = input[(i * isa::utils::pad(N, padding)) + j];
    }
  }
}

} // OpenCl
} // isa

#endif // TRANSPOSE_HPP
