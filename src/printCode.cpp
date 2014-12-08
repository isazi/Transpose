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

#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ctime>
#include <cmath>

#include <ArgumentList.hpp>
#include <utils.hpp>
#include <Transpose.hpp>


int main(int argc, char *argv[]) {
  unsigned int nrThreads = 0;
  unsigned int padding = 0;
  unsigned int vector = 0;
  unsigned int M = 0;
  unsigned int N = 0;
  std::string typeName;

	try {
    isa::utils::ArgumentList args(argc, argv);
    typeName = args.getSwitchArgument< std::string >("-type");
    padding = args.getSwitchArgument< unsigned int >("-padding");
    vector = args.getSwitchArgument< unsigned int >("-vector");
    nrThreads = args.getSwitchArgument< unsigned int >("-threads");
    M = args.getSwitchArgument< unsigned int >("-M");
    N = args.getSwitchArgument< unsigned int >("-N");
	} catch  ( isa::utils::SwitchNotFound & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }catch ( std::exception & err ) {
    std::cerr << "Usage: " << argv[0] << "-type ... -padding ... -vector ... -threads ... -M ... -N ..." << std::endl;
		return 1;
	}

  // Generate kernel
  std::string * code = isa::OpenCL::getTransposeOpenCL(nrThreads, M, N, padding, vector, typeName);
  std::cout << *code << std::endl;

	return 0;
}

