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
  unsigned int M = 0;
  unsigned int N = 0;
	long long unsigned int wrongItems = 0;

	try {
    isa::utils::ArgumentList args(argc, argv);
    print = args.getSwitch("-print");
    M = args.getSwitchArgument< unsigned int >("-M");
    N = args.getSwitchArgument< unsigned int >("-N");
	} catch  ( isa::Exceptions::SwitchNotFound &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }catch ( std::exception &err ) {
    std::cerr << "Usage: " << argv[0] << " -M ... -N ..." << std::endl;
		return 1;
	}

	// Allocate memory
  std::vector< float > input = std::vector< float >(M * isa::utils::pad(N, padding));
  std::vector< float > output = std::vector< float >(N * isa::utils::pad(M, padding));
  std::vector< float > output_c = std::vector< float >(N * isa::utils::pad(M, padding));

	srand(time(NULL));
  for ( unsigned int m = 0; m < M; m++ ) {
    for ( unsigned int n = 0; n < N; n++ ) {
      input[(m * isa::utils::pad(N, padding)) + n] = static_cast< float >(rand() % 10);
    }
	}

  // Run AVX kernel and CPU control
  isa::OpenCL::transposeAVX(M, N, input.data(), output.data());
  isa::OpenCL::transpose(M, N, padding, input, output_c);

  for ( unsigned int n = 0; n < N; n++ ) {
    for ( unsigned int m = 0; m < M; m++ ) {
      if ( ! isa::utils::same(output_c[(n * isa::utils::pad(M, padding)) + m], output[(n * isa::utils::pad(M, padding)) + m]) ) {
        wrongItems++;
      }
    }
  }

  if ( wrongItems > 0 ) {
    std::cout << "Wrong samples: " << wrongItems << " (" << (wrongItems * 100.0) / (static_cast< long long unsigned int >(M) * N) << "%)." << std::endl;
  } else {
    std::cout << "TEST PASSED." << std::endl;
  }

	return 0;
}

