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
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <utils.hpp>
#include <Transpose.hpp>

typedef float dataType;
std::string typeName("float");


int main(int argc, char *argv[]) {
  bool printCode = false;
  bool printData = false;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
  unsigned int nrThreads = 0;
  unsigned int padding = 0;
  unsigned int vector = 0;
  unsigned int M = 0;
  unsigned int N = 0;
	long long unsigned int wrongItems = 0;

	try {
    isa::utils::ArgumentList args(argc, argv);
    printCode = args.getSwitch("-print_code");
    printData = args.getSwitch("-print_data");
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    padding = args.getSwitchArgument< unsigned int >("-padding");
    vector = args.getSwitchArgument< unsigned int >("-vector");
    nrThreads = args.getSwitchArgument< unsigned int >("-threads");
    M = args.getSwitchArgument< unsigned int >("-M");
    N = args.getSwitchArgument< unsigned int >("-N");
	} catch  ( isa::utils::SwitchNotFound &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }catch ( std::exception &err ) {
    std::cerr << "Usage: " << argv[0] << " [-print_code] [-print_data] -opencl_platform ... -opencl_device ... -padding ... -vector ... -threads ... -M ... -N ..." << std::endl;
		return 1;
	}

	// Initialize OpenCL
	cl::Context * clContext = new cl::Context();
	std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
	std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
	std::vector< std::vector< cl::CommandQueue > > * clQueues = new std::vector< std::vector < cl::CommandQueue > >();

  isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

	// Allocate memory
  std::vector< dataType > input = std::vector< dataType >(M * isa::utils::pad(N, padding));
  cl::Buffer input_d;
  std::vector< dataType > output = std::vector< dataType >(N * isa::utils::pad(M, padding));
  cl::Buffer output_d;
  std::vector< dataType > output_c = std::vector< dataType >(N * isa::utils::pad(M, padding));
  try {
    input_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, input.size() * sizeof(dataType), NULL, NULL);
    output_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, output.size() * sizeof(dataType), NULL, NULL);
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error allocating memory: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

	srand(time(NULL));
  for ( unsigned int m = 0; m < M; m++ ) {
    for ( unsigned int n = 0; n < N; n++ ) {
      input[(m * isa::utils::pad(N, padding)) + n] = static_cast< dataType >(rand() % 10);
    }
	}

  // Copy data structures to device
  try {
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(input_d, CL_FALSE, 0, input.size() * sizeof(dataType), reinterpret_cast< void * >(input.data()));
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error H2D transfer: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

  // Generate kernel
  cl::Kernel * kernel;
  std::string * code = isa::OpenCL::getTransposeOpenCL(nrThreads, M, N, padding, vector, typeName);
  if ( printCode ) {
    std::cout << *code << std::endl;
  }

  try {
    kernel = isa::OpenCL::compile("transpose", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
  } catch ( isa::OpenCL::OpenCLError &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Run OpenCL kernel and CPU control
  try {
    cl::NDRange global(isa::utils::pad(M, padding), std::ceil(static_cast< double >(isa::utils::pad(N, padding)) / nrThreads));
    cl::NDRange local(nrThreads, 1);

    kernel->setArg(0, input_d);
    kernel->setArg(1, output_d);
    
    if ( printCode ) {
      std::cout << "NDRange global: " << isa::utils::pad(M, padding) << " " << static_cast< unsigned int >(std::ceil(static_cast< double >(isa::utils::pad(N, padding)) / nrThreads)) << std::endl;
      std::cout << "NDRange local: " << nrThreads << " " << 1 << std::endl;
      std::cout << std::endl;
    }
    clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, NULL, NULL);
    isa::OpenCL::transpose(M, N, padding, input, output_c);
    clQueues->at(clDeviceID)[0].enqueueReadBuffer(output_d, CL_TRUE, 0, output.size() * sizeof(dataType), reinterpret_cast< void * >(output.data()));
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

  if ( printData ) {
    for ( unsigned int m = 0; m < M; m++ ) {
      for ( unsigned int n = 0; n < N; n++ ) {
        std::cout << input[(m * isa::utils::pad(N, padding)) + n] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  for ( unsigned int n = 0; n < N; n++ ) {
    for ( unsigned int m = 0; m < M; m++ ) {
      if ( ! isa::utils::same(output_c[(n * isa::utils::pad(M, padding)) + m], output[(n * isa::utils::pad(M, padding)) + m]) ) {
        wrongItems++;
      }
      if ( printData ) {
        std::cout << output[(n * isa::utils::pad(M, padding)) + m] << " ";
      }
    }
    if ( printData ) {
      std::cout << std::endl;
    }
  }
  if ( printData ) {
    std::cout << std::endl;
  }

  if ( wrongItems > 0 ) {
    std::cout << "Wrong samples: " << wrongItems << " (" << (wrongItems * 100.0) / (static_cast< long long unsigned int >(M) * N) << "%)." << std::endl;
  } else {
    std::cout << "TEST PASSED." << std::endl;
  }

	return 0;
}

