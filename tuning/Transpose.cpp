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
#include <algorithm>

#include <ArgumentList.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <Transpose.hpp>
#include <utils.hpp>
#include <Timer.hpp>
#include <Stats.hpp>

typedef float dataType;
string typeName("float");


int main(int argc, char * argv[]) {
	unsigned int nrIterations = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	unsigned int minThreads = 0;
	unsigned int maxThreads = 0;
  unsigned int padding = 0;
  unsigned int vector = 0;
  unsigned int M = 0;
  unsigned int N = 0;

	try {
    isa::utils::ArgumentList args(argc, argv);

		nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    padding = args.getSwitchArgument< unsigned int >("-padding");
    vector = args.getSwitchArgument< unsigned int >("-vector");
    M = args.getSwitchArgument< unsigned int >("-M");
    N = args.getSwitchArgument< unsigned int >("-N");
		minThreads = args.getSwitchArgument< unsigned int >("-min_threads");
		maxThreads = args.getSwitchArgument< unsigned int >("-max_threads");
	} catch ( isa::utils::EmptyCommandLine &err ) {
		std::cerr << argv[0] << " -iterations ... -opencl_platform ... -opencl_device ... -padding ... - vector ... -M ... -N ... -min_threads ... -max_threads ..." << std::endl;
		return 1;
	} catch ( std::exception &err ) {
		std::cerr << err.what() << std::endl;
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
  try {
    input_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, input.size() * sizeof(dataType), 0, 0);
    output_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, output.size() * sizeof(dataType), 0, 0);
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error allocating memory: " << isa::utils::toString(err.err()) << "." << std::endl;
    return 1;
  }

	srand(time(0));
  for ( unsigned int m = 0; m < M; m++ ) {
    for ( unsigned int n = 0; n < N; n++ ) {
      input[(m * isa::utils::pad(N, padding)) + n] = static_cast< dataType >(rand() % 10);
    }
	}

  // Copy data structures to device
  try {
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(input_d, CL_FALSE, 0, input.size() * sizeof(dataType), reinterpret_cast< void * >(input.data()));
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error H2D transfer: " << isa::utils::toString(err.err()) << "." << std::endl;
    return 1;
  }

	// Find the parameters
	std::vector< unsigned int > threads;
	for ( unsigned int nrThreads = minThreads; nrThreads <= maxThreads; nrThreads += minThreads ) {
		if ( (M % nrThreads) == 0 ) {
			threads.push_back(nrThreads);
		}
	}

	std::cout << std::fixed << std::endl;
	std::cout << "# M N nrThreads GB/s err time err" << std::endl << std::endl;

  for ( std::vector< unsigned int >::iterator nrThreads = threads.begin(); nrThreads != threads.end(); ++nrThreads ) {
    // Generate kernel
    double gbs = isa::utils::giga(static_cast< long long unsigned int >(M) * N * 2 * sizeof(dataType));
    isa::utils::Timer timer("Kernel Timer");
    isa::utils::Stats< double > stats;
    cl::Event event;
    cl::Kernel * kernel;
    std::string * code = isa::OpenCL::getTransposeOpenCL(*nrThreads, M, N, padding, vector, typeName);

    try {
      kernel = isa::OpenCL::compile("transpose", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
    } catch ( isa::OpenCL::OpenCLError &err ) {
      std::cerr << err.what() << std::endl;
      return 1;
    }
    cl::NDRange global(M, std::ceil(static_cast< double >(N) / *nrThreads));
    cl::NDRange local(*nrThreads, 1);

    kernel->setArg(0, input_d);
    kernel->setArg(1, output_d);

    // Warm-up run
    try {
      clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
      event.wait();
    } catch ( cl::Error &err ) {
      std::cerr << "OpenCL error kernel execution: " << isa::utils::toString(err.err()) << "." << std::endl;
      continue;
    }
    // Tuning runs
    try {
      for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
        timer.start();
        clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
        event.wait();
        timer.stop();
        stats.addElement(gbs / timer.getLastRunTime());
      }
    } catch ( cl::Error &err ) {
      std::cerr << "OpenCL error kernel execution: " << isa::utils::toString(err.err()) << "." << std::endl;
      continue;
    }

    std::cout << M << " " << N << " " << *nrThreads << " " << std::setprecision(3) << stats.getAverage() << " " << stats.getStdDev() << " " << std::setprecision(6) << timer.getAverageTime() << " " << timer.getStdDev() << std::endl;
  }

	std::cout << std::endl;

	return 0;
}

