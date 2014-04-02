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

#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>
using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::exception;
using std::ofstream;
using std::fixed;
using std::setprecision;
using std::numeric_limits;

#include <ArgumentList.hpp>
using isa::utils::ArgumentList;
#include <InitializeOpenCL.hpp>
using isa::OpenCL::initializeOpenCL;
#include <CLData.hpp>
using isa::OpenCL::CLData;
#include <utils.hpp>
using isa::utils::toStringValue;
using isa::utils::pad;
#include <Timer.hpp>
using isa::utils::Timer;
#include <Transpose.hpp>
using isa::OpenCL::Transpose;

typedef float dataType;
const string typeName("float");


int main(int argc, char * argv[]) {
	unsigned int lowerNrThreads = 0;
	unsigned int nrIterations = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	unsigned int padding = 0;
	unsigned int vectorWidth = 32;
	unsigned int maxThreadsPerBlock = 0;
	unsigned int M = 0;
	unsigned int N = 0;
	CLData< dataType > * inputData = new CLData< dataType >("InputData", true);
	CLData< dataType > * transposedData = new CLData<dataType >("TranposedData", true);


	try {
		ArgumentList args(argc, argv);

		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
		nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
		padding = args.getSwitchArgument< unsigned int >("-padding");
		vectorWidth = args.getSwitchArgument< unsigned int >("-vector");
		lowerNrThreads = args.getSwitchArgument< unsigned int >("-lnt");
		maxThreadsPerBlock = args.getSwitchArgument< unsigned int >("-mnt");
		M = args.getSwitchArgument< unsigned int >("-M");
		N = args.getSwitchArgument< unsigned int >("-N");
	} catch ( EmptyCommandLine err ) {
		cerr << argv[0] << " -iterations ... -opencl_platform ... -opencl_device ... -padding ... -vector ... -lnt ... -mnt ... -M ... -N ..." << endl;
		return 1;
	} catch ( exception & err ) {
		cerr << err.what() << endl;
		return 1;
	}

	cl::Context * clContext = new cl::Context();
	vector< cl::Platform > * clPlatforms = new vector< cl::Platform >();
	vector< cl::Device > * clDevices = new vector< cl::Device >();
	vector< vector< cl::CommandQueue > > * clQueues = new vector< vector < cl::CommandQueue > >();

	initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

	cout << fixed << endl;
	cout << "# M N nrThreadPerBlock GB/s err time err" << endl << endl;

	// Allocate memory
	inputData->allocateHostData(M * pad(N, padding));
	inputData->blankHostData();
	transposedData->allocateHostData(N * pad(M, padding));
	transposedData->blankHostData();

	inputData->setCLContext(clContext);
	inputData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	transposedData->setCLContext(clContext);
	transposedData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));

	try {
		inputData->setDeviceReadOnly();
		inputData->allocateDeviceData();
		inputData->copyHostToDevice();
		transposedData->setDeviceWriteOnly();
		transposedData->allocateDeviceData();
		transposedData->copyHostToDevice();
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
	}

	// Find the parameters
	vector< unsigned int > configurations;
	for ( unsigned int threadsPerBlock = lowerNrThreads; threadsPerBlock <= maxThreadsPerBlock; threadsPerBlock += vectorWidth ) {
		if ( M % threadsPerBlock == 0 ) {
			configurations.push_back(threadsPerBlock);
		}
	}

	for ( vector< unsigned int >::const_iterator configuration = configurations.begin(); configuration != configurations.end(); configuration++ ) {
		try {
			// Generate kernel
			Transpose< dataType > clTranspose("clTranspose", typeName);
			clTranspose.bindOpenCL(clContext, &(clDevices->at(clDeviceID)), &((clQueues->at(clDeviceID)).at(0)));
			clTranspose.setDimensions(M, N);
			clTranspose.setPaddingFactor(padding);
			clTranspose.setVectorWidth(vectorWidth);
			clTranspose.setNrThreadsPerBlock(*configuration);
			clTranspose.generateCode();

			clTranspose(inputData, transposedData);
			(clTranspose.getTimer()).reset();
			clTranspose.resetStats();

			for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
				clTranspose(inputData, transposedData);
			}

			cout << M << " " << N << " " << *configuration << " " << setprecision(3) << clTranspose.getGBs() << " " << clTranspose.getGBsErr() << " " << setprecision(6) << clTranspose.getTimer().getAverageTime() << " " << clTranspose.getTimer().getStdDev() << endl;
		} catch ( OpenCLError err ) {
			cerr << err.what() << endl;
			continue;
		}
	}

	cout << endl;

	return 0;
}
