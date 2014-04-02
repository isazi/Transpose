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
using std::setw;
using std::numeric_limits;

#include <ArgumentList.hpp>
using isa::utils::ArgumentList;
#include <InitializeOpenCL.hpp>
using isa::OpenCL::initializeOpenCL;
#include <CLData.hpp>
using isa::OpenCL::CLData;
#include <utils.hpp>
using isa::utils::pad;
#include <Transpose.hpp>
using isa::OpenCL::Transpose;

typedef unsigned int dataType;
const string typeName("unsigned int");
const unsigned int padding = 32;
const unsigned int vectorWidth = 32;
const bool DEBUG = false;


int main(int argc, char *argv[]) {
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	unsigned int nrThreads = 0;
	unsigned int M = 0;
	unsigned int N = 0;
	long long unsigned int wrongValues = 0;
	CLData< dataType > * inputData = new CLData< dataType >("InputData", true);
	CLData< dataType > * transposeData = new CLData< dataType >("TransposeData", true);

	try {
		ArgumentList args(argc, argv);

		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
		nrThreads = args.getSwitchArgument< unsigned int >("-threads");
		M = args.getSwitchArgument< unsigned int >("-M");
		N = args.getSwitchArgument< unsigned int >("-N");

	} catch ( exception &err ) {
		cerr << err.what() << endl;
		return 1;
	}

	cl::Context * clContext = new cl::Context();
	vector< cl::Platform > * clPlatforms = new vector< cl::Platform >();
	vector< cl::Device > * clDevices = new vector< cl::Device >();
	vector< vector< cl::CommandQueue > > * clQueues = new vector< vector < cl::CommandQueue > >();

	initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

	// Allocate memory
	inputData->allocateHostData(M * pad(N, padding));
	transposeData->allocateHostData(N * pad(M, padding));

	inputData->setCLContext(clContext);
	inputData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	transposeData->setCLContext(clContext);
	transposeData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));

	try {
		inputData->setDeviceReadOnly();
		inputData->allocateDeviceData();
		transposeData->setDeviceWriteOnly();
		transposeData->allocateDeviceData();
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}

	for ( unsigned int m = 0; m < M; m++ ) {
		for ( unsigned int n = 0; n < N; n++ ) {
			inputData->setHostDataItem((m * pad(N, padding)) + n, (m * N) + n);
			if ( DEBUG ) {
				cout << setw(3) << inputData->getHostDataItem((m * pad(N, padding)) + n) << " ";
			}
		}
		if ( DEBUG ) {
			cout << endl;
		}
	}
	if ( DEBUG ) {
		cout << endl;
	}

	// Test
	try {
		// Generate kernel
		Transpose< dataType > clTranspose("clTranspose", typeName);
		clTranspose.bindOpenCL(clContext, &(clDevices->at(clDeviceID)), &((clQueues->at(clDeviceID)).at(0)));
		clTranspose.setDimensions(M, N);
		clTranspose.setPaddingFactor(padding);
		clTranspose.setVectorWidth(vectorWidth);
		clTranspose.setNrThreadsPerBlock(nrThreads);
		clTranspose.generateCode();

		inputData->copyHostToDevice();
		clTranspose(inputData, transposeData);
		transposeData->copyDeviceToHost();
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}

	// Check
	for ( unsigned int m = 0; m < M; m++ ) {
		for ( unsigned int n = 0; n < N; n++ ) {
			if ( inputData->getHostDataItem((m * pad(N, padding)) + n) != transposeData->getHostDataItem((n * pad(M, padding) + m)) ) {
				wrongValues++;
			}
		}
	}
	if ( DEBUG ) {
		for ( unsigned int n = 0; n < N; n++ ) {
			for ( unsigned int m = 0; m < M; m++ ) {
				cout << setw(3) << transposeData->getHostDataItem((n * pad(M, padding)) + m) << " ";
			}
			cout << endl;
		}
		cout << endl;
	}

	cout << endl;
	cout << "Wrong samples: " << wrongValues << " (" << (wrongValues * 100) / (static_cast< long long unsigned int >(M) * N) << "%)." << endl;
	cout << endl;

	return 0;
}
