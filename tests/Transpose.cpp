//
// Copyright (C) 2013
// Alessio Sclocco <a.sclocco@vu.nl>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

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
using isa::utils::same;
using isa::utils::pad;
#include <Transpose.hpp>
using isa::OpenCL::Transpose;

typedef unsigned int dataType;
const string typeName("unsigned int");
const unsigned int padding = 32;
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
			inputData->setHostDataItem((m * pad(N, padding)) + n, n);
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
