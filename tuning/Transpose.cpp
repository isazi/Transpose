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
using std::numeric_limits;

#include <ArgumentList.hpp>
using isa::utils::ArgumentList;
#include <Observation.hpp>
using AstroData::Observation;
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
const unsigned int maxThreadsPerBlock = 1024;
const unsigned int padding = 32;


int main(int argc, char * argv[]) {
	unsigned int lowerNrThreads = 0;
	unsigned int nrIterations = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	unsigned int M = 0;
	unsigned int N = 0;
	CLData< dataType > * inputData = new CLData< dataType >("InputData", true);
	CLData< dataType > * transposedData = new CLData<dataType >("TranposedData", true);


	try {
		ArgumentList args(argc, argv);

		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
		nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
		lowerNrThreads = args.getSwitchArgument< unsigned int >("-lnt");
		M = args.getSwitchArgument< unsigned int >("-M");
		N = args.getSwitchArgument< unsigned int >("-N");
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
	inputData->allocateHostData(M * pad(N));
	inputData->blankHostData()
	transposedData->allocateHostData(N * pad(M));
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
	for ( unsigned int threadsPerBlock = lowerNrThreads; threadsPerBlock <= maxThreadsPerBlock; threadsPerBlock++ ) {
		if ( M % threadsPerBlock == 0 ) {
			configurations.push_back(threadsPerBlock);
		}
	}

	for ( vector< unsigned int >::const_iterator configuration = configurations.begin(); configuration != configurations.end(); configuration++ ) {
		double Acur = 0.0;
		double Aold = 0.0;
		double Vcur = 0.0;
		double Vold = 0.0;

		try {
			// Generate kernel
			Transpose< dataType > clTranspose("clTranspose", typeName);
			clTranspose.bindOpenCL(clContext, &(clDevices->at(clDeviceID)), &((clQueues->at(clDeviceID)).at(0)));
			clTranspose.setDimensions(M, N);
			clTranspose.setNrThreadsPerBlock(*configuration);
			clTranspose.generateCode();

			clTranspose(inputData, transposedData);
			(clTranspose.getTimer()).reset();
			
			for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
				clTranspose(inputData, transposedData);
				
				if ( iteration == 0 ) {
					Acur = clTranspose.getGB() / clTranspose.getTimer().getLastRunTime();
				} else {
					Aold = Acur;
					Vold = Vcur;

					Acur = Aold + (((clTranspose.getGB() / clTranspose.getTimer().getLastRunTime()) - Aold) / (iteration + 1));
					Vcur = Vold + (((clTranspose.getGB() / clTranspose.getTimer().getLastRunTime()) - Aold) * ((clTranspose.getGB() / clTranspose.getTimer().getLastRunTime()) - Acur));
				}
			}
			Vcur = sqrt(Vcur / nrIterations);

			cout << M << " " << N << " " << *configuration << " " << setprecision(3) << Acur << " " << Vcur << " " << setprecision(6) << clTranspose.getTimer().getAverageTime() << " " << clTranspose.getTimer().getStdDev() << endl;
		} catch ( OpenCLError err ) {
			cerr << err.what() << endl;
			continue;
		}
	}

	cout << endl;

	return 0;
}
