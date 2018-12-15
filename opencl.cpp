#include "opencl.h"
#include <iostream>
#include <fstream>
#include <array>



using namespace std;
using namespace cl;

void opencl::setOpenCL() {
	// Get the platforms
	Platform::get(&platforms);
	// Assume only one platform.  Get GPU devices.
	
	platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

	// Just to test, print out device 0 name
	//cout << devices[0].getInfo<CL_DEVICE_NAME>() << endl;
	
	// Create a context with these devices
	context = std::move(cl::Context(devices));

	// Create a command queue for device 0
	queue = std::move(cl::CommandQueue(context, devices[0]/*, CL_QUEUE_PROFILING_ENABLE*/));//debug cl_queue_profiling
	setProgram();
}

void opencl::setProgram() {
	std::ifstream file("group.cl");
	if (!file)
		printf("error opening kernel file");
	std::string code(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));

	cl::Program::Sources source(1, std::make_pair(code.c_str(), code.length() + 1));
	program = std::move(cl::Program(context, source));
	program.build(devices);
}
