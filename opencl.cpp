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

	// Create a context with these devices
	context = std::move(cl::Context(devices));

	// Create a command queue for device 0
	queue = std::move(cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE));//debug cl_queue_profiling
}