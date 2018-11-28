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
	cout << devices[0].getInfo<CL_DEVICE_NAME>() << endl;
	
	// Create a context with these devices
	context = std::move(cl::Context(devices));

	// Create a command queue for device 0
	queue = std::move(cl::CommandQueue(context, devices[0]));
}
constexpr int ELEMENTS = 100;
constexpr int DATA_SIZE = ELEMENTS * sizeof(int);

void opencl::test() {
	
	//use array not vector
	array<int, ELEMENTS> A;
	array<int, ELEMENTS> B;
	array<int, ELEMENTS> C;
	for (std::size_t i = 0; i < ELEMENTS; ++i)
		A[i] = B[i] = i;

	// Create the buffers
	Buffer bufA(context, CL_MEM_READ_ONLY, DATA_SIZE);
	Buffer bufB(context, CL_MEM_READ_ONLY, DATA_SIZE);
	Buffer bufC(context, CL_MEM_WRITE_ONLY, DATA_SIZE);

	// Copy data to the GPU
	queue.enqueueWriteBuffer(bufA, CL_TRUE, 0, DATA_SIZE, &A);
	queue.enqueueWriteBuffer(bufB, CL_TRUE, 0, DATA_SIZE, &B);

	// Read in kernel source
	ifstream file("test.cl");
	if (!file) {
		cout << "error in reading file kernel" << endl;
		return;
	}
	string code(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));

	// Create program
	Program::Sources source(1, make_pair(code.c_str(), code.length() + 1));
	Program program(context, source);

	// Build program for devices
	program.build(devices);

	// Create the kernel
	Kernel vecadd_kernel(program, "vecadd");
	// Set kernel arguments
	vecadd_kernel.setArg(0, bufA);
	vecadd_kernel.setArg(1, bufB);
	vecadd_kernel.setArg(2, bufC);

	// Execute kernel
	NDRange global(ELEMENTS);
	NDRange local(25);
	queue.enqueueNDRangeKernel(vecadd_kernel, NullRange, global, local);

	// Copy result back.
	queue.enqueueReadBuffer(bufC, CL_TRUE, 0, DATA_SIZE, &C);
	file.close();
	
	
		
	
	
	return;
}

void test2() {
	// Initialise memory
	array<int, ELEMENTS> A;
	array<int, ELEMENTS> B;
	array<int, ELEMENTS> C;
	for (std::size_t i = 0; i < ELEMENTS; ++i)
		A[i] = B[i] = i;

	
		// Get the platforms
		vector<Platform> platforms;
		Platform::get(&platforms);

		// Assume only one platform.  Get GPU devices.
		vector<Device> devices;
		platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

		// Just to test, print out device 0 name
		cout << devices[0].getInfo<CL_DEVICE_NAME>() << endl;

		// Create a context with these devices
		Context context(devices);

		// Create a command queue for device 0
		CommandQueue queue(context, devices[0]);

		// Create the buffers
		Buffer bufA(context, CL_MEM_READ_ONLY, DATA_SIZE);
		Buffer bufB(context, CL_MEM_READ_ONLY, DATA_SIZE);
		Buffer bufC(context, CL_MEM_WRITE_ONLY, DATA_SIZE);

		// Copy data to the GPU
		queue.enqueueWriteBuffer(bufA, CL_TRUE, 0, DATA_SIZE, &A);
		queue.enqueueWriteBuffer(bufB, CL_TRUE, 0, DATA_SIZE, &B);

		// Read in kernel source
		ifstream file("test.cl");
		if (!file)
			cout << "error reading file" << endl;
		string code(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));

		// Create program
		Program::Sources source(1, make_pair(code.c_str(), code.length() + 1));
		Program program(context, source);

		// Build program for devices
		program.build(devices);

		// Create the kernel
		Kernel vecadd_kernel(program, "vecadd");

		// Set kernel arguments
		vecadd_kernel.setArg(0, bufA);
		vecadd_kernel.setArg(1, bufB);
		vecadd_kernel.setArg(2, bufC);

		// Execute kernel
		NDRange global(ELEMENTS);
		NDRange local(25);
		queue.enqueueNDRangeKernel(vecadd_kernel, NullRange, global, local);

		// Copy result back.
		queue.enqueueReadBuffer(bufC, CL_TRUE, 0, DATA_SIZE, &C);

		// Test that the results are correct
		for (int i = 0; i < ELEMENTS; ++i)
			/*if (C[i] != i + i)
				cout << "Error: " << i << endl;*/
			cout << C[i] << endl;

		cout << "Finished" << endl;
	
}