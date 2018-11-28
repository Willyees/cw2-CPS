#pragma once
#include <vector>

#include <CL/cl.hpp>

class opencl {
public:
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	cl::Context context;
	cl::CommandQueue queue;
	
	void setOpenCL();
	void test();
};

void test2();