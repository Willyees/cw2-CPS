#pragma once
#include <vector>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

class opencl {
public:
	void setOpenCL();
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	cl::Context context;
	cl::CommandQueue queue;
	cl::Program program;
	
private:
	
	void setProgram();
};

