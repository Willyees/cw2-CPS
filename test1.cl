__kernel void image_test(__global unsigned char* src, __global float* image0,__global float* image1,__global float* image2, int diff, int width){
	
	int idx = get_global_id(0);
		
	image0[idx / width * diff + idx] = (0.299     * src[idx*3 + 0]) + (0.587     * src[idx*3 + 1]) + (0.114     * src[idx*3 + 2])-128.0;
	image1[idx / width * diff + idx] = -(0.168736  * src[idx*3 + 0]) - (0.331264  * src[idx*3+ 1]) + (0.5       * src[idx*3 + 2]);
	image2[idx / width * diff + idx] = (0.5       * src[idx*3 + 0]) - (0.418688  * src[idx*3 + 1]) - (0.081312  * src[idx*3 + 2]);
	//printf("src1: %d src2: %d src3: %d img0: %d img1:%d img2:%d\n", src[idx*3 + 0], src[idx*3 + 1], src[idx*3 + 2], image0[idx], image1[idx], image2[idx]);
	//printf("img0: %f img1: %f img2: %f", (0.299     * src[idx*3 + 0]) + (0.587     * src[idx*3 + 1]) + (0.114     * src[idx*3 + 2])-128.0, -(0.168736  * src[idx*3 + 0]) - (0.331264  * src[idx*3+ 1]) + (0.5       * src[idx*3 + 2]), (0.5       * src[idx*3 + 0]) - (0.418688  * src[idx*3 + 1]) - (0.081312  * src[idx*3 + 2]));
}