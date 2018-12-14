__kernel void distortion(__global float* luma, signed int quant) {
	unsigned int idx = get_global_id(0);
	float px = luma[idx];
	if (px <= -128.f) {
		px -= quant;
    } else if (px >= 128.f) {
		px += quant;
    }
	//set pixel
	luma[idx] = px;
}