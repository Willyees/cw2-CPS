__kernel void subsample2x2(__global float* luma, __global float* blue, __global float* red, int m_x, __global float* blue_buff, __global float* red_buff) {
	
	int idx = get_global_id(0);
	int additional = m_x * ((idx * 2) / m_x);
	float a = 129-fabs(luma[idx * 2 + additional]);
    float b = 129-fabs(luma[idx * 2 + 1 + additional]);
    float c = 129-fabs(luma[idx * 2 + m_x + additional]);
    float d = 129-fabs(luma[idx * 2 + m_x + 1 + additional]);
	
	blue_buff[idx] = (blue[idx * 2 + additional] * a + blue[idx * 2 + 1 + additional] * b + blue[idx * 2 + m_x + additional] * c + blue[idx * 2 + m_x + 1 + additional] * d) / (a + b + c + d);
	red_buff[idx] = (red[idx * 2 + additional] * a + red[idx * 2 + 1 + additional] * b + red[idx * 2 + m_x + additional] * c + red[idx * 2 + m_x + 1 + additional] * d) / (a + b + c + d);
	//printf("id: %d blue %f red %f\n", idx, blue_buff[idx], red_buff[idx]);
} 