signed short round_to_zero(const double j, signed int quant){
	if (j < 0) {
			signed short jtmp = -j + (quant >> 1);
			return (jtmp < quant) ? 0 : ((signed short)(-(jtmp / quant)));
		} else {
			signed short jtmp = j + (quant >> 1);
			return (jtmp < quant) ? 0 : ((signed short)((jtmp / quant)));
		}
}

__kernel void dct_luma(__global float* luma, __global signed short* luma_buf, int width, __global unsigned char* s_zag, __global signed int* quant) {
	int idx = get_global_id(0);
	int idy = get_global_id(1);
	
	//find intial point in the luma array. skipping 8 positions in the x and y coordinate on each kernel
	int init = idx * 8 + idy * (width * 8);
	double z1, z2, z3, z4, z5, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp10, tmp11, tmp12, tmp13;
	double* data_ptr;
	//cant do this. 8x8 second row is under, so it is idx + width_row
	double luma_d[64];
	for(int y = 0; y < 8; y++) {
		for(int x = 0; x < 8; x++){
			luma_d[x + y * 8] = luma[init + y * width + x]; 
			
		}
	}
	
	
	//intial data pointer. Skipping 64 positions for each kernel
    data_ptr = luma_d;
	//printf("%f idx: %d\n", data_ptr[0], idx);
	//work on luma row by row
    for (int c=0; c < 8; c++) {
        tmp0 = data_ptr[0] + data_ptr[7];
        tmp7 = data_ptr[0] - data_ptr[7];
        tmp1 = data_ptr[1] + data_ptr[6];
        tmp6 = data_ptr[1] - data_ptr[6];
        tmp2 = data_ptr[2] + data_ptr[5];
        tmp5 = data_ptr[2] - data_ptr[5];
        tmp3 = data_ptr[3] + data_ptr[4];
        tmp4 = data_ptr[3] - data_ptr[4];
        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;
        data_ptr[0] = tmp10 + tmp11;
        data_ptr[4] = tmp10 - tmp11;
        z1 = (tmp12 + tmp13) * 0.541196100;
        data_ptr[2] = z1 + tmp13 * 0.765366865;
        data_ptr[6] = z1 + tmp12 * - 1.847759065;
        z1 = tmp4 + tmp7;
        z2 = tmp5 + tmp6;
        z3 = tmp4 + tmp6;
        z4 = tmp5 + tmp7;
        z5 = (z3 + z4) * 1.175875602;
        tmp4 *= 0.298631336;
        tmp5 *= 2.053119869;
        tmp6 *= 3.072711026;
        tmp7 *= 1.501321110;
        z1 *= -0.899976223;
        z2 *= -2.562915447;
        z3 *= -1.961570560;
        z4 *= -0.390180644;
        z3 += z5;
        z4 += z5;
        data_ptr[7] = tmp4 + z1 + z3;
        data_ptr[5] = tmp5 + z2 + z4;
        data_ptr[3] = tmp6 + z2 + z3;
        data_ptr[1] = tmp7 + z1 + z4;
		//skipping 8 poisitions for each loop
        data_ptr += 8;
    }
	//reset pointer to beginning of this group of luma
    data_ptr = luma_d;
	
	
	//work on luma column by column
    for (int c=0; c < 8; c++) {
        tmp0 = data_ptr[8*0] + data_ptr[8*7];
        tmp7 = data_ptr[8*0] - data_ptr[8*7];
        tmp1 = data_ptr[8*1] + data_ptr[8*6];
        tmp6 = data_ptr[8*1] - data_ptr[8*6];
        tmp2 = data_ptr[8*2] + data_ptr[8*5];
        tmp5 = data_ptr[8*2] - data_ptr[8*5];
        tmp3 = data_ptr[8*3] + data_ptr[8*4];
        tmp4 = data_ptr[8*3] - data_ptr[8*4];
        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;
        data_ptr[8*0] = (tmp10 + tmp11) / 8.0;
        data_ptr[8*4] = (tmp10 - tmp11) / 8.0;
        z1 = (tmp12 + tmp13) * 0.541196100;
        data_ptr[8*2] = (z1 + tmp13 * 0.765366865) / 8.0;
        data_ptr[8*6] = (z1 + tmp12 * -1.847759065) / 8.0;
        z1 = tmp4 + tmp7;
        z2 = tmp5 + tmp6;
        z3 = tmp4 + tmp6;
        z4 = tmp5 + tmp7;
        z5 = (z3 + z4) * 1.175875602;
        tmp4 *= 0.298631336;
        tmp5 *= 2.053119869;
        tmp6 *= 3.072711026;
        tmp7 *= 1.501321110;
        z1 *= -0.899976223;
        z2 *= -2.562915447;
        z3 *= -1.961570560;
        z4 *= -0.390180644;
        z3 += z5;
        z4 += z5;
        data_ptr[8*7] = (tmp4 + z1 + z3) / 8.0;
        data_ptr[8*5] = (tmp5 + z2 + z4) / 8.0;
        data_ptr[8*3] = (tmp6 + z2 + z3) / 8.0;
        data_ptr[8*1] = (tmp7 + z1 + z4) / 8.0;
        data_ptr++;
		
		
    }
	//writing to buffer
	/*for(int y = 0; y < 8; y++) {
		for(int x = 0; x < 8; x++){
			luma_buf[init + y * width + x] = luma_d[x + y * 8]; 
			
		}
	}*/
	int init_buf = 64 * (idx + (width / 8) * idy);
	for (int i = 0; i < 64; i++) {
        luma_buf[init_buf + i] = round_to_zero(luma_d[s_zag[i]], quant[i]);
		
    }
}

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

__kernel void image_test(__global unsigned char* src, __global float* image0,__global float* image1,__global float* image2, int diff, int width){
	
	int idx = get_global_id(0);
		
	image0[idx / width * diff + idx] = (0.299     * src[idx*3 + 0]) + (0.587     * src[idx*3 + 1]) + (0.114     * src[idx*3 + 2])-128.0;
	image1[idx / width * diff + idx] = -(0.168736  * src[idx*3 + 0]) - (0.331264  * src[idx*3+ 1]) + (0.5       * src[idx*3 + 2]);
	image2[idx / width * diff + idx] = (0.5       * src[idx*3 + 0]) - (0.418688  * src[idx*3 + 1]) - (0.081312  * src[idx*3 + 2]);
	//printf("src1: %d src2: %d src3: %d img0: %d img1:%d img2:%d\n", src[idx*3 + 0], src[idx*3 + 1], src[idx*3 + 2], image0[idx], image1[idx], image2[idx]);
	//printf("img0: %f img1: %f img2: %f", (0.299     * src[idx*3 + 0]) + (0.587     * src[idx*3 + 1]) + (0.114     * src[idx*3 + 2])-128.0, -(0.168736  * src[idx*3 + 0]) - (0.331264  * src[idx*3+ 1]) + (0.5       * src[idx*3 + 2]), (0.5       * src[idx*3 + 0]) - (0.418688  * src[idx*3 + 1]) - (0.081312  * src[idx*3 + 2]));
}