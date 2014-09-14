#include <stdlib.h>
#include <stdio.h>



void make_kernel(char* kernel, int filter_length)
{
	
	printf("inside \n");
/*	kernel = "__kernel void filter(__global int* input, __global int* output, __global int* filtersettings)"
		"{"
		"int id = get_global_id(0);"
		"int i = 0;"
		"int init_a = 0;"
		"int init_b = 0;"
		"int firstder[129];"
		"int e = 0;"
		"int filter_length = filtersettings[0];"
		"int old;"

		"int secondder_a;" 
		"int secondder_b;"


		"for( i = 0; i < filter_length; i ++)"
		"{"
		"init_a = init_a + input[id + i];"
		"init_b = init_b + input[id + filter_length];"

		"}"

		"firstder[0] = init_b - init_a;"

		"for(i = 0; i < (filter_length << 1); i++)"
		"{"
		"firstder[i+1] = firstder[i] + input[id+i] - (input[id + filter_length+i] << 1) + input[id + (filter_length << 1)+i];"
		"}"

		"init_b = 0;"
		"init_a = 0;"     

		"for(i = 0; i < filter_length; i++)"
		"{"
		"init_a = init_a + firstder[i];"
		"init_b = init_b + firstder[i  + filter_length];"

		"}"

		"secondder_a  = init_b - init_a;"

		"init_b = 0;"
		"init_a = 0;"

		"secondder_b = secondder_a + firstder[0] - ( firstder[filter_length] << 1) + firstder[filter_length << 1];"

		"if(secondder_a >= 0 && secondder_b < 0)"
		"{"

		"e = firstder[filter_length]/filter_length;"
		"if(e >= filtersettings[1])"
		"{"
		"old = atomic_add(&filtersettings[3],1);"
		"output[ (old << 1)] = e;"
		"output[ (old << 1) + 1] = id;"
		"}"
		"}";*/

	printf("first time was okay \n");
	
	sprintf(kernel, "__kernel void filter(__global int* input, __global int* output, __global int* filtersettings)"
			"{"
			"int id = get_global_id(0);"
			"int i = 0;"
			"int init_a = 0;"
			"int init_b = 0;"
			"int firstder[%d];"
			"int e = 0;"
			"int filter_length = filtersettings[0];"
			"int old;"

			"int secondder_a;" 
			"int secondder_b;"


			"for( i = 0; i < filter_length; i ++)"
			"{"
			"init_a = init_a + input[id + i];"
			"init_b = init_b + input[id + filter_length];"

			"}"

			"firstder[0] = init_b - init_a;"

			"for(i = 0; i < (filter_length << 1); i++)"
			"{"
			"firstder[i+1] = firstder[i] + input[id+i] - (input[id + filter_length+i] << 1) + input[id + (filter_length << 1)+i];"
			"}"

			"init_b = 0;"
			"init_a = 0;"     

			"for(i = 0; i < filter_length; i++)"
			"{"
			"init_a = init_a + firstder[i];"
			"init_b = init_b + firstder[i  + filter_length];"

			"}"

			"secondder_a  = init_b - init_a;"

			"init_b = 0;"
			"init_a = 0;"

			"secondder_b = secondder_a + firstder[0] - ( firstder[filter_length] << 1) + firstder[filter_length << 1];"

			"if(secondder_a >= 0 && secondder_b < 0)"
			"{"

			"e = firstder[filter_length]/filter_length;"
			"if(e >= filtersettings[1])"
			"{"
			"old = atomic_add(&filtersettings[3],1);"
			"output[ (old << 1)] = e;"
			"output[ (old << 1) + 1] = id;"
			"}"
			"}"
			"}", (2*filter_length +1));
		printf("second time not \n");


}






