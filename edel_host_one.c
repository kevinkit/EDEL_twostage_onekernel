#include <stdlib.h>
#include <stdio.h>
#include <glib.h>
#include "ocl.h"
#include "OCL_gpu.h"
#include "input.h"
#include "kernmaker.h"
#include "settings.h"
#include "writeback.h"


int main(int argc, char* argv[])
{

	//files for debugging
	cl_event event;
	cl_event write,readback;
	cl_ulong write_start, write_end, read_start, read_end;


	cl_ulong queued, submit,start, end;
	double total_time, read_time, write_time;

	int run = 0;
	char timefile[1024] = "time_fil2.txt";

	char readbacktime[1024] = "time_readback_fil2.txt";
	char writetime[1024] = "time_write_fil2.txt";




	FILE *file;
	        FILE *file_4;
        FILE *file_5;



	//Filterlength 1024 does not need anybody...
	for(; run <= 8; run++)
	{

		switch(run)
		{
			case 0: filter_length = 2; break;
			case 1: filter_length = 4; break;
			case 2: filter_length = 8; break;
			case 3: filter_length = 16; break;
			case 4: filter_length = 32; break;
			case 5: filter_length = 64; break;
			case 6: filter_length = 128; break;
			case 7: filter_length = 256; break;
			case 8: filter_length = 512; break;
		}

		sprintf(timefile,"time_fil%d_one_kernel.txt", filter_length);
		                sprintf(readbacktime,"time_readback_fil%d.txt", filter_length);
                sprintf(writetime,"time_write_fil%d.txt",filter_length);


		char  kernel_string[80000];
		const size_t SIZE_execution_bit =(input_length - 4*filter_length+1);
		const size_t SIZE_input_bit = sizeof(gint32)*(input_length+1);
		const size_t SIZE_settings_bit = sizeof(gint32)*4;

		size_t output_bit_on_counts;
		size_t* SIZE_execution_pointer = &SIZE_execution_bit;

		gint32 *filtersettings = (gint32*)malloc(SIZE_settings_bit);
		gint32 *input_vector =(gint32*)malloc(SIZE_input_bit);
		gint32 *positions = (gint32*)malloc(SIZE_input_bit);

		filtersettings[0] = filter_length;
		filtersettings[1] = threshhold;
		filtersettings[2] = input_length;
		filtersettings[3] = 0;
		make_kernel(kernel_string,filter_length); //changes the string in the kernel see kernmaker.h for fixes/setups
		if(debugmode != 0)
		{
			printf("%s", kernel_string);
		}

		//GPU-Init
		ocl = ocl_new(CL_DEVICE_TYPE_GPU,1);
		context = ocl_get_context(ocl);
		queue = ocl_get_cmd_queues (ocl)[0];
		clFinish(queue);


		program = ocl_create_program_from_source(ocl, kernel_string, NULL, &errcode);
		OCL_CHECK_ERROR(errcode);

		filter1 = clCreateKernel(program, "filter", &errcode);
		OCL_CHECK_ERROR(errcode);

		//GPU-Buffer which can be done before the Computation
		settings = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, SIZE_settings_bit, filtersettings, &errcode);
		OCL_CHECK_ERROR(errcode);

		input = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE_input_bit, NULL, &errcode);
		OCL_CHECK_ERROR(errcode);


		if(debugmode != 0)
		{
			srand((unsigned) time( NULL ));
			counter = rand_rects(expected,1,input_length,4*filter_length,4*filter_length,4*filter_length,peak_length,base+peak, input_vector, noise, base, 0,positions);
			if(harddebug != 0)
			{
				for(i = 0; i < input_length;i++)
				{
					if(input_length < 10000)
					{
						printf("input_vector[%i] = %d\n",i,input_vector[i]);
					}
					else
					{
						printf("input_vector[%i] = %d\t",i,input_vector[i]);
					}
				}
			}

			printf("\n counts = %d\n", counter);
			printf("%lu Bits needed for Output-Vector \n", output_bit_on_counts);

		}

		output_bit_on_counts = sizeof(gint32) * safetyfactor * input_length*2*((counter + 2));

		clEnqueueWriteBuffer(queue, input, CL_TRUE, 0, SIZE_input_bit, input_vector, 0, NULL, &write);

		gint32* energy_time = (gint32*)malloc(output_bit_on_counts);


		for(i = 0; i < safetyfactor * (2*counter+2); i++)
		{
			energy_time[i] = -9999;
		}


		output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_bit_on_counts, NULL , &errcode);
		OCL_CHECK_ERROR(errcode);


		OCL_CHECK_ERROR(clSetKernelArg(filter1, 0, sizeof(cl_mem), &input));
		OCL_CHECK_ERROR(clSetKernelArg(filter1, 1, sizeof(cl_mem), &output));
		OCL_CHECK_ERROR(clSetKernelArg(filter1, 2, sizeof(cl_mem), &settings));

		size_t local_item_size;
		size_t global_item_size = (size_t) (input_length - 3*filter_length +1);

		local_item_size = ocl_get_local_size(global_item_size, 2,1);






		if(debugmode != 0)
		{
			printf("local item size = %lu \n %lu", &local_item_size, local_item_size);
			if(local_item_size != 0)
			{
				printf("This works because you divide %lu / %lu \n and this is %lu", global_item_size,local_item_size, global_item_size/local_item_size);
			}
			else
			{
				FILE* attention;
				attention = fopen("filterlengthbad", "a+");
				if(attention == NULL)
				{
					printf("error in opening debug file \n");
					exit(1);
				}
				fprintf(attention, "The filterlength %d is not good for this filter, choose another filterlength ! \n", filter_length);
				fclose(attention);
				printf("There is no way to fit it evenly divided to workgroups, just let OpenCL do it \n");
			}
			if(harddebug != 0)
			{
				getchar();
			}

		}


		if(local_item_size == 0)
		{
			OCL_CHECK_ERROR(clEnqueueNDRangeKernel(queue, filter1, 1, NULL, &global_item_size, NULL, 0, NULL, &event));     
		}
		else
		{
			OCL_CHECK_ERROR(clEnqueueNDRangeKernel(queue, filter1, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &event));
		}




		//local_item_size = NULL;       
		clEnqueueReadBuffer(queue, output, CL_TRUE, 0, output_bit_on_counts, energy_time, 0, NULL, &readback);
		clEnqueueReadBuffer(queue, settings, CL_TRUE, 0, SIZE_settings_bit, filtersettings, 0, NULL, NULL);


		//Writing back the data
		for(i = 0; i < filtersettings[3]; i++)
		{
			writing_back(filemode, filename, filename_e,filename_t, energy_time,i);
		}

		if(debugmode != 0)
		{
			printf("The Positions are:\n");
			for(i=0; i < counter; i++)
			{
				printf("%d\t", positions[i]);
				printf("note that this postion is the middle of the rect \n");
			}
		}

		if(filtersettings[3] > counter)
		{
			///		safetyfactor = safetyfactor + 5*(filtersettings[3] - counter);
			if(safetyfactor <= 0)
			{
				safetyfactor = 10;
			}

			notexpect = filtersettings[3] - expected;
			if(safemode != 0 && notexpect >= notexpect_max)
			{
				printf("The Filter found to many peaks it. It expected %d. It found %d times more than expected.\n", expected, notexpect);
				printf("Safemode is on. Exit program \n");
				OCL_CHECK_ERROR(clReleaseMemObject(input));
				OCL_CHECK_ERROR(clReleaseMemObject(output));
				OCL_CHECK_ERROR(clReleaseMemObject(settings));
				OCL_CHECK_ERROR(clReleaseKernel(filter1));
				OCL_CHECK_ERROR(clReleaseProgram(program));

				ocl_free(ocl);

				free(input_vector);
				free(energy_time);
				free(positions);
				free(filtersettings);

			}
			else
			{
				printf("The Filter found to many peaks it. It expected %d. It found %d times more than expected \n", expected, notexpect);
			}
		}






		

                file_4 = fopen(readbacktime, "a+");
                if(file_4 == NULL)
                {
                        printf("could not open file \n");
                        exit(1);
                }


                file_5 = fopen(writetime, "a+");
                if(file_5 == NULL)
                {
                        printf("could not open file \n");
                        exit(1);
                }



                clGetEventProfilingInfo(readback, CL_PROFILING_COMMAND_START,sizeof(read_start), &read_start, NULL);
                clGetEventProfilingInfo(readback, CL_PROFILING_COMMAND_END, sizeof(read_end), &read_end, NULL);

                clGetEventProfilingInfo(write, CL_PROFILING_COMMAND_START,sizeof(write_start), &write_start,NULL);
                clGetEventProfilingInfo(write, CL_PROFILING_COMMAND_END, sizeof(write_end), &write_end, NULL);



                read_time = read_end - read_start;
                fprintf(file_4, "%0.3f\t", (read_time/1000000.0));


                write_time = write_end - write_start;
                fprintf(file_5, "%0.3f\t", (read_time/1000000.0));





















		file = fopen(timefile, "a+");
		if(file == NULL)
		{
			printf("could not open file \n");
			exit(1);
		}

		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(queued), &queued, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT,sizeof(submit), &submit, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,sizeof(start), &start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,sizeof(end), &end, NULL);

		total_time = end - start;
		printf("total time = %f\n",total_time);
		printf("Queued %llu\t Submit: %llu\t Start: %llu\t End: %llu \t \n",queued,submit,start,end);



		printf("Runtime %0.3f\n",(total_time/1000000.0));
		fprintf(file, "%0.3f\t", (total_time/1000000.0));



		OCL_CHECK_ERROR(clReleaseMemObject(input));
		OCL_CHECK_ERROR(clReleaseMemObject(output));
		OCL_CHECK_ERROR(clReleaseMemObject(settings));
		OCL_CHECK_ERROR(clReleaseKernel(filter1));
		OCL_CHECK_ERROR(clReleaseProgram(program));

		ocl_free(ocl);
		free(input_vector);
		free(energy_time);
		free(positions);
		free(filtersettings);
		printf("no segfault \n this was %d\n",run);
		fclose(file);
		fclose(file_4);
		fclose(file_5);

	}



}











