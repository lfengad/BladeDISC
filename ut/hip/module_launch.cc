#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include <stdio.h>
#include <fstream>
#include <iostream>

#define ROCM_DRIVER_CALL(x)                                                                    \
  {                                                                                            \
    hipError_t result = x;                                                                     \
    if (result != hipSuccess && result != hipErrorDeinitialized) {                             \
      std::cout << "ROCM HIP Error: " #x " failed with error: " << hipGetErrorString(result) << std::endl; \
    }                                                                                          \
  }

#define ROCM_CALL(func)                                              \
  {                                                                  \
    hipError_t e = (func);                                           \
    CHECK(e == hipSuccess) << "ROCM HIP: " << hipGetErrorString(e); \
  }
#define checkCudaErrors(val) \
	if (val != hipSuccess) exit(EXIT_FAILURE)


void LoadBinaryFromFile(const std::string& file_name, std::string* data) {
  std::ifstream fs(file_name, std::ios::in | std::ios::binary);
  // get its size:
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data->resize(size);
  fs.read(&(*data)[0], size);
}

hipFunction_t GetFunc(const std::string& file_name)  {
    std::string data;
    hipModule_t module;
    LoadBinaryFromFile(file_name, &data);
    ROCM_DRIVER_CALL(hipModuleLoadData(&module, data.c_str()));
    hipFunction_t func;
    std::string name = 
	     "main_kRowReduction_reduce__2_1_0___1b1rX_vectile2X_no_vectile";
    hipError_t result = hipModuleGetFunction(&func, module, name.c_str());
    if (result != hipSuccess) {
      std::cout << "ROCMError: hipModuleGetFunction " << name
                 << " failed with error: " << hipGetErrorString(result) << std::endl;
    }
    return func;
}

void LaunchKernel(hipFunction_t func, int d0, int d1, float* input, float* output) {
    int arg0 = 256;
    int arg1 = d0 * 256;
    uint8_t args[32];
    size_t size = 32;
  //  void *args[] = {&input, &arg0, &arg1, &output};
    
    *((float**)(&args[0])) = input;
    *((int**)(&args[8])) = &arg0;
    *((int**)(&args[16])) = &arg1;
    *((float**)(&args[24])) = output;
    std::cout <<  
	   (*(float**)(&args[0])) <<  " " <<
(*(int*)(&args[8])) << " " <<
(*(int*)(&args[12])) << " " <<
	   (*(float**)(&args[16])) <<  " " << std::endl;
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, (void*)args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &size, HIP_LAUNCH_PARAM_END};

    ROCM_DRIVER_CALL(hipModuleLaunchKernel(func, d0, 1, 1, 
                256, 1, 1, 
                0,
                0, 
		nullptr,
		reinterpret_cast<void**>(&config)
		));
		//nullptr));

   // ROCM_DRIVER_CALL(hipLaunchKernel(func, dim3(d0), dim3(256), args, 0, NULL));
}




int main(int argc, char* argv[]) {
  const int d0 = 1;
  const int d1 = 1024;

  float *input, *output;
  uint64_t mem_size = d1 * sizeof(float);
  uint64_t mem_size0 = d0 * sizeof(float);
  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&input), mem_size));
  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&output), mem_size0));
  float* h_init = reinterpret_cast<float*>(malloc(mem_size));
  for (int64_t i = 0; i < 1024; i++) {
    h_init[i] = 0.25;
  }
  checkCudaErrors(hipMemcpy(input, h_init, mem_size, hipMemcpyDefault));
  checkCudaErrors(hipMemset(output, 0, mem_size0));

  free(h_init);
//   float yita = 0.0001;
//   float gamma = 1.3;
//   float beta = 1.3;

  hipEvent_t start, stop;
  checkCudaErrors(hipEventCreate(&start));
  checkCudaErrors(hipEventCreate(&stop));
  hipFunction_t func = GetFunc(argv[1]);

  // warmup
  checkCudaErrors(hipEventRecord(start));
  int times = 20;
  for (int i = 0; i < times; i++)
    // softmax1st<<<1, 256>>>(input, output);
    LaunchKernel(func, d0, d1, input, output);
  checkCudaErrors(hipEventRecord(stop));
  checkCudaErrors(hipEventSynchronize(stop));
  float msec = 0.0f;
  checkCudaErrors(hipEventElapsedTime(&msec, start, stop));
  printf("execution time = %f\n", msec);

  float* o_init = reinterpret_cast<float*>(malloc(mem_size0));
  memset(o_init, 0, mem_size0);
  checkCudaErrors(hipMemcpy(o_init, output, mem_size0, hipMemcpyDefault));
  for (int64_t i = 0; i < d0; i++) {
      printf(" %f ", o_init[i]);
  }


  return 0;
}
