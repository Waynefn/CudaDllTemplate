// dllCuda.h

#define CUDADLLEXMPL_EXPORTS

#ifdef CUDADLLEXMPL_EXPORTS
#define DLLCUDA_API EXTERN_C __declspec(dllexport)
#else
#define DLLCUDA_API EXTERN_C __declspec(dllimport)
#endif

/*
calculate vector c=a+b, whose length are all size.
return  0: Success.
return >0: Cuda error code.
return -1: Do not call deviceReset.
*/
DLLCUDA_API int addWithCuda(int *c, const int *a, const int *b, unsigned int size);
/*
find and reset device.
return  0: Success.
return >0: Cuda error code.
return -1: No cuda device.
*/
DLLCUDA_API int deviceReset(void);
/*
test function.
return a+b.
*/
DLLCUDA_API int add(int a,int b);

/*
resl: float, length n_a
return:
return  0: Success.
return >0: Cuda error code.
return -1: Do not call deviceReset.
*/
DLLCUDA_API  int mapWithCuda(int interm_j, int interp_num, int n_a, float*resl);
