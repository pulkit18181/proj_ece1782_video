#include "gpu_fns.cu"

#define BLOCK_SIZEY MB
#define BLOCK_SIZEX NB
#define GRID_SIZE 1 

int main(){

    FILE    *fid           = fopen("akiyo_qcif.y","rb");
    FILE    *fid2          = fopen("akiyo_qcif_ppred.y","wb");
    FILE    *fid3          = fopen("akiyo_qcif_res.y","wb");
	
	uint8_t *h_Yonly         = (uint8_t *) malloc(M*N*F*sizeof(uint8_t)); if (h_Yonly         == NULL) fprintf(stderr, "Bad malloc on Yonly          \n");
    uint8_t *h_predicted     = (uint8_t *) malloc(M*N*F*sizeof(uint8_t)); if (h_predicted     == NULL) fprintf(stderr, "Bad malloc on predicted      \n");
    uint8_t *h_reconstructed = (uint8_t *) malloc(M*N*F*sizeof(uint8_t)); if (h_reconstructed == NULL) fprintf(stderr, "Bad malloc on reconstructed  \n");
    uint8_t *h_motion_vector = (uint8_t *) malloc(M*N*F*sizeof(uint8_t)); if (h_motion_vector == NULL) fprintf(stderr, "Bad malloc on motion_vector  \n");
    uint8_t *h_Res_orig      = (uint8_t *) malloc(M*N*F*sizeof(uint8_t)); if (h_Res_orig      == NULL) fprintf(stderr, "Bad malloc on Res_orig       \n");    
	
	uint8_t *d_Yonly;
	uint8_t *d_predicted;
	uint8_t *d_reconstructed;
	uint8_t *d_motion_vector;
	uint8_t *d_Res_orig;
	
    printf("single thread with config -> M=%d, N=%d, B=%d R=%d\n", M, N, B, R);

	//Allocating memory on GPU
	cudaMalloc((void **)&d_Yonly, M*N*F*sizeof(uint8_t));
	cudaMalloc((void **)&d_predicted, M*N*F*sizeof(uint8_t));
	cudaMalloc((void **)&d_reconstructed, M*N*F*sizeof(uint8_t));
	cudaMalloc((void **)&d_motion_vector, M*N*F*sizeof(uint8_t));
	cudaMalloc((void **)&d_Res_orig, M*N*F*sizeof(uint8_t));
	
	//Reading from files
	get_Y(fid,h_Yonly,1);
    fclose(fid);
	
 	start_timer();
	
	//Transferring data from the CPU to the GPU
	cudaMemcpy(d_Yonly, h_Yonly, (M*N*F*sizeof(uint8_t)), cudaMemcpyHostToDevice);
	
	end_timer("Mem Copy: CPU->GPU");
	
	start_timer();
	
	dim3 blockDim(BLOCK_SIZEX, BLOCK_SIZEY);
	dim3 gridDim(GRID_SIZE, GRID_SIZE);
	//dim3 gridDim((N + blockDim.x - 1)/ blockDim.x, (M + blockDim.y - 1) / blockDim.y, 1);
	
	//Logic happens here
    for (int kk=0; kk<F; kk++){      
	  // if (ii==1) goto XX;
	  // printf("%u, %u, %u \n",ii,jj, kk);
	  // process_pblock(Yonly, reconstructed, predicted, motion_vector, ii,jj,kk);
	  process_pblock<<<gridDim,blockDim>>>(d_Yonly, d_Yonly, d_predicted, d_motion_vector,kk);	  
	  cudaDeviceSynchronize();
	  // end_timer("Thread Logic");
	}
	
	end_timer("Entire Logic");
	
	start_timer();
	
	//Transferring data from the GPU to the CPU
	cudaMemcpy(h_predicted, d_predicted, (M*N*F*sizeof(uint8_t)), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	end_timer("Mem Copy: GPU->CPU");

	//Writing files
	write_Y(fid2,h_predicted);
    write_diff_Y(fid3,h_Yonly,h_predicted);
	
	//Free memory on GPU
	cudaFree(d_Yonly);
	cudaFree(d_predicted);
	cudaFree(d_reconstructed);
	cudaFree(d_motion_vector);
	cudaFree(d_Res_orig);
	
	//Free memory on CPU
    free(h_Yonly);
    free(h_predicted);
    free(h_reconstructed);
    free(h_motion_vector);
    free(h_Res_orig);
}	