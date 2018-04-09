// yuv format (YMN -> UMN, VMN)xF
// y   format (YMN)xF

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <stdint.h>

#define M 160
#define N 192
#define MN 30720
#define Br 5
#define Bc 6

#define C 3
#define F 300
#define R 16
#define MB 32
#define NB 32
#define INTMAX 147483647 

#define MYTYPE uint8_t

#define getPr(K) ((K)*Br)
#define getPc(K) ((K)*Bc)

#define ptr(f,r,c) ((f)*MN +getPr(r)*N +getPc(c))
#define ptrp(f,r,c) ((f)*MN +(r)*N +(c))
#define getBr(X) (floor(X/Br))
#define getBc(X) (floor(X/Bc))
#define mymax(X,Y) ((X)>(Y)?(X):(Y))
#define mymin(X,Y) ((X)<(Y)?(X):(Y))
#define myabs(X)   ((X)>0?(X):-1*(X))

// ###########################################################################
//                     TIMER RELATED
// ###########################################################################
struct timeval tv;
long int start, end, elapsed;
void start_timer(){
  gettimeofday(&tv,NULL);
  start = tv.tv_sec*(long int)1000000+tv.tv_usec;
}
long int end_timer(const char msg[]){
  gettimeofday(&tv,NULL);
  end = tv.tv_sec*(long int)1000000+tv.tv_usec;
  elapsed = end - start;
  printf("%s - time taken = %ld us\n",msg,elapsed);
  return elapsed;
}

//Need to change all references to files and memory

__global__ void disp_YUV(uint8_t Y[M][N][C][F], int comp){
  int disp_factor = 1;
  printf("\n");
  printf("\n");
  for (int ll=0; ll<F; ll++){
    printf("\n");
    for (int ii=0; ii<M/disp_factor; ii++){
      printf("\n");
      for (int jj=0; jj<N/disp_factor; jj++){
	printf("%3u ",Y[ii][jj][comp][ll]);
      }
    }
  }
}

void get_Y(FILE *fid, uint8_t *Yonly, int comps){
  uint8_t temp;
  for (int ll=0; ll<F; ll++) {
    for (int kk=0; kk<comps; kk++){
      for (int ii=0; ii<M; ii++){
	for (int jj=0; jj<N; jj++){
	  int idx = ll*MN +ii*N +jj;
	  if (kk==0){
	    fread (&Yonly[idx],sizeof(char),1,fid); //CHECK
	  }
	  else if ((ii%2==0 && jj%2==0) || kk==0){
	    fread (&temp,sizeof(char),1,fid);	//CHECK
	  }
	}
      }
    }
  }
}

void write_Y(FILE *fid, uint8_t *Yonly){
  for (int ll=0; ll<F; ll++){
    for (int ii=0; ii<M; ii++){
      for (int jj=0; jj<N; jj++){
	int idx = ll*MN +ii*N +jj;
	fwrite(&Yonly[idx],sizeof(char),1,fid);	//CHECK
      }
    }
  }
}

void write_diff_Y(FILE *fid, uint8_t *Y1, uint8_t *Y2){
  for (int ll=0; ll<F; ll++){
    for (int ii=0; ii<M; ii++){
      for (int jj=0; jj<N; jj++){
	int idx = ll*MN +ii*N +jj;
	int val = Y1[idx]-Y2[idx];
	fwrite(&val,sizeof(char),1,fid);	//CHECK
      }
    }
  }
}


__global__ void deviceInit(uint8_t *_A, uint8_t *_B, uint8_t *_C, uint8_t *_D, uint8_t *_E){
  for (register int kk=0; kk<M*N*F; kk++) {
    _A[kk] = 0;
    _B[kk] = 0;
    _C[kk] = 0;
    _D[kk] = 0;
    _E[kk] = 0;
  }
  return;
}


__device__ int get_sad(uint8_t *Y1, int br1,int bc1,int bf1, uint8_t *Y2, int br2,int bc2,int bf2, bool print=0){
  int val = 0;
  register int idx1 = bf1*MN +getPr(br1)*N +getPc(bc1);
  register int idx2 = bf2*MN +getPr(br2)*N +getPc(bc2);
  for (register int ii=0; ii<Br; ii++){
    for (register int jj=0; jj<Bc; jj++){
      register int idx1f = idx1 + ii*N + jj;
      register int idx2f = idx2 + ii*Bc + jj;
      if (print) printf("(%d,%d) = %3d %3d \n",idx1f,idx2f,Y1[idx1f], Y2[idx2f] );
      val += abs(Y1[idx1f] - Y2[idx2f]);
    }
  }
  return val;
  // return val>>(lBB);
}

__device__ int get_sad2(MYTYPE *Y1, int br1,int bc1,int bf1, MYTYPE *Y2, int pr2,int pc2,int pf2, bool print=0){
  register int val = 0;
  register int idx1 = ptr (bf1,br1,bc1);
  register int idx2 = ptrp(pf2,pr2,pc2);
#ifdef UNROLL
#pragma unroll
#endif
  for (register int ii=0; ii<Br; ii++, idx1+=(N-Bc),idx2+=(N-Bc)){
#ifdef UNROLL
#pragma unroll
#endif
    for (register int jj=0; jj<Bc; jj++,idx1++,idx2++){
      val = val +  myabs((Y1[idx1] - Y1[idx2]));
    }
  }
  if (pr2<M && pr2>=0 && pc2<N && pc2>=0)  return val;
  else return INTMAX;
}

__global__ void process_pblock(uint8_t *Yonly, uint8_t *reconstructed, uint8_t *predicted_frame, uint8_t *motion_vector, int bf){

  int br = blockIdx.y * blockDim.y + threadIdx.y;
  int bc = blockIdx.x * blockDim.x + threadIdx.x;
	
  register int pr         = (unsigned int)getPr(br);
  register int pc         = (unsigned int)getPc(bc);
  register int curr       = (unsigned int)(bf*MN +(pr)*N +(pc));

  // printf("block -> br,bc = %u,%u \n",br, bc);
  if (br > MB || bc > NB) {
    // printf("no such block -> br,bc = %u,%u \n",br, bc);
    return;
  }
  // ------------ first frame ------------
  if (bf==0) { 

    // printf("processing first frame");

#ifdef UNROLL
#pragma unroll
#endif
    for (int ii=0; ii<Br; ii++, curr+=(N-Bc)){
#ifdef UNROLL
#pragma unroll
#endif
      for (int jj=0; jj<Bc; jj++, curr++){
	predicted_frame[curr] = 127;
      }
    }
  }

  // ------------ non-first frame ------------
  else {
    register unsigned int lowest_sad = INTMAX;
    register int lowest_r;	
    register int lowest_c;	
    unsigned int local_sad [2*R+1][2*R+1];

#define act_rr (pr+rr-R)
#define act_cc (pc+cc-R)
// #ifdef UNROLL
// #pragma unroll
// #endif
    for (register int rr=0; rr<=2*R; rr+=5){
// #ifdef UNROLL
// #pragma unroll
// #endif
      for (register int cc=0; cc<=2*R; cc++){
				local_sad[rr][cc] = get_sad2(Yonly,br,bc,bf,reconstructed,act_rr,act_cc,bf-1);
				local_sad[rr+1][cc] = get_sad2(Yonly,br,bc,bf,reconstructed,act_rr+1,act_cc,bf-1);
				local_sad[rr+2][cc] = get_sad2(Yonly,br,bc,bf,reconstructed,act_rr+2,act_cc,bf-1);
				local_sad[rr+3][cc] = get_sad2(Yonly,br,bc,bf,reconstructed,act_rr+3,act_cc,bf-1);
				local_sad[rr+4][cc] = get_sad2(Yonly,br,bc,bf,reconstructed,act_rr+4,act_cc,bf-1);
			
      }
    }
#ifdef UNROLL
#pragma unroll
#endif
    for (register int rr=0; rr<=2*R; rr++){
#ifdef UNROLL
#pragma unroll
#endif
      for (register int cc=0; cc<=2*R; cc++){
	if (local_sad[rr][cc] < lowest_sad){
	  lowest_r = act_rr;
	  lowest_c = act_cc;
	  lowest_sad = local_sad[rr][cc];
	}
      }
    }

    // printf("proc (%u,%u)%d ..... selected %d,%d,%d with best=%d\n", pr, pc, bf, lowest_r,lowest_c,bf-1, lowest_sad);

    // ------------ returning the predicted frame ------------
    {
      register int idx2 = ptrp(bf-1,lowest_r,lowest_c);
      // printf("selected %d,%d,%d for (%d,%d),%d with best=%d, curr=%d, idx2=%d ..... start end=%u,%u, %u,%u\n", lowest_r,lowest_c,bf-1, pr,pc,bf,lowest_sad, curr, idx2, startr, endr,startc,endc);
#ifdef UNROLL
#pragma unroll
#endif
      for (register int ii=0; ii<Br; ii++, curr+=(N-Bc), idx2+=(N-Bc)){
#ifdef UNROLL
#pragma unroll
#endif
	for (register int jj=0; jj<Bc; jj++, curr++, idx2++){
	  predicted_frame[curr] = reconstructed[idx2];
	}
      }
    }
  }

}
