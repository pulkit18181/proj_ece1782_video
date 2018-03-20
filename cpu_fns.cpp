// yuv format (YMN -> UMN, VMN)xF
// y   format (YMN)xF


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <stdint.h>
#include <pthread.h>

#define MYTYPE uint8_t
#define F 300
#define C 3
// #define enable_prints


#define M 144
#define N 176
#define MN 25344
#define R 16
#define B  8	  
#define MB 18    // (M/B)
#define NB 22    // (N/B)
#define BB 64    // (B*B)
#define lBB 6    // log2(BB)


// #define M 144
// #define N 176
// #define MN 25344
// #define R 8
// #define B 4            // #define B 8       // #define B  16	  
// #define MB 36          // #define MB 18     // #define MB 9    
// #define NB 44          // #define NB 22     // #define NB 11   
// #define BB 16          // #define BB 64     // #define BB 256  
// #define lBB 2          // #define lBB 6     // #define lBB 8   

// #define M 288
// #define N 352
// #define R  16
// #define MN 101376  
// #define B  16      // #define B  32  
// #define MB 18      // #define MB 9   
// #define NB 22      // #define NB 11  
// #define BB 256     // #define BB 1024
// #define lBB 8      // #define lBB 10 


#define MBbT ((MB/2)*2 == MB ? MB/2 : MB/2+1) // optimized for 8 threads
#define NBbT ((NB/4)*4 == NB ? NB/4 : NB/4+1) // optimized for 8 threads

#define getP(X) ((X)*B)
#define ptr(f,r,c) ((f)*MN +getP(r)*N +getP(c))
#define ptrp(f,r,c) ((f)*MN +(r)*N +(c))
#define getB(X) (floor(X/B))
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

void disp_Y(MYTYPE Y[M][N][F]){
    int disp_factor = 1;
    printf("\n");
    printf("\n");
    for (int ll=0; ll<F; ll++){
        printf("\n");
        for (int ii=0; ii<M/disp_factor; ii++){
            printf("\n");
            for (int jj=0; jj<N/disp_factor; jj++){
                printf("%3u ",Y[ii][jj][ll]);
            }
        }
    }
}

void disp_YUV(MYTYPE Y[M][N][C][F], int comp){
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

void get_Y(FILE *fid, MYTYPE *Yonly, int comps){
    MYTYPE temp;
    for (int ll=0; ll<F; ll++) {
        for (int kk=0; kk<comps; kk++){
            for (int ii=0; ii<M; ii++){
                for (int jj=0; jj<N; jj++){
                    int idx = ll*MN +ii*N +jj;
                    if (kk==0){
                        fread (&Yonly[idx],sizeof(char),1,fid);
                    }
                    else if ((ii%2==0 && jj%2==0) || kk==0){
                        fread (&temp,sizeof(char),1,fid);
                    }
                }
            }
        }
    }
}

void write_Y(FILE *fid, MYTYPE *Yonly){
    MYTYPE temp;
    for (int ll=0; ll<F; ll++){
        for (int ii=0; ii<M; ii++){
            for (int jj=0; jj<N; jj++){
                int idx = ll*MN +ii*N +jj;
                fwrite(&Yonly[idx],sizeof(char),1,fid);
            }
        }
    }
}

void write_diff_Y(FILE *fid, MYTYPE *Y1, MYTYPE *Y2, int less){
    MYTYPE temp;
    for (int ll=0; ll<F-less; ll++){
        for (int ii=0; ii<M; ii++){
            for (int jj=0; jj<N; jj++){
                int idx = ll*MN +ii*N +jj;
                int val = Y1[idx]-Y2[idx];
                fwrite(&val,sizeof(char),1,fid);
            }
        }
    }
}


int get_sad(MYTYPE *Y1, int br1,int bc1,int bf1, MYTYPE *Y2, int br2,int bc2,int bf2, bool print=0){
    register int val = 0;
#define size2 B;   // fix me
    // register int idx1 = ptr(bf1,br1,bc1);
    // register int idx2 = ptr(bf2,br2,bc2);
    // for (register int ii=0; ii<B; ii++){
    //     for (register int jj=0; jj<B; jj++){
    //         register MYTYPE idx1f = Y1[idx1 + ii*N + jj];
    //         register MYTYPE idx2f = Y2[idx2 + ii*size2 + jj];
    // 	    val += myabs(idx1f - idx2f);
    //     }
    // }
    
    register MYTYPE idx1 = ptr(bf1,br1,bc1);
    register MYTYPE idx2 = ptr(bf2,br2,bc2);
    for (register int ii=0; ii<B; ii++){
        for (register int jj=0; jj<B; jj++){
	    val += myabs(Y1[idx1++] - Y1[idx2++]);
        }
	idx1 += (N-B);
    }
    return val;
    // return val>>(lBB);
}

int get_sad2(MYTYPE *Y1, int br1,int bc1,int bf1, MYTYPE *Y2, int pr2,int pc2,int pf2, bool print=0){
    register int val = 0;
    register int idx1 = ptr (bf1,br1,bc1);
    register int idx2 = ptrp(pf2,pr2,pc2);
    // printf("%d %d %d-> %d----- %d,%d,%d-> %d\n",bf1,br1,bc1, idx1,pf2,pr2,pc2, idx2);
    for (register int ii=0; ii<B; ii++){
      for (register int jj=0; jj<B; jj++){
	int temp = (Y1[idx1] - Y1[idx2]);
	val = val +  myabs(temp);
	#ifdef enable_prints
	// printf("%d   -- %d %d----- %d,%d\n",val,temp, myabs(temp), idx1, idx2);
        #endif
	idx1++; idx2++;
      }
      idx1 += (N-B);
      idx2 += (N-B);
    }
    return val;
    // return val>>(lBB);
}


void process_iblock(MYTYPE *Yonly, MYTYPE *reconstructed, MYTYPE *predicted_frame, MYTYPE *motion_vector, int br,int bc,int bf){
    MYTYPE      local_block      [B][B];
    register int local_sad;
    MYTYPE      lowest_sad_block [B][B];
    register int lowest_mode;
    register int lowest_sad;

    register int curr = ptr(bf,br  ,bc  );
    register int left = ptr(bf,br  ,bc-1);
    register int top  = ptr(bf,br-1,bc  );

    lowest_sad = 147483647;
    lowest_mode = 0;



    // ------------ mode 0 (horizontal on the left) ------------
    if (bc==0) {
      for (int jj=0; jj<B; jj++){
        for (int ii=0; ii<B; ii++){
          local_block[ii][jj] = 127;
        }
      }
    }
    else {
      for (int ii=0; ii<B; ii++){
        register MYTYPE val = reconstructed[curr + ii*N - 1];
        for (int jj=0; jj<B; jj++){
          local_block[ii][jj] = val;
        }
      }
    }
    local_sad = get_sad(Yonly,br,bc,bf,(MYTYPE*)local_block,0,0,0);
#ifdef enable_prints
    printf("mode=%u, sad=%d, loc_sad=%d\n",0, lowest_sad, local_sad);
#endif
    if (local_sad < lowest_sad){
        lowest_mode = 0;
        lowest_sad = local_sad;
        memcpy(&lowest_sad_block, &local_block, sizeof(lowest_sad_block));
    }


    // ------------ mode 1 (vertical or on the top) ------------
    if (br==0) {
      for (int jj=0; jj<B; jj++){
        for (int ii=0; ii<B; ii++){
          local_block[ii][jj] = 127;
        }
      }
    }
    else {
      for (int jj=0; jj<B; jj++){
        register MYTYPE val = reconstructed[curr - N + jj];
        for (int ii=0; ii<B; ii++){
          local_block[ii][jj] = val;
        }
      }
    }
    local_sad = get_sad(Yonly,br,bc,bf,(MYTYPE*)local_block,0,0,0);
#ifdef enable_prints
    printf("mode=%u, sad=%d, loc_sad=%d\n",1, lowest_sad, local_sad);
#endif
    if (local_sad < lowest_sad){
        lowest_mode = 1;
        lowest_sad = local_sad;
        memcpy(&lowest_sad_block, &local_block, sizeof(lowest_sad_block));
    }



    // ------------ returning the predicted frame ------------
    for (int ii=0; ii<B; ii++){
        for (int jj=0; jj<B; jj++){
            register int idx = curr + ii*N +jj;
            predicted_frame[idx] = lowest_sad_block[ii][jj];
        }
    }

}


struct pblock_t{
  MYTYPE *Yonly;
  MYTYPE *reconstructed;
  MYTYPE *predicted;
  MYTYPE *motion_vector;
  int br, bc, bf;
  pblock_t (MYTYPE *_Yonly, MYTYPE *_reconstructed, MYTYPE* _predicted, MYTYPE* _motion_vector, int _ii,int _jj,int _kk){
    Yonly          = _Yonly           ;
    reconstructed  = _reconstructed   ;
    predicted      = _predicted       ;
    motion_vector  = _motion_vector   ;
    br             = _ii              ;
    bc             = _jj              ;
    bf             = _kk              ;
  }
  pblock_t (){}
} ;

#ifdef MT
  void *process_pblock_mt(void *args){
#else
  void process_pblock_mt(void *args){
#endif

  struct pblock_t *targs = (pblock_t*) args;
  MYTYPE *Yonly           = targs -> Yonly           ;
  MYTYPE *reconstructed   = targs -> reconstructed   ;
  MYTYPE *predicted_frame = targs -> predicted       ;
  MYTYPE *motion_vector   = targs -> motion_vector   ;
  register int br         = (unsigned int)targs -> br              ;
  register int bc         = (unsigned int)targs -> bc              ;
  register int bf         = (unsigned int)targs -> bf              ;
  register int pr         = (unsigned int)getP(br);
  register int pc         = (unsigned int)getP(bc);
  register int curr       = (unsigned int)(bf*MN +(pr)*N +(pc));

  // ------------ first frame ------------
  if (bf==0) {
#ifdef enable_prints
    // printf("processing first frame");
#endif
    for (int ii=0; ii<B; ii++){
      for (int jj=0; jj<B; jj++){
	predicted_frame[curr++] = 127;
      }
      curr += (N - B);
    }
  }

  // ------------ non-first frame ------------
  else {
    register unsigned int lowest_sad = 147483647;
    register int lowest_r;	
    register int lowest_c;	
    register int startr, endr;
    register int startc, endc;
    startr = mymax(0,(pr)-R);
    startc = mymax(0,(pc)-R);
    endr = mymin(M,(pr)+R);
    endc = mymin(N,(pc)+R);
#ifdef enable_prints
    printf("processing window_block=%u,%u, %u,%u\n",startr, endr,startc,endc);
#endif

    for (; startr<endr; startr++){
      for (register int cc=startc; cc<endc; cc++){
	register int local_sad = get_sad2(Yonly,br,bc,bf,reconstructed,startr,cc,bf-1);
#ifdef enable_prints
	printf("processing window_block=%u,%u, sad=%d, loc_sad=%d\n",startr,cc, lowest_sad, local_sad);
#endif
	if (local_sad < lowest_sad){
	  lowest_r = startr;
	  lowest_c = cc;
	  lowest_sad = local_sad;
	  if (lowest_sad == 0) goto LABEL; 
	}
      }
    }

    LABEL:
#ifdef enable_prints
    printf("selected %d,%d,%d for %d,%d,%d\n", lowest_r,lowest_c,bf-1, pr,pc,bf);
#endif
    // ------------ returning the predicted frame ------------
    {
      register int idx2 = ptrp(bf-1,lowest_r,lowest_c);
      for (register int ii=0; ii<B; ii++){
	for (register int jj=0; jj<B; jj++){
#ifdef enable_prints
	  // printf("preping %d,%d\n", curr,idx2);
#endif
	  predicted_frame[curr++] = reconstructed[idx2++];
	}
	curr += (N-B);
	idx2 += (N-B);
      }
    }
  }
#ifdef MT
  pthread_exit(NULL);
#endif
}
