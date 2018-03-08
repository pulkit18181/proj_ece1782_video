// yuv format (YMN -> UMN, VMN)xF
// y   format (YMN)xF



#define M 144
#define N 176
#define C 3
#define F 25
#define R 8
#define B 16 
#define MN 25344 // M*N
#define MB 9     // M/B
#define NB 11    // N/B
#define BB 256   // B*B
#define lBB 8    // log2(BB)

#define getP(X) (X*B)
#define getB(X) (floor(X/B))
#define max(X,Y) (X>Y?X:Y)
#define min(X,Y) (X<Y?X:Y)

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

void disp_Y(uint8_t Y[M][N][F]){
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

void disp_YUV(uint8_t Y[M][N][C][F], int comp){
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

void write_Y(FILE *fid, uint8_t *Yonly){
    uint8_t temp;
    for (int ll=0; ll<F; ll++){
        for (int ii=0; ii<M; ii++){
            for (int jj=0; jj<N; jj++){
                int idx = ll*MN +ii*N +jj;
                fwrite(&Yonly[idx],sizeof(char),1,fid);
            }
        }
    }
}

void write_diff_Y(FILE *fid, uint8_t *Y1, uint8_t *Y2){
    uint8_t temp;
    for (int ll=0; ll<F; ll++){
        for (int ii=0; ii<M; ii++){
            for (int jj=0; jj<N; jj++){
                int idx = ll*MN +ii*N +jj;
		int val = Y1[idx]-Y2[idx];
                fwrite(&val,sizeof(char),1,fid);
            }
        }
    }
}


int get_sad(uint8_t *Y1, int br1,int bc1,int bf1, uint8_t *Y2, int br2,int bc2,int bf2, bool print=0){
    int val = 0;
    int size2 = B;
    register int idx1 = bf1*MN +getP(br1)*N +getP(bc1);
    register int idx2 = bf2*MN +getP(br2)*N +getP(bc2);
    for (register int ii=0; ii<B; ii++){
        for (register int jj=0; jj<B; jj++){
            register int idx1f = idx1 + ii*N + jj;
            register int idx2f = idx2 + ii*size2 + jj;
	    if (print) printf("(%d,%d) = %3d %3d \n",idx1f,idx2f,Y1[idx1f], Y2[idx2f] );
            val += abs(Y1[idx1f] - Y2[idx2f]);
        }
    }
    return val;
    // return val>>(lBB);
}


void process_iblock(uint8_t *Yonly, uint8_t *reconstructed, uint8_t *predicted_frame, uint8_t *motion_vector, int br,int bc,int bf){
    uint8_t      local_block      [B][B];
    register int local_sad;
    uint8_t      lowest_sad_block [B][B];
    register int lowest_mode;
    register int lowest_sad; 

    register int curr = bf*MN +getP(br)*N +getP(bc);
    register int left = bf*MN +getP(br)*N +getP(bc-1);
    register int top  = bf*MN +getP(br-1)*N +getP(bc);

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
        uint8_t val = reconstructed[left+ ii*N - 1];
        for (int jj=0; jj<B; jj++){
	  local_block[ii][jj] = val;
        }
      }
    }
    local_sad = get_sad(Yonly,br,bc,bf,(unsigned char*)local_block,0,0,0);
    // printf("mode=%u, sad=%d, loc_sad=%d\n",0, lowest_sad, local_sad);
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
        uint8_t val = reconstructed[left - N + jj];
        for (int ii=0; ii<B; ii++){
	  local_block[ii][jj] = val;
        }
      }
    }
    local_sad = get_sad(Yonly,br,bc,bf,(unsigned char*)local_block,0,0,0);
    // printf("mode=%u, sad=%d, loc_sad=%d\n",1, lowest_sad, local_sad);
    if (local_sad < lowest_sad){
        lowest_mode = 1;
        lowest_sad = local_sad;
        memcpy(&lowest_sad_block, &local_block, sizeof(lowest_sad_block));
    }


    
    // ------------ returning the predicted frame ------------ 
    for (int ii=0; ii<B; ii++){
        for (int jj=0; jj<B; jj++){
            int idx = curr + ii*N +jj;
            predicted_frame[idx] = lowest_sad_block[ii][jj];
        }
    }

}


void process_pblock(uint8_t *Yonly, uint8_t *reconstructed, uint8_t *predicted_frame, uint8_t *motion_vector, int br,int bc,int bf){
    uint8_t      local_block      [B][B];
    register int local_sad;
    uint8_t      lowest_sad_block [B][B];
    register int lowest_sad; 
    register int lowest_x;
    register int lowest_y;

    register int curr = bf*MN +getP(br)*N +getP(bc);

    lowest_sad = 147483647;
    lowest_x   = 0;
    lowest_y   = 0;
    
    // ------------ first frame ------------ 
    if (bf==0) {
      // printf("processing first frame");
      for (int ii=0; ii<B; ii++){
        for (int jj=0; jj<B; jj++){
	  int idx = curr + ii*N +jj;
	  predicted_frame[idx] = 127;
        }
      }
    }


    // ------------ non-first frame ------------ 
    else {
      int startx, endx;
      int starty, endy;
      
      startx = max(0,getP(br)-R);
      starty = max(0,getP(bc)-R);
      endx = min(M,getP(br)+R);
      endy = min(N,getP(bc)+R);
      for (int xx=startx; xx<endx; xx++){
	for (int yy=starty; yy<endy; yy++){
	  register int ref = (bf-1)*MN +xx*N +yy;

	  // -----for each of the possible blocks, compute the block, sad and compare
	  for (int ii=0; ii<B; ii++){
	    for (int jj=0; jj<B; jj++){
	      local_block[ii][jj] = reconstructed[ref + ii*N + jj];
	    }
	  }


	  local_sad = get_sad(Yonly,br,bc,bf,(unsigned char*)local_block,0,0,0);
	  // printf("processing window_block=%u,%u, sad=%d, loc_sad=%d\n",xx,yy, lowest_sad, local_sad);
	  if (local_sad < lowest_sad){
	    lowest_x = xx;
	    lowest_y = yy;
	    lowest_sad = local_sad;
	    memcpy(&lowest_sad_block, &local_block, sizeof(lowest_sad_block));
	  }
	}
      }
      // ------------ returning the predicted frame ------------ 
      for (int ii=0; ii<B; ii++){
        for (int jj=0; jj<B; jj++){
	  int idx = curr + ii*N +jj;
	  predicted_frame[idx] = lowest_sad_block[ii][jj];
        }
      }

      
      // printf("choosing window_block=%u,%u when start was - %d,%d\n",lowest_x, lowest_y, getP(br), getP(bc));

    }

}
