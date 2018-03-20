#include "cpu_fns.cpp"
#define NUM_THREADS 8

int main () {

    FILE    *fid           = fopen("backup/news_qcif.y","rb");
    FILE    *fid2          = fopen("news_qcif_cpu_ppred_mt.y","wb");
    FILE    *fid3          = fopen("news_qcif_cpu_pres_mt.y","wb");
 // FILE    *fid4          = fopen("news_qcif_cpu_pact_mt.y","wb");
    MYTYPE *Yonly         = (MYTYPE *) malloc(M*N*F*sizeof(MYTYPE))   ; if (Yonly         == NULL) fprintf(stderr, "Bad malloc on Yonly          \n");
    MYTYPE *predicted     = (MYTYPE *) malloc(M*N*F*sizeof(MYTYPE))   ; if (predicted     == NULL) fprintf(stderr, "Bad malloc on predicted      \n");
    MYTYPE *reconstructed = (MYTYPE *) malloc(M*N*F*sizeof(MYTYPE))   ; if (reconstructed == NULL) fprintf(stderr, "Bad malloc on reconstructed  \n");
    MYTYPE *motion_vector = (MYTYPE *) malloc(M*N*F*14*sizeof(MYTYPE)); if (motion_vector == NULL) fprintf(stderr, "Bad malloc on motion_vector  \n");
    MYTYPE *Res_orig      = (MYTYPE *) malloc(M*N*F*sizeof(MYTYPE))   ; if (Res_orig      == NULL) fprintf(stderr, "Bad malloc on Res_orig       \n");

    get_Y(fid,Yonly,1);
    fclose(fid);

    pthread_t       pid [NUM_THREADS] = {0};
    struct pblock_t args[NUM_THREADS];
    register int    err ;

    start_timer();

    // printf("PULKITA sent T%d MB%d,NB%d,MBbT %d,NBbT %d\n",NUM_THREADS,MB,NB,MBbT,NBbT);

    #ifdef OP1
    printf("8 threads with config THREAD SCH1 -> M=%d, N=%d, B=%d R=%d\n", M, N, B, R);
    for (register int kk=0; kk<F; kk++){
      for (register int rr=0; rr<MBbT; rr++){
	for (register int cc=0; cc<NBbT; cc++){
	  for (register int tid=0; tid<NUM_THREADS; tid++){
	    register int r,c;
	    r = (tid/4)*MBbT+rr;
	    c = (tid%4)*NBbT+cc;
	    if ( c >= NB) {continue;}
	    if ( r >= MB) {continue;}
	    pthread_join(pid[tid], NULL);
	    args[tid] = pblock_t(Yonly, Yonly, predicted, motion_vector, r,c,kk);
	    if (pthread_create(&pid[tid], NULL, &process_pblock_mt, &args[tid]))
	      exit(1);
            #ifdef enable_prints
	    printf("PULKITA sent %d,%d,%d tid %d\n",r,c,kk,tid);
            #endif
	  }
	}
      }
    }
    #endif

    #ifdef OP2
    printf("8 threads with config THREAD SCH2 -> M=%d, N=%d, B=%d R=%d\n", M, N, B, R);
    for (int kk=0; kk<F; kk++){
      for (int ii=0; ii<MB; ii++){
        for (int jj=0; jj<NB; jj++){
          register int tid;
          tid = (ii*NB + jj) % NUM_THREADS;
          // tid = (jj) % NUM_THREADS;
          pthread_join(pid[tid], NULL);
          args[tid] = pblock_t(Yonly, Yonly, predicted, motion_vector, ii,jj,kk);
          if (pthread_create(&pid[tid], NULL, &process_pblock_mt, &args[tid]))
            exit(1);
          #ifdef enable_prints
	  printf("PULKITA sent %d,%d,%d tid %d\n",ii,jj,kk,tid);
          #endif
        }
      }
    }
    #endif

    for (int tid = 0; tid<NUM_THREADS; tid++)
      pthread_join(pid[tid], NULL);
    end_timer("static p");

 XX:
    write_Y(fid2,predicted);
    write_diff_Y(fid3,Yonly,predicted,0);
    // write_diff_Y(fid4,Yonly,&Yonly[MN],1);
    free(Yonly         );
    free(predicted     );
    free(reconstructed );
    free(motion_vector );
    free(Res_orig      );

    fclose (fid2);
    fclose (fid3);
    // fclose (fid4);
    pthread_exit(NULL);
    
    
}
