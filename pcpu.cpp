#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <stdint.h>

#include "cpu_fns.cpp"

int main(){

    FILE    *fid           = fopen("backup/foreman_cif.y","rb");
    FILE    *fid2          = fopen("foreman_cif_ppred.y","wb");
    FILE    *fid3          = fopen("foreman_cif_res.y","wb");
    MYTYPE *Yonly         = (MYTYPE *) malloc(M*N*F*sizeof(MYTYPE)); if (Yonly         == NULL) fprintf(stderr, "Bad malloc on Yonly          \n");
    MYTYPE *predicted     = (MYTYPE *) malloc(M*N*F*sizeof(MYTYPE)); if (predicted     == NULL) fprintf(stderr, "Bad malloc on predicted      \n");
    MYTYPE *reconstructed = (MYTYPE *) malloc(M*N*F*sizeof(MYTYPE)); if (reconstructed == NULL) fprintf(stderr, "Bad malloc on reconstructed  \n");
    MYTYPE *motion_vector = (MYTYPE *) malloc(M*N*F*14*sizeof(MYTYPE)); if (motion_vector == NULL) fprintf(stderr, "Bad malloc on motion_vector  \n");
    MYTYPE *Res_orig      = (MYTYPE *) malloc(M*N*F*sizeof(MYTYPE)); if (Res_orig      == NULL) fprintf(stderr, "Bad malloc on Res_orig       \n");
      
    struct pblock_t args;

    get_Y(fid,Yonly,1);
    fclose(fid);

    printf("single thread with config -> M=%d, N=%d, B=%d R=%d\n", M, N, B, R);

    start_timer();
    for (int kk=0; kk<F; kk++){
      for (int ii=0; ii<MB; ii++){
	for (int jj=0; jj<NB; jj++){
          args = pblock_t(Yonly, Yonly, predicted, motion_vector, ii,jj,kk);
	  process_pblock_mt(&args);
#ifdef enable_prints
	  printf("PULKITA sent %d,%d,%d\n",ii,jj,kk);
#endif
	}
      }
    }
    end_timer("static p");
 
 XX:
    write_Y(fid2,predicted);
    write_diff_Y(fid3,Yonly,predicted,0);
    free(Yonly         );
    free(predicted     );
    free(reconstructed );
    free(motion_vector );
    free(Res_orig      );
}
