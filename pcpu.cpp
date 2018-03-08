#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <stdint.h>

#include "cpu_fns.cpp"

int main(){

    FILE    *fid           = fopen("backup/akiyo_qcif.y","rb");
    FILE    *fid2          = fopen("akiyo_qcif_ppred.y","wb");
    FILE    *fid3          = fopen("akiyo_qcif_res.y","wb");
    uint8_t *Yonly         = (uint8_t *) malloc(M*N*F*sizeof(uint8_t)); if (Yonly         == NULL) fprintf(stderr, "Bad malloc on Yonly          \n");
    uint8_t *predicted     = (uint8_t *) malloc(M*N*F*sizeof(uint8_t)); if (predicted     == NULL) fprintf(stderr, "Bad malloc on predicted      \n");
    uint8_t *reconstructed = (uint8_t *) malloc(M*N*F*sizeof(uint8_t)); if (reconstructed == NULL) fprintf(stderr, "Bad malloc on reconstructed  \n");
    uint8_t *motion_vector = (uint8_t *) malloc(M*N*F*14*sizeof(uint8_t)); if (motion_vector == NULL) fprintf(stderr, "Bad malloc on motion_vector  \n");
    uint8_t *Res_orig      = (uint8_t *) malloc(M*N*F*sizeof(uint8_t)); if (Res_orig      == NULL) fprintf(stderr, "Bad malloc on Res_orig       \n");
      
    get_Y(fid,Yonly,1);
    fclose(fid);

    for (int kk=0; kk<F; kk++){
      start_timer();
      for (int ii=0; ii<MB; ii++){
	for (int jj=0; jj<NB; jj++){
	  // if (ii==1) goto XX;
	  // printf("%u, %u, %u \n",ii,jj, kk);
	  // process_pblock(Yonly, reconstructed, predicted, motion_vector, ii,jj,kk);
	  process_pblock(Yonly, Yonly, predicted, motion_vector, ii,jj,kk);
	}
      }
      end_timer("static p");
    }

 XX:
    write_Y(fid2,predicted);
    write_diff_Y(fid3,Yonly,predicted);
    free(Yonly         );
    free(predicted     );
    free(reconstructed );
    free(motion_vector );
    free(Res_orig      );
}
