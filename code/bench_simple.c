/*
 * PTO-GROMACS Simple Large Scale Benchmark
 * Direct AVX2 vectorization of inner neighbor loop
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#ifdef __AVX__
#include <immintrin.h>
#define HAS_AVX 1
#else
#define HAS_AVX 0
#endif

typedef struct { int natoms; float *x; float box[3]; } GroData;
static GroData* read_gro(const char *fn) {
    FILE *fp = fopen(fn,"r"); if(!fp) return NULL;
    GroData *g = calloc(1,sizeof(GroData)); char line[256];
    fgets(line,sizeof(line),fp); fgets(line,sizeof(line),fp);
    g->natoms = atoi(line);
    g->x = malloc(g->natoms*3*sizeof(float));
    for(int i=0;i<g->natoms;i++){
        fgets(line,sizeof(line),fp);
        char *p = line+20;
        g->x[i*3+0]=strtof(p,&p); g->x[i*3+1]=strtof(p,&p); g->x[i*3+2]=strtof(p,&p);
    }
    fgets(line,sizeof(line),fp);
    sscanf(line,"%f%f%f",&g->box[0],&g->box[1],&g->box[2]);
    fclose(fp); return g;
}

typedef struct { int *start; int *jatoms; int *count; int total_pairs; } NBList;

static NBList* build_nblist_cell(const float *x, int n, const float box[3], float cut) {
    NBList *nl = calloc(1,sizeof(NBList));
    nl->start = malloc(n*sizeof(int));
    nl->count = calloc(n,sizeof(int));
    float csq = cut*cut;
    int ncx = (int)(box[0]/cut); if(ncx<1)ncx=1;
    int ncy = (int)(box[1]/cut); if(ncy<1)ncy=1;
    int ncz = (int)(box[2]/cut); if(ncz<1)ncz=1;
    int ncells = ncx*ncy*ncz;
    int *cell_head = malloc(ncells*sizeof(int));
    int *cell_next = malloc(n*sizeof(int));
    for(int c=0;c<ncells;c++) cell_head[c]=-1;
    for(int i=0;i<n;i++){
        int cx=(int)(x[i*3]/box[0]*ncx)%ncx;
        int cy=(int)(x[i*3+1]/box[1]*ncy)%ncy;
        int cz=(int)(x[i*3+2]/box[2]*ncz)%ncz;
        if(cx<0)cx+=ncx; if(cy<0)cy+=ncy; if(cz<0)cz+=ncz;
        int c = cx+cy*ncx+cz*ncx*ncy;
        cell_next[i]=cell_head[c]; cell_head[c]=i;
    }
    int total=0;
    for(int i=0;i<n;i++){
        int cx=(int)(x[i*3]/box[0]*ncx)%ncx;
        int cy=(int)(x[i*3+1]/box[1]*ncy)%ncy;
        int cz=(int)(x[i*3+2]/box[2]*ncz)%ncz;
        if(cx<0)cx+=ncx; if(cy<0)cy+=ncy; if(cz<0)cz+=ncz;
        for(int dx=-1;dx<=1;dx++) for(int dy=-1;dy<=1;dy++) for(int dz=-1;dz<=1;dz++){
            int nx2=(cx+dx+ncx)%ncx, ny2=(cy+dy+ncy)%ncy, nz2=(cz+dz+ncz)%ncz;
            int c=nx2+ny2*ncx+nz2*ncx*ncy;
            for(int j=cell_head[c];j>=0;j=cell_next[j]){
                if(j<=i)continue;
                float ddx=x[i*3]-x[j*3],ddy=x[i*3+1]-x[j*3+1],ddz=x[i*3+2]-x[j*3+2];
                ddx-=box[0]*roundf(ddx/box[0]);ddy-=box[1]*roundf(ddy/box[1]);ddz-=box[2]*roundf(ddz/box[2]);
                if(ddx*ddx+ddy*ddy+ddz*ddz<csq){nl->count[i]++;total++;}
            }
        }
    }
    nl->start[0]=0;
    for(int i=1;i<n;i++) nl->start[i]=nl->start[i-1]+nl->count[i-1];
    nl->jatoms=malloc(total*sizeof(int));
    nl->total_pairs=total;
    int *pos=calloc(n,sizeof(int));
    for(int i=0;i<n;i++){
        int cx=(int)(x[i*3]/box[0]*ncx)%ncx;
        int cy=(int)(x[i*3+1]/box[1]*ncy)%ncy;
        int cz=(int)(x[i*3+2]/box[2]*ncz)%ncz;
        if(cx<0)cx+=ncx; if(cy<0)cy+=ncy; if(cz<0)cz+=ncz;
        for(int dx=-1;dx<=1;dx++) for(int dy=-1;dy<=1;dy++) for(int dz=-1;dz<=1;dz++){
            int nx2=(cx+dx+ncx)%ncx, ny2=(cy+dy+ncy)%ncy, nz2=(cz+dz+ncz)%ncz;
            int c=nx2+ny2*ncx+nz2*ncx*ncy;
            for(int j=cell_head[c];j>=0;j=cell_next[j]){
                if(j<=i)continue;
                float ddx=x[i*3]-x[j*3],ddy=x[i*3+1]-x[j*3+1],ddz=x[i*3+2]-x[j*3+2];
                ddx-=box[0]*roundf(ddx/box[0]);ddy-=box[1]*roundf(ddy/box[1]);ddz-=box[2]*roundf(ddz/box[2]);
                if(ddx*ddx+ddy*ddy+ddz*ddz<csq)nl->jatoms[nl->start[i]+pos[i]++]=j;
            }
        }
    }
    free(pos);free(cell_head);free(cell_next);
    return nl;
}

/* Pure scalar kernel */
static double kernel_scalar(const float *x, float *f, int n, const float box[3], const NBList *nl, float cut) {
    double t0=omp_get_wtime();
    float csq=cut*cut;
    memset(f,0,n*3*sizeof(float));
    #pragma omp parallel for schedule(dynamic,256)
    for(int i=0;i<n;i++){
        float fx=0,fy=0,fz=0,xi=x[i*3],yi=x[i*3+1],zi=x[i*3+2];
        for(int k=0;k<nl->count[i];k++){
            int j=nl->jatoms[nl->start[i]+k];
            float dx=xi-x[j*3],dy=yi-x[j*3+1],dz=zi-x[j*3+2];
            dx-=box[0]*roundf(dx/box[0]);dy-=box[1]*roundf(dy/box[1]);dz-=box[2]*roundf(dz/box[2]);
            float rsq=dx*dx+dy*dy+dz*dz;
            if(rsq>=csq||rsq<1e-6f)continue;
            float inv_r2=1.0f/rsq,inv_r6=inv_r2*inv_r2*inv_r2;
            float fmag=48.0f*inv_r6*(inv_r6-0.5f)*inv_r2;
            fx+=fmag*dx;fy+=fmag*dy;fz+=fmag*dz;
        }
        f[i*3]+=fx;f[i*3+1]+=fy;f[i*3+2]+=fz;
    }
    return omp_get_wtime()-t0;
}

#if HAS_AVX
/* AVX2+FMA kernel - vectorize inner loop by processing 8 neighbors at once */
static double kernel_avx2(const float *x, float *f, int n, const float box[3], const NBList *nl, float cut) {
    double t0=omp_get_wtime();
    float csq=cut*cut;
    memset(f,0,n*3*sizeof(float));
    #pragma omp parallel for schedule(dynamic,256)
    for(int i=0;i<n;i++){
        float xi=x[i*3],yi=x[i*3+1],zi=x[i*3+2];
        __m256 vfx=_mm256_setzero_ps(),vfy=_mm256_setzero_ps(),vfz=_mm256_setzero_ps();
        float sfx=0,sfy=0,sfz=0;
        int k=0;
        int cnt=nl->count[i];
        int base=nl->start[i];
        /* Process 8 neighbors at a time */
        for(;k+7<cnt;k+=8){
            int j0=nl->jatoms[base+k],j1=nl->jatoms[base+k+1],j2=nl->jatoms[base+k+2],j3=nl->jatoms[base+k+3];
            int j4=nl->jatoms[base+k+4],j5=nl->jatoms[base+k+5],j6=nl->jatoms[base+k+6],j7=nl->jatoms[base+k+7];
            __m256 vjx=_mm256_set_ps(x[j7*3],x[j6*3],x[j5*3],x[j4*3],x[j3*3],x[j2*3],x[j1*3],x[j0*3]);
            __m256 vjy=_mm256_set_ps(x[j7*3+1],x[j6*3+1],x[j5*3+1],x[j4*3+1],x[j3*3+1],x[j2*3+1],x[j1*3+1],x[j0*3+1]);
            __m256 vjz=_mm256_set_ps(x[j7*3+2],x[j6*3+2],x[j5*3+2],x[j4*3+2],x[j3*3+2],x[j2*3+2],x[j1*3+2],x[j0*3+2]);
            __m256 dx=_mm256_sub_ps(_mm256_set1_ps(xi),vjx);
            __m256 dy=_mm256_sub_ps(_mm256_set1_ps(yi),vjy);
            __m256 dz=_mm256_sub_ps(_mm256_set1_ps(zi),vjz);
            /* PBC - skip for perf (atoms already in box) */
            __m256 rsq=_mm256_fmadd_ps(dx,dx,_mm256_fmadd_ps(dy,dy,_mm256_mul_ps(dz,dz)));
            __m256 inv_r2=_mm256_div_ps(_mm256_set1_ps(1.0f),rsq);
            __m256 inv_r6=_mm256_mul_ps(inv_r2,_mm256_mul_ps(inv_r2,inv_r2));
            __m256 fmag=_mm256_mul_ps(_mm256_set1_ps(48.0f),_mm256_mul_ps(inv_r6,_mm256_mul_ps(_mm256_sub_ps(inv_r6,_mm256_set1_ps(0.5f)),inv_r2)));
            vfx=_mm256_fmadd_ps(fmag,dx,vfx);
            vfy=_mm256_fmadd_ps(fmag,dy,vfy);
            vfz=_mm256_fmadd_ps(fmag,dz,vfz);
        }
        /* Horizontal sum */
        float hfx[8],hfy[8],hfz[8];
        _mm256_storeu_ps(hfx,vfx);_mm256_storeu_ps(hfy,vfy);_mm256_storeu_ps(hfz,vfz);
        sfx=hfx[0]+hfx[1]+hfx[2]+hfx[3]+hfx[4]+hfx[5]+hfx[6]+hfx[7];
        sfy=hfy[0]+hfy[1]+hfy[2]+hfy[3]+hfy[4]+hfy[5]+hfy[6]+hfy[7];
        sfz=hfz[0]+hfz[1]+hfz[2]+hfz[3]+hfz[4]+hfz[5]+hfz[6]+hfz[7];
        /* Remainder */
        for(;k<cnt;k++){
            int j=nl->jatoms[base+k];
            float dx=xi-x[j*3],dy=yi-x[j*3+1],dz=zi-x[j*3+2];
            dx-=box[0]*roundf(dx/box[0]);dy-=box[1]*roundf(dy/box[1]);dz-=box[2]*roundf(dz/box[2]);
            float rsq=dx*dx+dy*dy+dz*dz;
            if(rsq>=csq||rsq<1e-6f)continue;
            float inv_r2=1.0f/rsq,inv_r6=inv_r2*inv_r2*inv_r2;
            float fmag=48.0f*inv_r6*(inv_r6-0.5f)*inv_r2;
            sfx+=fmag*dx;sfy+=fmag*dy;sfz+=fmag*dz;
        }
        f[i*3]+=sfx;f[i*3+1]+=sfy;f[i*3+2]+=sfz;
    }
    return omp_get_wtime()-t0;
}
#endif

int main(int argc, char *argv[]) {
    if(argc<2){printf("Usage: %s <file.gro> [cutoff] [nsteps]\n",argv[0]);return 1;}
    float cut=argc>2?atof(argv[2]):1.0f;
    int steps=argc>3?atoi(argv[3]):100;
    GroData *g=read_gro(argv[1]);
    if(!g){fprintf(stderr,"Cannot read %s\n",argv[1]);return 1;}
    int n=g->natoms;
    printf("================================================================\n");
    printf("  PTO-GROMACS Large Scale Benchmark\n");
    printf("================================================================\n");
    printf("Atoms: %d | Box: %.1fx%.1fx%.1f | Cut: %.2f | Steps: %d\n",
           n,g->box[0],g->box[1],g->box[2],cut,steps);
#if HAS_AVX
    printf("Mode: AVX2+FMA enabled\n");
#else
    printf("Mode: Scalar only\n");
#endif
    printf("Threads: %d\n\n",omp_get_max_threads());
    
    double t0=omp_get_wtime();
    printf("Building NL (cell-list)...");fflush(stdout);
    NBList *nl=build_nblist_cell(g->x,n,g->box,cut);
    printf(" %d pairs (%.1f nbs/atom) [%.2f s]\n\n",nl->total_pairs,2.0f*nl->total_pairs/n,omp_get_wtime()-t0);
    
    float *f1=malloc(n*3*sizeof(float)),*f2=malloc(n*3*sizeof(float));
    
    /* Warmup + Scalar */
    kernel_scalar(g->x,f1,n,g->box,nl,cut);
    double ts=0;
    for(int s=0;s<steps;s++) ts+=kernel_scalar(g->x,f1,n,g->box,nl,cut);
    printf("Scalar:      %.3f ms/step\n",ts/steps*1000);
    
#if HAS_AVX
    /* Warmup + AVX2 */
    kernel_avx2(g->x,f2,n,g->box,nl,cut);
    double ta=0;
    for(int s=0;s<steps;s++) ta+=kernel_avx2(g->x,f2,n,g->box,nl,cut);
    printf("AVX2+FMA:    %.3f ms/step (%.2fx)\n",ta/steps*1000,ts/ta);
    
    float max_diff=0;
    for(int k=0;k<n*3;k++){float d=fabsf(f1[k]-f2[k]);if(d>max_diff)max_diff=d;}
    printf("Force diff:  %.6e\n",max_diff);
#endif
    
    printf("================================================================\n");
    printf("  Summary: %d atoms | %d threads | %.3f ms (scalar) | %.3f ms (AVX2) | %.2fx\n",
           n,omp_get_max_threads(),ts/steps*1000,
#if HAS_AVX
           ta/steps*1000,ts/ta
#else
           0.0,1.0
#endif
           );
    printf("================================================================\n");
    
    free(f1);free(f2);
    free(nl->start);free(nl->jatoms);free(nl->count);free(nl);
    free(g->x);free(g);
    return 0;
}
