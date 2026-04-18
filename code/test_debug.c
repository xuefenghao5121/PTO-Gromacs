#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Minimal test: verify tile-sorted neighbor list produces same pairs */

typedef struct { int natoms; float *x; float box[3]; } GroData;
static GroData* read_gro(const char *fn) {
    FILE *fp = fopen(fn,"r"); if(!fp) return NULL;
    GroData *g = calloc(1,sizeof(GroData)); char line[256];
    fgets(line,sizeof(line),fp); fgets(line,sizeof(line),fp);
    g->natoms = atoi(line);
    g->x = malloc(g->natoms*3*sizeof(float));
    for(int i=0;i<g->natoms;i++){
        fgets(line,sizeof(line),fp); char *p = line+20;
        g->x[i*3+0]=strtof(p,&p); g->x[i*3+1]=strtof(p,&p); g->x[i*3+2]=strtof(p,&p);
    }
    fgets(line,sizeof(line),fp);
    sscanf(line,"%f%f%f",&g->box[0],&g->box[1],&g->box[2]);
    fclose(fp); return g;
}

typedef struct { int *start; int *jatoms; int *count; int total_pairs; } NBList;

static NBList* build_nblist(const float *x, int n, const float box[3], float cut) {
    NBList *nl = calloc(1,sizeof(NBList));
    nl->start = malloc(n*sizeof(int));
    nl->count = calloc(n,sizeof(int));
    float csq = cut*cut; int total=0;
    for(int i=0;i<n;i++) for(int j=i+1;j<n;j++){
        float dx=x[i*3]-x[j*3], dy=x[i*3+1]-x[j*3+1], dz=x[i*3+2]-x[j*3+2];
        dx-=box[0]*roundf(dx/box[0]); dy-=box[1]*roundf(dy/box[1]); dz-=box[2]*roundf(dz/box[2]);
        if(dx*dx+dy*dy+dz*dz < csq) { nl->count[i]++; total++; }
    }
    nl->start[0]=0;
    for(int i=1;i<n;i++) nl->start[i]=nl->start[i-1]+nl->count[i-1];
    nl->jatoms = malloc(total*sizeof(int));
    nl->total_pairs = total;
    int *pos = calloc(n,sizeof(int));
    for(int i=0;i<n;i++) for(int j=i+1;j<n;j++){
        float dx=x[i*3]-x[j*3], dy=x[i*3+1]-x[j*3+1], dz=x[i*3+2]-x[j*3+2];
        dx-=box[0]*roundf(dx/box[0]); dy-=box[1]*roundf(dy/box[1]); dz-=box[2]*roundf(dz/box[2]);
        if(dx*dx+dy*dy+dz*dz < csq) nl->jatoms[nl->start[i]+pos[i]++]=j;
    }
    free(pos); return nl;
}

/* Compute scalar forces */
static void compute_scalar(float *x, float *f, int n, float box[3], NBList *nl, float cut) {
    float csq=cut*cut, eps=0.5f, ssq=0.09f;
    memset(f,0,n*3*sizeof(float));
    for(int i=0;i<n;i++){
        float fx=0,fy=0,fz=0, xi=x[i*3],yi=x[i*3+1],zi=x[i*3+2];
        for(int k=0;k<nl->count[i];k++){
            int j=nl->jatoms[nl->start[i]+k];
            float dx=xi-x[j*3], dy=yi-x[j*3+1], dz=zi-x[j*3+2];
            dx-=box[0]*roundf(dx/box[0]); dy-=box[1]*roundf(dy/box[1]); dz-=box[2]*roundf(dz/box[2]);
            float rsq=dx*dx+dy*dy+dz*dz;
            if(rsq<csq && rsq>1e-8f){
                float ir=1.0f/rsq, s2=ssq*ir, s6=s2*s2*s2, s12=s6*s6;
                float fr=24.0f*eps*(2.0f*s12-s6)*ir;
                fx+=fr*dx; fy+=fr*dy; fz+=fr*dz;
                f[j*3]-=fr*dx; f[j*3+1]-=fr*dy; f[j*3+2]-=fr*dz;
            }
        }
        f[i*3]+=fx; f[i*3+1]+=fy; f[i*3+2]+=fz;
    }
}

/* Compute tile-sorted forces */
static void compute_tile(float *x, float *f, int n, float box[3], NBList *nl, float cut, int tile_size) {
    float csq=cut*cut, eps=0.5f, ssq=0.09f;
    int num_tiles = (n+tile_size-1)/tile_size;
    
    /* SoA */
    float *sx=malloc(n*sizeof(float)), *sy=malloc(n*sizeof(float)), *sz=malloc(n*sizeof(float));
    float *lfx=calloc(n,sizeof(float)), *lfy=calloc(n,sizeof(float)), *lfz=calloc(n,sizeof(float));
    for(int i=0;i<n;i++){sx[i]=x[i*3]; sy[i]=x[i*3+1]; sz[i]=x[i*3+2];}
    
    /* Sort neighbors by tile */
    int *sorted_j = malloc(nl->total_pairs*sizeof(int));
    int **run_start = malloc(n*sizeof(int*));
    int **run_count = malloc(n*sizeof(int*));
    for(int i=0;i<n;i++){
        run_start[i]=calloc(num_tiles+1,sizeof(int));
        run_count[i]=calloc(num_tiles,sizeof(int));
    }
    
    for(int i=0;i<n;i++){
        int cnt=nl->count[i], base=nl->start[i];
        for(int k=0;k<cnt;k++){
            int j=nl->jatoms[base+k];
            int ti=j/tile_size; if(ti>=num_tiles)ti=num_tiles-1;
            run_count[i][ti]++;
        }
        run_start[i][0]=0;
        for(int ti=1;ti<=num_tiles;ti++) run_start[i][ti]=run_start[i][ti-1]+run_count[i][ti-1];
        int *pos=calloc(num_tiles,sizeof(int));
        for(int k=0;k<cnt;k++){
            int j=nl->jatoms[base+k];
            int ti=j/tile_size; if(ti>=num_tiles)ti=num_tiles-1;
            sorted_j[base+run_start[i][ti]+pos[ti]++]=j;
        }
        free(pos);
    }
    
    /* Compute */
    memset(f,0,n*3*sizeof(float));
    for(int i=0;i<n;i++){
        float xi=sx[i],yi=sy[i],zi=sz[i];
        float fx=0,fy=0,fz=0;
        for(int ti=0;ti<num_tiles;ti++){
            int rs=run_start[i][ti], rc=run_count[i][ti];
            for(int k=0;k<rc;k++){
                int j=sorted_j[nl->start[i]+rs+k];
                float dx=xi-sx[j], dy=yi-sy[j], dz=zi-sz[j];
                dx-=box[0]*roundf(dx/box[0]); dy-=box[1]*roundf(dy/box[1]); dz-=box[2]*roundf(dz/box[2]);
                float rsq=dx*dx+dy*dy+dz*dz;
                if(rsq<csq&&rsq>1e-8f){
                    float ir=1.0f/rsq,s2=ssq*ir,s6=s2*s2*s2,s12=s6*s6;
                    float fr=24.0f*eps*(2.0f*s12-s6)*ir;
                    fx+=fr*dx;fy+=fr*dy;fz+=fr*dz;
                    lfx[j]-=fr*dx;lfy[j]-=fr*dy;lfz[j]-=fr*dz;
                }
            }
        }
        lfx[i]+=fx;lfy[i]+=fy;lfz[i]+=fz;
    }
    for(int k=0;k<n;k++){f[k*3+0]=lfx[k];f[k*3+1]=lfy[k];f[k*3+2]=lfz[k];}
    
    /* Verify pair counts */
    int sorted_total=0;
    for(int i=0;i<n;i++) for(int ti=0;ti<num_tiles;ti++) sorted_total+=run_count[i][ti];
    printf("Pair count check: original=%d sorted=%d match=%s\n",
           nl->total_pairs, sorted_total, nl->total_pairs==sorted_total?"YES":"NO");
    
    free(sx);free(sy);free(sz);free(lfx);free(lfy);free(lfz);free(sorted_j);
    for(int i=0;i<n;i++){free(run_start[i]);free(run_count[i]);}
    free(run_start);free(run_count);
}

int main() {
    GroData *g = read_gro("../tests/results/md_test/water_24.gro");
    NBList *nl = build_nblist(g->x, g->natoms, g->box, 1.0f);
    
    float *f1 = malloc(g->natoms*3*sizeof(float));
    float *f2 = malloc(g->natoms*3*sizeof(float));
    
    compute_scalar(g->x, f1, g->natoms, g->box, nl, 1.0f);
    compute_tile(g->x, f2, g->natoms, g->box, nl, 1.0f, 64);
    
    float maxd=0; int max_k=0;
    for(int k=0;k<g->natoms*3;k++){
        float d=fabsf(f1[k]-f2[k]);
        if(d>maxd){maxd=d;max_k=k;}
    }
    printf("Max force diff: %.6e at index %d (atom %d, comp %d)\n", maxd, max_k, max_k/3, max_k%3);
    printf("  scalar: %.6f tile: %.6f\n", f1[max_k], f2[max_k]);
    
    /* Check a few atoms */
    for(int i=0;i<5;i++){
        printf("atom %d: scalar=(%.4f,%.4f,%.4f) tile=(%.4f,%.4f,%.4f)\n",
               i, f1[i*3],f1[i*3+1],f1[i*3+2], f2[i*3],f2[i*3+1],f2[i*3+2]);
    }
    
    return 0;
}
