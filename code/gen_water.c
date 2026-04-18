/* Generate a water box GRO file with N water molecules */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main(int argc, char *argv[]) {
    int nw = argc > 1 ? atoi(argv[1]) : 1000;
    float density = 33.4f; /* waters/nm^3 */
    float box = powf(nw / density, 1.0f/3.0f);
    float spacing = box / powf(nw, 1.0f/3.0f);
    
    printf("Generated water box\n%d\n", nw * 3);
    int idx = 1;
    for (int w = 0; w < nw; w++) {
        int gx = w % (int)(powf(nw, 1.0f/3.0f) + 0.5f);
        int gy = (w / (int)(powf(nw, 1.0f/3.0f) + 0.5f)) % (int)(powf(nw, 1.0f/3.0f) + 0.5f);
        int gz = w / ((int)(powf(nw, 1.0f/3.0f) + 0.5f) * (int)(powf(nw, 1.0f/3.0f) + 0.5f));
        float ox = (gx + 0.5f) * spacing;
        float oy = (gy + 0.5f) * spacing;
        float oz = (gz + 0.5f) * spacing;
        /* O */
        printf("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n", w+1, "SOL", "OW", idx++, ox, oy, oz);
        /* H1 */
        printf("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n", w+1, "SOL", "HW1", idx++, ox+0.057f, oy+0.075f, oz);
        /* H2 */
        printf("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n", w+1, "SOL", "HW2", idx++, ox-0.075f, oy+0.057f, oz);
    }
    printf("%10.5f%10.5f%10.5f\n", box, box, box);
    return 0;
}
