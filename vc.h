#pragma once
#define _CRT_SECURE_NO_WARNINGS

#define MAX(a,b) ( (a) > (b) ? (a) : (b) )
#define MIN(a,b) ( (a) < (b) ? (a) : (b) )
#define MAX3(a,b,c) ( MAX(MAX((a),(b)),(c)) )
#define MIN3(a,b,c) ( MIN(MIN((a),(b)),(c)) )

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	unsigned char* data;
	int width, height;
	int channels;
	int levels;
	int bytesperline;
} IVC;

typedef struct {
	int x, y, width, height;
	int area;
	int xc, yc;
	int perimetro;
	int label;
} OVC;

IVC* vc_image_new(int width, int height, int channels, int levels);
IVC* vc_image_free(IVC* image);

int vc_bgr_to_rgb(IVC* src);

// Segmentação HSV (calcula HSV internamente a partir de RGB)
int vc_hsv_segmentation(IVC* src_rgb, IVC* dst_bin,
	int hmin, int hmax, int smin, int smax, int vmin, int vmax);

int vc_binary_dilate_tunado(IVC* src, IVC* dst, int size, int min_white_neighbors);

OVC* vc_binary_blob_labelling(IVC* src, IVC* dst, int* nlabels);
int vc_binary_blob_info(IVC* src, OVC* blobs, int nblobs);

#ifdef __cplusplus
}
#endif