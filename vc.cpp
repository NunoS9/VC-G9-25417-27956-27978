#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "vc.h"

IVC* vc_image_new(int width, int height, int channels, int levels)
{
	IVC* image = (IVC*)malloc(sizeof(IVC));
	if (image == NULL) return NULL;
	if ((levels <= 0) || (levels > 255)) return NULL;

	image->width = width;
	image->height = height;
	image->channels = channels;
	image->levels = levels;
	image->bytesperline = image->width * image->channels;

	image->data = (unsigned char*)malloc(image->width * image->height * image->channels);
	if (image->data == NULL)
	{
		return vc_image_free(image);
	}
	return image;
}

IVC* vc_image_free(IVC* image)
{
	if (image != NULL)
	{
		if (image->data != NULL)
		{
			free(image->data);
			image->data = NULL;
		}
		free(image);
	}
	return NULL;
}

int vc_bgr_to_rgb(IVC* src)
{
	if (!src || !src->data) return 0;
	if (src->channels != 3) return 0;

	unsigned char* d = (unsigned char*)src->data;
	int width = src->width;
	int height = src->height;
	int bpl = src->bytesperline;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			long pos = y * bpl + x * 3;
			unsigned char B = d[pos + 0];
			unsigned char G = d[pos + 1];
			unsigned char R = d[pos + 2];
			d[pos + 0] = R;
			d[pos + 1] = G;
			d[pos + 2] = B;
		}
	}
	return 1;
}

// Segmentação HSV: src é RGB, dst é binária (1 canal)
// h: graus [0..360], s: % [0..100], v: % [0..100]
int vc_hsv_segmentation(IVC* src, IVC* dst, int hmin, int hmax, int smin, int smax, int vmin, int vmax)
{
	if (!src || !dst || !src->data || !dst->data) return 0;
	if (src->width != dst->width || src->height != dst->height) return 0;
	if (src->channels != 3 || dst->channels != 1) return 0;

	unsigned char* sdata = src->data;
	unsigned char* ddata = dst->data;
	int width = src->width;
	int height = src->height;
	int sbpl = src->bytesperline;
	int dbpl = dst->bytesperline;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			long pS = y * sbpl + x * 3;
			long pD = y * dbpl + x;

			float r = (float)sdata[pS + 0];
			float g = (float)sdata[pS + 1];
			float b = (float)sdata[pS + 2];

			float Maxv = MAX3(r, g, b);
			float Minv = MIN3(r, g, b);
			float V = Maxv; // 0..255

			float S, H;

			if (Maxv == 0.0f || Maxv == Minv)
			{
				S = 0.0f;
				H = 0.0f;
			}
			else
			{
				S = (Maxv - Minv) * 100.0f / Maxv; // 0..100

				if (r == Maxv && g >= b) H = 60.0f * (g - b) / (Maxv - Minv);
				else if (r == Maxv && b > g) H = 360.0f + 60.0f * (g - b) / (Maxv - Minv);
				else if (g == Maxv) H = 120.0f + 60.0f * (b - r) / (Maxv - Minv);
				else /* Maxv==b */ H = 240.0f + 60.0f * (r - g) / (Maxv - Minv);
			}

			// converter V para percent (como no código do ano passado)
			float Vp = (V / 255.0f) * 100.0f;

			int okH;
			if (hmin > hmax)
				okH = ((H >= 0.0f && H <= (float)hmax) || (H >= (float)hmin && H <= 360.0f));
			else
				okH = (H >= (float)hmin && H <= (float)hmax);

			int ok = okH &&
				(S >= (float)smin && S <= (float)smax) &&
				(Vp >= (float)vmin && Vp <= (float)vmax);

			ddata[pD] = ok ? 255 : 0;
		}
	}
	return 1;
}

int vc_binary_dilate_tunado(IVC* src, IVC* dst, int size, int min_white_neighbors)
{
	if (!src || !dst || !src->data || !dst->data) return 0;
	if (src->width != dst->width || src->height != dst->height) return 0;
	if (src->channels != 1 || dst->channels != 1) return 0;

	unsigned char* s = src->data;
	unsigned char* d = dst->data;
	int width = src->width;
	int height = src->height;
	int bpl = src->bytesperline;

	int s2 = (size - 1) / 2;
	int s1 = -s2;

	memcpy(d, s, bpl * height);

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			long pos = y * bpl + x;

			int count = 0;
			for (int yk = s1; yk <= s2; yk++)
			{
				int yy = y + yk;
				if (yy < 0 || yy >= height) continue;

				for (int xk = s1; xk <= s2; xk++)
				{
					int xx = x + xk;
					if (xx < 0 || xx >= width) continue;

					long posk = yy * bpl + xx;
					if (s[posk] == 255) count++;
				}
			}

			if (count >= min_white_neighbors) d[pos] = 255;
		}
	}
	return 1;
}

OVC* vc_binary_blob_labelling(IVC* src, IVC* dst, int* nlabels)
{
	if (!src || !dst || !src->data || !dst->data || !nlabels) return NULL;
	if (src->width != dst->width || src->height != dst->height) return NULL;
	if (src->channels != 1 || dst->channels != 1) return NULL;

	unsigned char* datasrc = src->data;
	unsigned char* datadst = dst->data;
	int width = src->width;
	int height = src->height;
	int bpl = src->bytesperline;

	memcpy(datadst, datasrc, bpl * height);

	// Normalizar para 0/255
	for (long i = 0, size = (long)bpl * height; i < size; i++)
		if (datadst[i] != 0) datadst[i] = 255;

	// limpar bordas
	for (int y = 0; y < height; y++)
	{
		datadst[y * bpl + 0] = 0;
		datadst[y * bpl + (width - 1)] = 0;
	}
	for (int x = 0; x < width; x++)
	{
		datadst[0 * bpl + x] = 0;
		datadst[(height - 1) * bpl + x] = 0;
	}

	int* labeltable = (int*)calloc(width * height, sizeof(int));
	if (!labeltable) return NULL;

	int label = 1;

	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			long posA = (y - 1) * bpl + (x - 1);
			long posB = (y - 1) * bpl + (x);
			long posC = (y - 1) * bpl + (x + 1);
			long posD = (y)*bpl + (x - 1);
			long posX = (y)*bpl + (x);

			if (datadst[posX] != 0)
			{
				if (datadst[posA] == 0 && datadst[posB] == 0 && datadst[posC] == 0 && datadst[posD] == 0)
				{
					datadst[posX] = (unsigned char)label;
					labeltable[label] = label;
					label++;
				}
				else
				{
					int num = width * height;

					if (datadst[posA] != 0) num = labeltable[datadst[posA]];
					if (datadst[posB] != 0 && labeltable[datadst[posB]] < num) num = labeltable[datadst[posB]];
					if (datadst[posC] != 0 && labeltable[datadst[posC]] < num) num = labeltable[datadst[posC]];
					if (datadst[posD] != 0 && labeltable[datadst[posD]] < num) num = labeltable[datadst[posD]];

					datadst[posX] = (unsigned char)num;
					labeltable[num] = num;

					// equivalências
					int neigh[4] = { datadst[posA], datadst[posB], datadst[posC], datadst[posD] };
					for (int k = 0; k < 4; k++)
					{
						int L = neigh[k];
						if (L != 0 && labeltable[L] != num)
						{
							int old = labeltable[L];
							for (int a = 1; a < label; a++)
								if (labeltable[a] == old) labeltable[a] = num;
						}
					}
				}
			}
		}
	}

	// relabel final
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			long posX = y * bpl + x;
			if (datadst[posX] != 0)
				datadst[posX] = (unsigned char)labeltable[datadst[posX]];
		}
	}

	// contar labels únicas
	for (int a = 1; a < label - 1; a++)
		for (int b = a + 1; b < label; b++)
			if (labeltable[a] == labeltable[b]) labeltable[b] = 0;

	*nlabels = 0;
	for (int a = 1; a < label; a++)
	{
		if (labeltable[a] != 0)
		{
			labeltable[*nlabels] = labeltable[a];
			(*nlabels)++;
		}
	}

	if (*nlabels == 0)
	{
		free(labeltable);
		return NULL;
	}

	OVC* blobs = (OVC*)calloc((*nlabels), sizeof(OVC));
	if (!blobs)
	{
		free(labeltable);
		return NULL;
	}

	for (int a = 0; a < (*nlabels); a++)
		blobs[a].label = labeltable[a];

	free(labeltable);
	return blobs;
}

int vc_binary_blob_info(IVC* src, OVC* blobs, int nblobs)
{
	if (!src || !src->data || !blobs) return 0;
	if (src->channels != 1) return 0;

	unsigned char* data = src->data;
	int width = src->width;
	int height = src->height;
	int bpl = src->bytesperline;

	for (int i = 0; i < nblobs; i++)
	{
		int xmin = width - 1, ymin = height - 1;
		int xmax = 0, ymax = 0;
		long sumx = 0, sumy = 0;

		blobs[i].area = 0;
		blobs[i].perimetro = 0;

		for (int y = 1; y < height - 1; y++)
		{
			for (int x = 1; x < width - 1; x++)
			{
				long pos = y * bpl + x;

				if (data[pos] == blobs[i].label)
				{
					blobs[i].area++;
					sumx += x;
					sumy += y;

					if (x < xmin) xmin = x;
					if (y < ymin) ymin = y;
					if (x > xmax) xmax = x;
					if (y > ymax) ymax = y;

					// perímetro (4-vizinhos)
					if (data[pos - 1] != blobs[i].label ||
						data[pos + 1] != blobs[i].label ||
						data[pos - bpl] != blobs[i].label ||
						data[pos + bpl] != blobs[i].label)
					{
						blobs[i].perimetro++;
					}
				}
			}
		}

		blobs[i].x = xmin;
		blobs[i].y = ymin;
		blobs[i].width = (xmax - xmin) + 1;
		blobs[i].height = (ymax - ymin) + 1;

		blobs[i].xc = (int)(sumx / MAX(blobs[i].area, 1));
		blobs[i].yc = (int)(sumy / MAX(blobs[i].area, 1));
	}

	return 1;
}