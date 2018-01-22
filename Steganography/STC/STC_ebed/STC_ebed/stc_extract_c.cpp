#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include "stc_extract_c.h"

int stc_extract(const u8*vector, int vectorlength, u8 *message, int syndromelength, int matrixheight)
{
	int i, j, k, index, index2, base, height;
	
	u8 *binmat[2];
	int *matrices, *widths;

	height = matrixheight;

	if(matrixheight > 31) {
		fprintf(stderr, "Submatrix height must not exceed 31.");
		return -1;
	}

	{
		double invalpha;
		int shorter, longer, worm;
		u32 *columns[2];

		matrices = (int *)malloc(syndromelength * sizeof(int));
		widths = (int *)malloc(syndromelength * sizeof(int));

		invalpha = (double)vectorlength / syndromelength;
		if(invalpha < 1) {
			fprintf(stderr, "The message cannot be longer than the cover object.\n");
			return -1;
		}
		shorter = (int)floor(invalpha);
		longer = (int)ceil(invalpha);
		if((columns[0] = getMatrix(shorter, matrixheight)) == NULL) {
			free(widths);
			free(matrices);
			return -1;
		}
		if((columns[1] = getMatrix(longer, matrixheight)) == NULL) {
			free(columns[0]);
			free(widths);
			free(matrices);
			return -1;
		}
		worm = 0;
		for(i = 0; i < syndromelength; i++) {
			if(worm + longer <= (i + 1) * invalpha + 0.5) {
				matrices[i] = 1;
				widths[i] = longer;
				worm += longer;
			} else {
				matrices[i] = 0;
				widths[i] = shorter;
				worm += shorter;
			}
		}

		//之前的代码与stc_embed_c.cpp一样

		//binmat[]存储submatrix的列，存储顺序是每一列从上往下、每列依次存放。
		binmat[0] = (u8*)malloc(shorter * matrixheight * sizeof(u8));
		binmat[1] = (u8*)malloc(longer * matrixheight * sizeof(u8));
		for(i = 0, index = 0; i < shorter; i++) {
			for(j = 0; j < matrixheight; j++, index++) {
				binmat[0][index] = (columns[0][i] & (1 << j)) ? 1 : 0;
			}
		}
		for(i = 0, index = 0; i < longer; i++) {
			for(j = 0; j < matrixheight; j++, index++) {
				binmat[1][index] = (columns[1][i] & (1 << j)) ? 1 : 0;
			}
		}
		free(columns[0]);
		free(columns[1]);
	}

	for(i = 0; i < syndromelength; i++) {
		message[i] = 0;
	}

	//这里提取秘密信息比特，没有采用 parity-check matrix 与cover的比特直接相乘的方法，而是根据每一个cover比特至多影响matrixheight个秘密信息比特的原理，
	//以 parity-check matrix 中 submatrix 的列为单位，每经过 matrixheight 次加法（异或）确定一个秘密信息比特。
	for(index = 0, index2 = 0; index2 < syndromelength; index2++) {
		for(k = 0, base = 0; k < widths[index2]; k++, index++, base += matrixheight) {
			if(vector[index]) {
				for(i = 0; i < height; i++) {
					message[index2 + i] ^= binmat[matrices[index2]][base + i];
				}
			}
		}
		if(syndromelength - index2 <= matrixheight)
			height--;
	}

	free(matrices);
	free(widths);
	free(binmat[0]);
	free(binmat[1]);

	return 0;
}
