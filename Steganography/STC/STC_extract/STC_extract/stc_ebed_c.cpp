#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <limits>
#include <emmintrin.h>
#include <cstdio>
#include "stc_embed_c.h"

void *aligned_malloc(unsigned int bytes, int align)
{
	int shift;
	char *temp = (char *)malloc(bytes + align);

	if(temp == NULL)
		return temp;
	shift = align - (int)(((unsigned long long)temp) & (align - 1));
	temp = temp + shift;
	temp[-1] = shift;
	return (void *)temp;
}

void aligned_free(void *vptr)
{
	char *ptr = (char *)vptr;
	free(ptr - ptr[-1]);
	return;
}

inline __m128i maxLessThan255(const __m128i v1, const __m128i v2)
{
	register __m128i mask = _mm_set1_epi32(0xffffffff);
	return _mm_max_epu8(_mm_andnot_si128(_mm_cmpeq_epi8(v1, mask), v1), _mm_andnot_si128(_mm_cmpeq_epi8(v2, mask), v2));
}

inline u8 max16B(__m128i maxp)
{
	u8 mtemp[4];
	maxp = _mm_max_epu8(maxp, _mm_srli_si128(maxp, 8));
	maxp = _mm_max_epu8(maxp, _mm_srli_si128(maxp, 4));
	*((int*)mtemp) = _mm_cvtsi128_si32(maxp);
	if(mtemp[2] > mtemp[0])
		mtemp[0] = mtemp[2];
	if(mtemp[3] > mtemp[1])
		mtemp[1] = mtemp[3];
	if(mtemp[1] > mtemp[0])
		return mtemp[1];
	else 
		return mtemp[0];
}

inline u8 min16B(__m128i minp)
{
	u8 mtemp[4];
	minp = _mm_min_epu8(minp, _mm_srli_si128(minp, 8));
	minp = _mm_min_epu8(minp, _mm_srli_si128(minp, 4));
	*((int*)mtemp) = _mm_cvtsi128_si32(minp);
	if(mtemp[2] < mtemp[0])
		mtemp[0] = mtemp[2];
	if(mtemp[3] < mtemp[1])
		mtemp[1] = mtemp[3];
	if(mtemp[1] < mtemp[0])
		return mtemp[1];
	else 
		return mtemp[0];
}

double stc_embed(const u8 *vector, int vectorlength, u8 *syndrome, int syndromelength, const void *pricevectorv, bool usefloat, u8 *stego, int matrixheight)
{
	int height, i, k, l, index, index2, parts, m, sseheight, altm, pathindex;
	u32 column, colmask, state;
	double totalprice; //����·�����ܴ���

	u8 *ssedone;
	u32 *path, *columns[2];
	int *matrices, *widths;

	if(matrixheight > 31) {
		fprintf(stderr, "Submatrix height must not exceed 31.");
		return -1;
	}

	height = 1 << matrixheight;
	colmask = height - 1;
	height = (height + 31) & (~31);//��heightС��32ʱ����heightǿ�Ʊ��32����matrixheight����С��5

	parts = height >> 5;
	
	if(stego != NULL) {
		path = (u32*)malloc(vectorlength * parts * sizeof(u32));
		if(path == NULL) {
			fprintf(stderr, "Not enough memory (%u byte array could not be allocated).\n", vectorlength * parts * sizeof(u32));
			return -1;
		}
		pathindex = 0;
	}

	{
		int shorter, longer, worm;
		double invalpha;

		matrices = (int *)malloc(syndromelength * sizeof(int));
		widths = (int *)malloc(syndromelength * sizeof(int));

		invalpha = (double)vectorlength / syndromelength;
		if(invalpha < 1) {
			free(matrices);
			free(widths);
			if(stego != NULL)
				free(path);
			fprintf(stderr, "The message cannot be longer than the cover object.\n");
			return -1;
		}
		if(invalpha < 2) {
			printf("The relative payload is greater than 1/2. This may result in poor embedding efficiency.\n");
		}
		shorter = (int)floor(invalpha);
		longer = (int)ceil(invalpha);
		if((columns[0] = getMatrix(shorter, matrixheight)) == NULL) {
			free(matrices);
			free(widths);
			if(stego != NULL)
				free(path);
			return -1;
		}
		if((columns[1] = getMatrix(longer, matrixheight)) == NULL) {
			free(columns[0]);
			free(matrices);
			free(widths);
			if(stego != NULL)
				free(path);
			return -1;
		}

		//��Ϊcover����vectorlength��һ����message����syndromelength����������submatrix�Ŀ��w������shorter��longer��
		//��matrices[i] = 0 �ĸ���Ϊx��matrices[i] = 1 �ĸ���Ϊy�����Ҫʹ�� worm = x * shorter + y * longer = vectorlength
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
	}

	if(usefloat) {
	/*
		SSE FLOAT VERSION
	*/
		int pathindex8 = 0;
		int shift[2] = {0, 4};
		u8 mask[2] = {0xf0, 0x0f};
		float *prices;
		u8 *path8 = (u8*)path;
		double *pricevector = (double*)pricevectorv;
		double total = 0; //����cover���ص��ܴ���
		float inf = std::numeric_limits<float>::infinity();	//���ر�����Ԥ����ġ�������󡱣���1.#INF //Return Value:The representation of positive infinity for a type, if available.

		sseheight = height >> 2; //sseheight��ÿһ��state��ѭ���Ĵ�������Ϊʹ����SSEָ��֮��ÿһ�αȽ�4��state������ܵ�ѭ��������Ҫ����4
		ssedone = (u8*)malloc(sseheight * sizeof(u8)); //ssedone[m]��ȡֵΪ1��Ϊ0����ʾ��ǰ���еĵ�m��state��·���������Ƿ��Ѿ�����������ô�������Ľ���1��
		prices = (float*)aligned_malloc(height * sizeof(float), 16); //prices[]�洢����һ��������state�Ĵ���

		{
			__m128 fillval = _mm_set1_ps(inf); //_mm_set1_ps( float w )����������4��ֵΪͬһ��ֵ��r0 = r1 = r2 = r3 = w

			//�����д���ֵ��ʼ��Ϊ�����
			for(i = 0; i < height; i+= 4) {	
				_mm_store_ps(&prices[i], fillval); //_mm_store_ps(float *p, __m128 a )���ѼĴ��� a ���ĸ������ȸ���ֵ��ֵ��ָ��p��ָ��ĵ�ַ��p[0] = a0, p[1] = a1, p[2] = a2, p[3] = a3
				ssedone[i >> 2] = 0;
			}
		}

		prices[0] = 0.0f; //�ڴ����1��ʱ�����еĵ�1������ֵҪΪ0

		//Viterbi�㷨������ѭ������
		//block��ѭ��
		for(index = 0, index2 = 0; index2 < syndromelength; index2++) {	//index��trellisÿһ�У���Ӧcover��ÿһ��Ԫ�أ�������ֵ��index2��trellisÿһblock����ӦҪǶ���ÿ�����أ�������ֵ
			register __m128 c1, c2;

			//��ѭ��
			for(k = 0; k < widths[index2]; k++, index++) { //widths[index2]��ÿһblock�е�����
				column = columns[matrices[index2]][k] & colmask; //column��submatrix��block�еĵĵ�k�У�Ϊʲô��Ҫ��colmask��λ�����㣬������Ľ���3��

				//c1��ˮƽ·����stego�ı���Ϊ0���Ĵ��ۣ�c2����б·����stego�ı���Ϊ1���Ĵ���
				if(vector[index] == 0) { //���cover�ı���Ϊ0����ˮƽ·������Ҫ�޸�cover�ı��أ���c1��0������б·����Ҫ�޸�cover�ı��أ���Ҫ��Ԥ�ȵõ��Ĵ��۸�ֵ��c2
					c1 = _mm_setzero_ps(); //_mm_setzero_ps(void)������ĸ������ȵĸ���ֵ��r0 := r1 := r2 := r3 := 0.0
					c2 = _mm_set1_ps((float)pricevector[index]);
				} else {
					c1 = _mm_set1_ps((float)pricevector[index]);
					c2 = _mm_setzero_ps();
				}

				total += pricevector[index]; //�ۼ�����cover�����ܵĴ��ۣ�������û���޸�cover���ء�

				//���е�stateѭ��
				for(m = 0; m < sseheight; m++) {
					if(!ssedone[m]) {
						register __m128 v1, v2, v3, v4;
						altm = (m ^ (column >> 2));	//altm�ǽ��뵱ǰstate��·��Ϊ1ʱ��ǰһ�ж�Ӧ��state��������ʵ��state�ĺ���λ������Ϊһ�αȽ�4��state����m����4Ϊ��λ�ģ�����columnҪ����4����������ʱ�ų�����λ��Ӱ�졣
						v1 = _mm_load_ps(&prices[m << 2]); //_mm_load_ps(float * p )����ָ��p��ָ��λ�õ�ǰ�ĸ������ȵĸ���ֵ��ֵ���Ĵ���������֮��r0 = p[0], r1 = p[1], r2 = p[2], r3 = p[3]
						v2 = _mm_load_ps(&prices[altm << 2]); //v1��v2�ֱ��ǽ��뵱ǰstate��·��Ϊ0��1ʱ��ǰһ�ж�Ӧ������state�Ĵ��ۡ���v1��v2��ʵ���ֱ������4��state�Ĵ��ۣ���Ϊһ�αȽ�4��state����prices[]����1Ϊ��λ�ģ�m��altm����4Ϊ��λ������m��altmҪ����2λ��
						v3 = v1; //Ϊʲô���v1��v2����һ�£�������Ľ���1��
						v4 = v2;
						ssedone[m] = 1;
						ssedone[altm] = 1;

						//ǰ����state��column���ɻ��ʱ��û�п��Ǻ��������أ�������switch����м��Կ��ǡ���ǰ���state����
						switch(column & 3) {
						case 0:
							break;
						case 1:
							v2 = _mm_shuffle_ps(v2, v2, 0xb1); //_mm_shuffle_ps(a, b, i)���������� i ���� a �� b ѡ���ĸ��ض��ĵ����ȸ���ֵ����� http://blog.csdn.net/zhuliting/article/details/6009672 �� http://forums.codeguru.com/printthread.php?t=337156
							v3 = _mm_shuffle_ps(v3, v3, 0xb1); ////Ϊʲô���v3(v1)����v2һ���Ĳ�����������Ľ���1��
							break;
						case 2:
							v2 = _mm_shuffle_ps(v2, v2, 0x4e);
							v3 = _mm_shuffle_ps(v3, v3, 0x4e);
							break;
						case 3:
							v2 = _mm_shuffle_ps(v2, v2, 0x1b);
							v3 = _mm_shuffle_ps(v3, v3, 0x1b);
							break;
						/*�� case 1 Ϊ��

						��ǰ�е�4������state��a��b��c��dӳ�䵽ǰһ�е�4������state��e��f��g��h�����Ǿ���������state�໥��������֪����
						    ��ǰ��state     	column			 XOR(state�ĺ���λ��ʮ����)
						a     ......00			......01		  1			f
						b     ......01			......01		  0			e
						c     ......02			......01		  3			h
						d     ......03			......01		  2			g
							  
									   �Ĵ������ĸ��洢��Ԫ
									      a   b   c   d
						                -----------------
							ӳ��ǰ		| 0 | 1 | 2 | 3 |		
						                -----------------
						     ӳ��          \ /     \ /
						     ����          / \     / \
									    -----------------
							ӳ���      | 1 | 0 | 3 | 2 |
									    -----------------
										  f	  e	  h	  g

						v2 = _mm_shuffle_ps(v2, v2, 0xb1) �ǰ�v2���ұ�����2��3��0��1��Ԫ�طֱ��Ƶ��µ�v2�����������ĸ��洢��Ԫ�С�0xb1��ʾ2��3��0��1�����ǴӼĴ�����Ԫ���ұ߿�ʼ����ģ������������������ͼ�е�1��0��3��2.

						*/
						}

						//ǰһ���ж�Ӧstate�Ĵ���v����·������Ĵ���c���õ���ǰ�е�ǰstate�Ĵ���v��
						//����1
						//�����ǰ�еĵ�m��state������ǰһ�еĵ�m��state����Ӧˮƽ·�����͵�altm������Ӧ��б·����state��
						//��ô��ǰ�еĵ�altm��stateҲ������ǰһ�еĵ�m��state����Ӧ��б·�����͵�altm������Ӧˮƽ·����state��
						//������������£�ǰһ���ж�Ӧstate�Ĵ��۶���һ���ģ�ֻ��·���Ĵ��ۻ�ߵ����������ԲŻ���v1 = _mm_add_ps(v1, c1)��v3 = _mm_add_ps(v3, c2)����v1��v3֮ǰ����ȵģ���c1��c2���෴�ġ�
						//��Ϊ�����Ϲ�ϵ�����Ե�ǰ�еĵ�m��state�͵�altm��state��·��������ۿ���ͬʱȷ����
						//��Ҳ��Ϊʲôv1��v2Ҫ����Ϊv3��v4�Լ�Ҫ������ssedone[]��ԭ����Ϊ��m��state�͵�altm��state��һ���������ģ��������ȷ���˵�altm��state��·���ʹ��ۣ��ȱ���forѭ�����е� m == altm ʱ����Ϊssedone[altm]Ϊ1���Ϳ��������ˡ�
						v1 = _mm_add_ps(v1, c1); //_mm_add_ps(a, b)���� a �� b ���ĸ������ȸ���ֵ�ֱ��Ӧ��Ӳ�������ӽ����r0 = a0 + b0, r1 = a1 + b1, r2 = a2 + b2, r3 = a3 + b3
						v2 = _mm_add_ps(v2, c2);
						v3 = _mm_add_ps(v3, c2);
						v4 = _mm_add_ps(v4, c1);

						//ÿһ��state����ǰһ�е�����state�����ӣ�ʹ��ǰstate����������·������������ֵ������ѡ�������С�ġ�
						v1 = _mm_min_ps(v1, v2); //_mm_min_ps(a , b )��ȡ a �� b ���ĸ����Ӧ�����ȸ���ֵ�еĽ�Сֵ�����ؽ����ri = min(ai, bi)��i = 0,1,2,3 
						v4 = _mm_min_ps(v3, v4);
						
						//prices[]ԭ���洢����ǰһ��������state�Ĵ��ۣ���Ϊ��ǰ�еĵ�m��state�͵�altm��state��·��������۶�ͬʱȷ���ˣ�
						//��ôǰһ�еĵ�m��state�͵�altm��state�Ĵ��۾��ò����ˣ����Ը���֮��ʹprices[]�洢��ǰ����state�Ĵ��ۡ�����������α���벻һ���������ǵȵ�ǰ�е�����state��·�������۶�ȷ���󣬲�һ����ȫ������prices[]��
						_mm_store_ps(&prices[m << 2], v1);
						_mm_store_ps(&prices[altm << 2], v4);

						//����2
						//�����state�Ĵ���֮���������¼������state��·����0��1��
						if(stego != NULL) {
							//v1��ԭv1��v2�еĽ�Сֵ��v2�洢������state���ӵ�·��Ϊ1�Ĵ���ֵ��v2 = _mm_cmpeq_ps(v1, v2)�õ��µ�v2��
							//��v2�Ĵ������ĸ���Ԫ�У��洢�����ݱ���˽�С���۶�Ӧ��·����·��0��0x0000��·��1��0xffff��
							v2 = _mm_cmpeq_ps(v1, v2); //_mm_cmpeq_ps (__m128 a , __m128 b )������һ��_m128�ļĴ������ֱ�Ƚ� a �� b ��Ӧλ��32bit�����ȸ������Ƿ���ȣ�����ȣ���λ�÷���0xffff�����򷵻�0x0��ri = (ai == bi) ? 0xffffffff : 0x0��i = 0,1,2,3
							v3 = _mm_cmpeq_ps(v3, v4);
							//path[]�洢���������е�����state��Ӧ��·����0��1������32λΪ��λ��path8[]ָ�����path[]����8λΪ��λ��
							//pathindex��pathindex8��path[]��path8[]ÿһ�е�һ��state�ĵ�ַ����ʼ�׵�ַ����һ�е�һ��state����ƫ�ơ�
							//���������ǵõ�һ��state����4Ϊ��λ����Ӧ��·������4���أ�����path8[]�С�
							//_mm_movemask_ps(v2)�õ���4����Ϊ4��·��������path8[]����8����λ��λ�ģ�����ÿ����state����4Ϊ��λ����һ��8��·��Ҫ�ϲ�Ϊ1���ֽڣ���mask[m & 1]��shift[m & 1]���������ϲ�һ���ֽڵġ�
							//��2��state����4Ϊ��λ����һ��8��·��Ҫ�ϲ�Ϊ1���ֽں�����һ���ֽ������8��·���ı������������еģ����ֽڵ����ұ�һλ��ŵ���8��·���еĵ�һ����Ҳ��state�ǰ��һ������
							//u32�����ڵ�ÿ8��state����4Ϊ��λ����һ��32��·�����������еģ����intת��Ϊchar���ڴ��ֽ�˳���йأ�����u32����֮��Ĵ�������˳�����еģ�����ͼ��ʾ��
							/*
								p[i]��path8[i]

							            |     32bit     |     32bit     |
										|    path[0]    |    path[1]    |
							            ----------------------------------
							     		| 3 | 2 | 1 | 0 | 3 | 2 | 1 | 0 |		
						                ----------------------------------
										p[3]p[2]p[1]p[0]p[7]p[6]p[5]p[4]
							*/
							path8[pathindex8 + (m >> 1)] = (path8[pathindex8 + (m >> 1)] & mask[m & 1]) | (_mm_movemask_ps(v2) << shift[m & 1]); //int _mm_movemask_ps( __m128 a )��r = sign(a3)<<3 | sign(a2)<<2 | sign(a1)<<1 | sign(a0)
							path8[pathindex8 + (altm >> 1)] = (path8[pathindex8 + (altm >> 1)] & mask[altm & 1]) | (_mm_movemask_ps(v3) << shift[altm & 1]);
						}
					} //end of if(!ssedone[m])
				} //end of stateѭ��

				for(i = 0; i < sseheight; i++) { 
					ssedone[i] = 0;
				}

				pathindex += parts;	// parts �� path[] ��ÿһ��state��ƫ�ƾ���
				pathindex8 += parts << 2; // parts << 2 �� path8[] ��ÿһ��state��ƫ�ƾ���
			} //end of ��ѭ��

			//α�����е� prune state ���̣�������ҪǶ��ı��أ�syndrome[index2]��ѡ��ǰblock���һ���е�state���ӵ���һ��block�С�
			//���state��1Ϊ��λ����Ҫѡ������state��ż��state�������state����4Ϊ��λ��һ�ο�������state����4Ϊ��λ����ʵ������һ���Դ�8��state����1Ϊ��λ����ѡ��4��state����1Ϊ��λ��
			if(syndrome[index2] == 0) {
				for(i = 0, l = 0; i < sseheight; i+= 2, l += 4) { //һ�αȽ����ڵ�����__m128�Ĵ���������i += 2��һ�εõ��µ�4�����ۣ�����l += 4��
					_mm_store_ps(&prices[l], _mm_shuffle_ps(_mm_load_ps(&prices[i << 2]), _mm_load_ps(&prices[(i + 1) << 2]), 0x88)); // 0x88��ʾ�������Ĵ������ұ�����ѵ�2���͵�0��Ԫ��ȡ��
				}
			} else {
				for(i = 0, l = 0; i < sseheight; i+= 2, l += 4) {
					_mm_store_ps(&prices[l], _mm_shuffle_ps(_mm_load_ps(&prices[i << 2]), _mm_load_ps(&prices[(i + 1) << 2]), 0xdd)); // 0x88��ʾ�������Ĵ������ұ�����ѵ�3���͵�1��Ԫ��ȡ��
				}
			}
			/*�� syndrome[index2] == 0 Ϊ��

					���ڼĴ�����Ԫ��˳����Ǵ��ҵ���ģ����Ե� syndrome[index2] == 0 ʱ��_mm_shuffle_ps()�ĵ�3��������0x88������ syndrome[index2] == 1 ʱ��_mm_shuffle_ps()�ĵ�3��������0xdd��
												3	2	1	0
						                      -----------------
					prices[i << 2]		      | d | c | b | a |		
						                      -----------------
									          -----------------
					prices[(i + 1) << 2]      | h | g | f | e |
									          -----------------
									          -----------------
					_mm_shuffle_ps���        | g | e | c | a |
									          -----------------
			*/

			//����3
			//��Ϊ parity-check matrix ��������syndromelength��������vectorlength����submatrix��block����������matrixheight��
			//���Ե� syndromelength - index2 <= matrixheight ʱ��submatrix��Ҫ�ض����������С�
			//columns[][]�洢����submatrixÿһ�е�ֵ���ǰ�submatrixÿһ�еı��ش������ϱ�ʾ��10���ƣ���submatrix��ÿһ�е����һ�б�ʾ���λ����һ�б�ʾ���λ��
			//���統 syndromelength - index2 == matrixheight ʱ����Ҫ��ǰ��submatrix��ÿһ�е����һ�н�ȥ����colmask����һλ��colmask���λ�����0������λ����1��
			//��ʱcolumns[matrices[index2]][k] & colmask���൱�ڰ�submatrix��ÿһ�е����һ�н�ȥ�ˡ���Ҳ��Ϊʲôǰ���� column = columns[matrices[index2]][k] & colmask; ��ԭ��
			if(syndromelength - index2 <= matrixheight)
				colmask >>= 1;

			{
				//prices[]��ǰһ��λ�ô洢����ѡ��state֮�󣬺�һ��λ����Ҫ��ֵΪ�����
				register __m128 fillval = _mm_set1_ps(inf);
				for(l >>= 2; l < sseheight; l++) { //l >>= 2 ֮�� l ���� sseheight ��1/2
					_mm_store_ps(&prices[l << 2], fillval);
				}
			}
		} //end of blockѭ��

		totalprice = prices[0];	//����·���ܵĴ���

		aligned_free(prices);
		free(ssedone);

		if(totalprice >= total) {
			free(matrices);
			free(widths);
			free(columns[0]);
			free(columns[1]);
			if(stego != NULL)
				free(path);
			fprintf(stderr, "The syndrome is not in the range of the syndrome matrix.\n");
			return -1;
		}
	} else {
	/*
		SSE UINT8 VERSION
	*/
		int pathindex16 = 0, subprice = 0;
		u8 maxc = 0, minc = 0;
		u8 *prices, *pricevector = (u8*)pricevectorv;
		u16 *path16 = (u16 *)path;
		__m128i *prices16B;

		sseheight = height >> 4;
		ssedone = (u8*)malloc(sseheight * sizeof(u8));
		prices = (u8*)aligned_malloc(height * sizeof(u8), 16);
		prices16B = (__m128i*)prices;

		{
			__m128i napln = _mm_set1_epi32(0xffffffff);
			for(i = 0; i < sseheight; i++) {
				_mm_store_si128(&prices16B[i], napln);
				ssedone[i] = 0;
			}
		}

		prices[0] = 0;

		for(index = 0, index2 = 0; index2 < syndromelength; index2++) {
			register __m128i c1, c2, maxp, minp;

			if((u32)maxc + pricevector[index] >= 254) {
				aligned_free(path);
				free(ssedone);
				free(matrices);
				free(widths);
				free(columns[0]);
				free(columns[1]);
				if(stego != NULL)
					free(path);
				fprintf(stderr, "Price vector limit exceeded.");
				return -1;
			}

			for(k = 0; k < widths[index2]; k++, index++) {
				column = columns[matrices[index2]][k] & colmask;

				if(vector[index] == 0) {
					c1 = _mm_setzero_si128();
					c2 = _mm_set1_epi8(pricevector[index]);
				} else {
					c1 = _mm_set1_epi8(pricevector[index]);
					c2 = _mm_setzero_si128();
				}

				minp = _mm_set1_epi8(-1);
				maxp = _mm_setzero_si128();

				for(m = 0; m < sseheight; m++) {
					if(!ssedone[m]) {
						register __m128i v1, v2, v3, v4;
						altm = (m ^ (column >> 4));
						v1 = _mm_load_si128(&prices16B[m]);
						v2 = _mm_load_si128(&prices16B[altm]);
						v3 = v1;
						v4 = v2;
						ssedone[m] = 1;
						ssedone[altm] = 1;
						if(column & 8) {
							v2 = _mm_shuffle_epi32(v2, 0x4e);
							v3 = _mm_shuffle_epi32(v3, 0x4e);
						}
						if(column & 4) {
							v2 = _mm_shuffle_epi32(v2, 0xb1);
							v3 = _mm_shuffle_epi32(v3, 0xb1);
						}
						if(column & 2) {
							v2 = _mm_shufflehi_epi16(v2, 0xb1);
							v3 = _mm_shufflehi_epi16(v3, 0xb1);
							v2 = _mm_shufflelo_epi16(v2, 0xb1);
							v3 = _mm_shufflelo_epi16(v3, 0xb1);
						}
						if(column & 1) {
							v2 = _mm_or_si128(_mm_srli_epi16(v2, 8), _mm_slli_epi16(v2, 8));
							v3 = _mm_or_si128(_mm_srli_epi16(v3, 8), _mm_slli_epi16(v3, 8));
						}
						v1 = _mm_adds_epu8(v1, c1);
						v2 = _mm_adds_epu8(v2, c2);
						v3 = _mm_adds_epu8(v3, c2);
						v4 = _mm_adds_epu8(v4, c1);

						v1 = _mm_min_epu8(v1, v2);
						v4 = _mm_min_epu8(v3, v4);
											
						_mm_store_si128(&prices16B[m], v1);
						_mm_store_si128(&prices16B[altm], v4);

						minp = _mm_min_epu8(minp, _mm_min_epu8(v1, v4));
						maxp = _mm_max_epu8(maxp, maxLessThan255(v1, v4));
						
						if(stego != NULL) {
							v2 = _mm_cmpeq_epi8(v1, v2);
							v3 = _mm_cmpeq_epi8(v3, v4);
							path16[pathindex16 + m] = (u16)_mm_movemask_epi8(v2);
							path16[pathindex16 + altm] = (u16)_mm_movemask_epi8(v3);
						}
					}
				}

				maxc = max16B(maxp);
				minc = min16B(minp);

				maxc -= minc;
				subprice += minc;
				{
					register __m128i mask = _mm_set1_epi32(0xffffffff);
					register __m128i m = _mm_set1_epi8(minc);
					for(i = 0; i < sseheight; i++) {
						register __m128i res;
						register __m128i pr = prices16B[i];
						res = _mm_andnot_si128(_mm_cmpeq_epi8(pr, mask), m);
						prices16B[i] = _mm_sub_epi8(pr, res);
						ssedone[i] = 0;
					}
				}

				pathindex += parts;
				pathindex16 += parts << 1;
			}

			{
				register __m128i mask = _mm_set1_epi32(0x00ff00ff);

				if(minc == 255) {
					aligned_free(path);
					free(ssedone);
					free(matrices);
					free(widths);
					free(columns[0]);
					free(columns[1]);
					if(stego != NULL)
						free(path);
					fprintf(stderr, "The syndrome is not in the syndrome matrix range.\n");
					return -1;
				}

				if(syndrome[index2] == 0) {
					for(i = 0, l = 0; i < sseheight; i += 2, l++) {
						_mm_store_si128(&prices16B[l], _mm_packus_epi16(_mm_and_si128(_mm_load_si128(&prices16B[i]), mask), 
							_mm_and_si128(_mm_load_si128(&prices16B[i + 1]), mask)));
					}
				} else {
					for(i = 0, l = 0; i < sseheight; i += 2, l++) {
						_mm_store_si128(&prices16B[l], _mm_packus_epi16(_mm_and_si128(_mm_srli_si128(_mm_load_si128(&prices16B[i]), 1), mask), 
							_mm_and_si128(_mm_srli_si128(_mm_load_si128(&prices16B[i + 1]), 1), mask)));
					}
				}

				if(syndromelength - index2 <= matrixheight)
					colmask >>= 1;

				register __m128i fillval = _mm_set1_epi32(0xffffffff);
				for(; l < sseheight; l++)
					_mm_store_si128(&prices16B[l], fillval);
			}
		}
			
		totalprice = subprice + prices[0];
		
		aligned_free(prices);
		free(ssedone);
	}
	
	//����α�����е�backward part of theViterbi algorithm
	if(stego != NULL) {
		pathindex -= parts;
		index--;
		index2--;
		state = 0;

		int h = syndromelength;
		state = 0;
		colmask = 0;
		for(; index2 >= 0; index2--) { //blockѭ��
			for(k = widths[index2] - 1; k >= 0; k--, index--) {	//block�е���ѭ��
				if(k == widths[index2] - 1) {
					state = (state << 1) | syndrome[index2];
					if(syndromelength - index2 <= matrixheight)	//��ǰ��Ľ���3���Ӧ
						colmask = (colmask << 1) | 1; //��Viterbi�㷨��ǰ�򲿷����֮��colmask�Ѿ������ 000...0000 ,��ִ��Viterbi�㷨�ĺ��򲿷�ʱ����Ҫ������λ����һ�α��000...0001������֮ǰ������̡�
				}

				//����·����0��1д��stego�ı��ء���ǰ��Ľ���2���Ӧ
				//path[pathindex]ֻ��ÿһ�е�һ��state���׵�ַ����Ϊpath[]����32λΪ��λ�ģ���ÿ32��state����λΪ1������洢��һ��
				//����path[pathindex + (state >> 5)]��state����׵�ַ���ÿ�����32��·����0��1����
				//1 << (state & 31)��λ����32��·�����������·��
				if(path[pathindex + (state >> 5)] & (1 << (state & 31))) { //·����1
					stego[index] = 1;
					state = state ^ (columns[matrices[index2]][k] & colmask); //·����1ʱ��Ҫ���»��ǰһ�е�state
				} else { //·����0
					stego[index] = 0; //·����0ʱǰ�����е�state��ͬ
				}

				pathindex -= parts;	// path[] ��λ��ǰһ�е�һ��state�ĵ�ַ
			}
		}
		free(path);
	}

	free(matrices);
	free(widths);
	free(columns[0]);
	free(columns[1]);

	return totalprice;
}
