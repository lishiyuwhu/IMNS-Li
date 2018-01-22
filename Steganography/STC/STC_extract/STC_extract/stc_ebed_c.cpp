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
	double totalprice; //最优路径的总代价

	u8 *ssedone;
	u32 *path, *columns[2];
	int *matrices, *widths;

	if(matrixheight > 31) {
		fprintf(stderr, "Submatrix height must not exceed 31.");
		return -1;
	}

	height = 1 << matrixheight;
	colmask = height - 1;
	height = (height + 31) & (~31);//当height小于32时，把height强制变成32，即matrixheight不能小于5

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

		//因为cover长度vectorlength不一定是message长度syndromelength的整数倍，submatrix的宽度w可能是shorter或longer，
		//设matrices[i] = 0 的个数为x，matrices[i] = 1 的个数为y，最后要使得 worm = x * shorter + y * longer = vectorlength
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
		double total = 0; //所有cover比特的总代价
		float inf = std::numeric_limits<float>::infinity();	//返回编译器预定义的“正无穷大”，即1.#INF //Return Value:The representation of positive infinity for a type, if available.

		sseheight = height >> 2; //sseheight是每一列state的循环的次数。因为使用了SSE指令之后，每一次比较4个state，因此总的循环次数需要除以4
		ssedone = (u8*)malloc(sseheight * sizeof(u8)); //ssedone[m]的取值为1或为0，表示当前列中的第m个state的路径及代价是否已经计算过。其用处见下面的解释1。
		prices = (float*)aligned_malloc(height * sizeof(float), 16); //prices[]存储的是一列中所有state的代价

		{
			__m128 fillval = _mm_set1_ps(inf); //_mm_set1_ps( float w )：设置所有4个值为同一个值。r0 = r1 = r2 = r3 = w

			//把所有代价值初始化为无穷大
			for(i = 0; i < height; i+= 4) {	
				_mm_store_ps(&prices[i], fillval); //_mm_store_ps(float *p, __m128 a )：把寄存器 a 的四个单精度浮点值赋值给指针p所指向的地址。p[0] = a0, p[1] = a1, p[2] = a2, p[3] = a3
				ssedone[i >> 2] = 0;
			}
		}

		prices[0] = 0.0f; //在处理第1列时，该列的第1个代价值要为0

		//Viterbi算法的整数循环过程
		//block大循环
		for(index = 0, index2 = 0; index2 < syndromelength; index2++) {	//index是trellis每一列（对应cover的每一个元素）的索引值，index2是trellis每一block（对应要嵌入的每个比特）的索引值
			register __m128 c1, c2;

			//列循环
			for(k = 0; k < widths[index2]; k++, index++) { //widths[index2]是每一block中的列数
				column = columns[matrices[index2]][k] & colmask; //column是submatrix即block中的的第k列，为什么还要与colmask做位与运算，见下面的解释3。

				//c1是水平路径（stego的比特为0）的代价，c2是倾斜路径（stego的比特为1）的代价
				if(vector[index] == 0) { //如果cover的比特为0，则水平路径不需要修改cover的比特，故c1置0；而倾斜路径需要修改cover的比特，故要把预先得到的代价赋值给c2
					c1 = _mm_setzero_ps(); //_mm_setzero_ps(void)：清除四个单精度的浮点值。r0 := r1 := r2 := r3 := 0.0
					c2 = _mm_set1_ps((float)pricevector[index]);
				} else {
					c1 = _mm_set1_ps((float)pricevector[index]);
					c2 = _mm_setzero_ps();
				}

				total += pricevector[index]; //累加所有cover比特总的代价，不管有没有修改cover比特。

				//列中的state循环
				for(m = 0; m < sseheight; m++) {
					if(!ssedone[m]) {
						register __m128 v1, v2, v3, v4;
						altm = (m ^ (column >> 2));	//altm是进入当前state的路径为1时，前一列对应的state（不包括实际state的后两位）。因为一次比较4个state，且m是以4为单位的，所以column要除以4，在异或操作时排除后两位的影响。
						v1 = _mm_load_ps(&prices[m << 2]); //_mm_load_ps(float * p )：把指针p所指向位置的前四个单精度的浮点值赋值给寄存器并返回之。r0 = p[0], r1 = p[1], r2 = p[2], r3 = p[3]
						v2 = _mm_load_ps(&prices[altm << 2]); //v1和v2分别是进入当前state的路径为0或1时，前一列对应的两个state的代价。（v1和v2其实都分别包含了4个state的代价，因为一次比较4个state）。prices[]是以1为单位的，m和altm是以4为单位，所以m和altm要左移2位。
						v3 = v1; //为什么会把v1和v2备份一下，见下面的解释1。
						v4 = v2;
						ssedone[m] = 1;
						ssedone[altm] = 1;

						//前面在state与column相疑惑的时候，没有考虑后两个比特，下面在switch语句中加以考虑。即前面的state代价
						switch(column & 3) {
						case 0:
							break;
						case 1:
							v2 = _mm_shuffle_ps(v2, v2, 0xb1); //_mm_shuffle_ps(a, b, i)：根据掩码 i ，从 a 和 b 选择四个特定的单精度浮点值。详见 http://blog.csdn.net/zhuliting/article/details/6009672 和 http://forums.codeguru.com/printthread.php?t=337156
							v3 = _mm_shuffle_ps(v3, v3, 0xb1); ////为什么会对v3(v1)做与v2一样的操作，见下面的解释1。
							break;
						case 2:
							v2 = _mm_shuffle_ps(v2, v2, 0x4e);
							v3 = _mm_shuffle_ps(v3, v3, 0x4e);
							break;
						case 3:
							v2 = _mm_shuffle_ps(v2, v2, 0x1b);
							v3 = _mm_shuffle_ps(v3, v3, 0x1b);
							break;
						/*以 case 1 为例

						当前列的4个连续state：a、b、c、d映射到前一列的4个连续state：e、f、g、h，但是具体哪两个state相互关联还不知道。
						    当前列state     	column			 XOR(state的后两位，十进制)
						a     ......00			......01		  1			f
						b     ......01			......01		  0			e
						c     ......02			......01		  3			h
						d     ......03			......01		  2			g
							  
									   寄存器的四个存储单元
									      a   b   c   d
						                -----------------
							映射前		| 0 | 1 | 2 | 3 |		
						                -----------------
						     映射          \ /     \ /
						     过程          / \     / \
									    -----------------
							映射后      | 1 | 0 | 3 | 2 |
									    -----------------
										  f	  e	  h	  g

						v2 = _mm_shuffle_ps(v2, v2, 0xb1) 是把v2从右边数第2、3、0、1个元素分别移到新的v2从左边数起的四个存储单元中。0xb1表示2、3、0、1，这是从寄存器单元的右边开始数起的，如果从左边数起就是上图中的1、0、3、2.

						*/
						}

						//前一列中对应state的代价v加上路径本身的代价c，得到当前列当前state的代价v。
						//解释1
						//如果当前列的第m个state连接了前一列的第m个state（对应水平路径）和第altm个（对应倾斜路径）state，
						//那么当前列的第altm个state也会连接前一列的第m个state（对应倾斜路径）和第altm个（对应水平路径）state，
						//在这两种情况下，前一列中对应state的代价都是一样的，只是路径的代价会颠倒过来，所以才会有v1 = _mm_add_ps(v1, c1)和v3 = _mm_add_ps(v3, c2)，即v1和v3之前是相等的，而c1和c2是相反的。
						//因为有以上关系，所以当前列的第m个state和第altm个state的路径及其代价可以同时确定。
						//这也是为什么v1和v2要备份为v3和v4以及要有数组ssedone[]的原因，因为第m个state和第altm个state不一定是连续的，如果现在确定了第altm个state的路径和代价，等本层for循环运行到 m == altm 时，因为ssedone[altm]为1，就可以跳过了。
						v1 = _mm_add_ps(v1, c1); //_mm_add_ps(a, b)：把 a 和 b 的四个单精度浮点值分别对应相加并返回相加结果。r0 = a0 + b0, r1 = a1 + b1, r2 = a2 + b2, r3 = a3 + b3
						v2 = _mm_add_ps(v2, c2);
						v3 = _mm_add_ps(v3, c2);
						v4 = _mm_add_ps(v4, c1);

						//每一个state会与前一列的两个state所连接，使当前state有两条进入路径和两个代价值，从中选择代价最小的。
						v1 = _mm_min_ps(v1, v2); //_mm_min_ps(a , b )：取 a 和 b 的四个相对应单精度浮点值中的较小值并返回结果。ri = min(ai, bi)，i = 0,1,2,3 
						v4 = _mm_min_ps(v3, v4);
						
						//prices[]原本存储的是前一列中所有state的代价，因为当前列的第m个state和第altm个state的路径及其代价都同时确定了，
						//那么前一列的第m个state和第altm个state的代价就用不到了，所以覆盖之，使prices[]存储当前列中state的代价。（这与论文伪代码不一样，后者是等当前列的所有state的路径及代价都确定后，才一次性全部覆盖prices[]）
						_mm_store_ps(&prices[m << 2], v1);
						_mm_store_ps(&prices[altm << 2], v4);

						//解释2
						//计算出state的代价之后，在这里记录下连接state的路径（0或1）
						if(stego != NULL) {
							//v1是原v1和v2中的较小值，v2存储的是与state连接的路径为1的代价值，v2 = _mm_cmpeq_ps(v1, v2)得到新的v2，
							//则v2寄存器的四个单元中，存储的内容变成了较小代价对应的路径，路径0是0x0000，路径1是0xffff。
							v2 = _mm_cmpeq_ps(v1, v2); //_mm_cmpeq_ps (__m128 a , __m128 b )：返回一个_m128的寄存器，分别比较 a 和 b 对应位置32bit单精度浮点数是否相等，若相等，该位置返回0xffff，否则返回0x0。ri = (ai == bi) ? 0xffffffff : 0x0，i = 0,1,2,3
							v3 = _mm_cmpeq_ps(v3, v4);
							//path[]存储的是所有列的所有state对应的路径（0或1），以32位为单位；path8[]指向的是path[]，以8位为单位。
							//pathindex和pathindex8是path[]和path8[]每一列第一个state的地址与起始首地址（第一列第一个state）的偏移。
							//下面的语句是得到一个state（以4为单位）对应的路径，有4比特，存入path8[]中。
							//_mm_movemask_ps(v2)得到的4比特为4个路径，由于path8[]是以8比特位单位的，所以每两个state（以4为单位）的一共8个路径要合并为1个字节，而mask[m & 1]和shift[m & 1]就是用来合并一个字节的。
							//对2个state（以4为单位）的一共8个路径要合并为1个字节后，在这一个字节里，代表8个路径的比特是逆序排列的，即字节的最右边一位存放的是8个路径中的第一个（也即state最靠前的一个）。
							//u32类型内的每8个state（以4为单位）的一共32个路径是逆序排列的（这跟int转化为char的内存字节顺序有关），而u32类型之间的次序仍是顺序排列的，如下图所示。
							/*
								p[i]即path8[i]

							            |     32bit     |     32bit     |
										|    path[0]    |    path[1]    |
							            ----------------------------------
							     		| 3 | 2 | 1 | 0 | 3 | 2 | 1 | 0 |		
						                ----------------------------------
										p[3]p[2]p[1]p[0]p[7]p[6]p[5]p[4]
							*/
							path8[pathindex8 + (m >> 1)] = (path8[pathindex8 + (m >> 1)] & mask[m & 1]) | (_mm_movemask_ps(v2) << shift[m & 1]); //int _mm_movemask_ps( __m128 a )：r = sign(a3)<<3 | sign(a2)<<2 | sign(a1)<<1 | sign(a0)
							path8[pathindex8 + (altm >> 1)] = (path8[pathindex8 + (altm >> 1)] & mask[altm & 1]) | (_mm_movemask_ps(v3) << shift[altm & 1]);
						}
					} //end of if(!ssedone[m])
				} //end of state循环

				for(i = 0; i < sseheight; i++) { 
					ssedone[i] = 0;
				}

				pathindex += parts;	// parts 是 path[] 中每一列state的偏移距离
				pathindex8 += parts << 2; // parts << 2 是 path8[] 中每一列state的偏移距离
			} //end of 列循环

			//伪代码中的 prune state 过程，即根据要嵌入的比特（syndrome[index2]）选择当前block最后一列中的state连接到下一个block中。
			//如果state以1为单位，则要选择奇数state或偶数state；这里的state按照4为单位，一次考虑两个state（以4为单位），实际上是一次性从8个state（以1为单位）中选择4个state（以1为单位）
			if(syndrome[index2] == 0) {
				for(i = 0, l = 0; i < sseheight; i+= 2, l += 4) { //一次比较相邻的两个__m128寄存器，所以i += 2；一次得到新的4个代价，所以l += 4。
					_mm_store_ps(&prices[l], _mm_shuffle_ps(_mm_load_ps(&prices[i << 2]), _mm_load_ps(&prices[(i + 1) << 2]), 0x88)); // 0x88表示从两个寄存器的右边数起把第2个和第0个元素取出
				}
			} else {
				for(i = 0, l = 0; i < sseheight; i+= 2, l += 4) {
					_mm_store_ps(&prices[l], _mm_shuffle_ps(_mm_load_ps(&prices[i << 2]), _mm_load_ps(&prices[(i + 1) << 2]), 0xdd)); // 0x88表示从两个寄存器的右边数起把第3个和第1个元素取出
				}
			}
			/*以 syndrome[index2] == 0 为例

					由于寄存器单元的顺序的是从右到左的，所以当 syndrome[index2] == 0 时，_mm_shuffle_ps()的第3个参数是0x88；而当 syndrome[index2] == 1 时，_mm_shuffle_ps()的第3个参数是0xdd。
												3	2	1	0
						                      -----------------
					prices[i << 2]		      | d | c | b | a |		
						                      -----------------
									          -----------------
					prices[(i + 1) << 2]      | h | g | f | e |
									          -----------------
									          -----------------
					_mm_shuffle_ps结果        | g | e | c | a |
									          -----------------
			*/

			//解释3
			//因为 parity-check matrix 的行数是syndromelength，列数是vectorlength，而submatrix（block）的行数是matrixheight，
			//所以当 syndromelength - index2 <= matrixheight 时，submatrix需要截断下面若干行。
			//columns[][]存储的是submatrix每一列的值，是把submatrix每一列的比特从下往上表示成10进制，即submatrix中每一列的最后一行表示最高位，第一行表示最低位。
			//例如当 syndromelength - index2 == matrixheight 时，需要提前把submatrix中每一列的最后一行截去；而colmask右移一位后，colmask最高位变成了0，其它位都是1，
			//这时columns[matrices[index2]][k] & colmask就相当于把submatrix中每一列的最后一行截去了。这也是为什么前面有 column = columns[matrices[index2]][k] & colmask; 的原因。
			if(syndromelength - index2 <= matrixheight)
				colmask >>= 1;

			{
				//prices[]的前一半位置存储了挑选的state之后，后一半位置需要赋值为无穷大。
				register __m128 fillval = _mm_set1_ps(inf);
				for(l >>= 2; l < sseheight; l++) { //l >>= 2 之后 l 等于 sseheight 的1/2
					_mm_store_ps(&prices[l << 2], fillval);
				}
			}
		} //end of block循环

		totalprice = prices[0];	//最优路径总的代价

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
	
	//论文伪代码中的backward part of theViterbi algorithm
	if(stego != NULL) {
		pathindex -= parts;
		index--;
		index2--;
		state = 0;

		int h = syndromelength;
		state = 0;
		colmask = 0;
		for(; index2 >= 0; index2--) { //block循环
			for(k = widths[index2] - 1; k >= 0; k--, index--) {	//block中的列循环
				if(k == widths[index2] - 1) {
					state = (state << 1) | syndrome[index2];
					if(syndromelength - index2 <= matrixheight)	//与前面的解释3相对应
						colmask = (colmask << 1) | 1; //当Viterbi算法的前向部分完成之后，colmask已经变成了 000...0000 ,在执行Viterbi算法的后向部分时，需要向左移位（第一次变成000...0001），即之前的逆过程。
				}

				//根据路径是0或1写入stego的比特。与前面的解释2相对应
				//path[pathindex]只是每一列第一个state的首地址，因为path[]是以32位为单位的，即每32个state（单位为1）逆序存储在一起，
				//所以path[pathindex + (state >> 5)]是state块的首地址，该块内有32个路径（0或1）。
				//1 << (state & 31)定位到这32个路径里面的最优路径
				if(path[pathindex + (state >> 5)] & (1 << (state & 31))) { //路径是1
					stego[index] = 1;
					state = state ^ (columns[matrices[index2]][k] & colmask); //路径是1时需要重新获得前一列的state
				} else { //路径是0
					stego[index] = 0; //路径是0时前后两列的state相同
				}

				pathindex -= parts;	// path[] 定位到前一列第一个state的地址
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
