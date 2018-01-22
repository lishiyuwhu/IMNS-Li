#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <io.h>  
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include<windows.h>

#include "stc_embed_c.h"
#include "stc_extract_c.h"
#include "jpeglib.h"
#include "joint.h"
#include "math.h"
#include "float.h"
#include "time.h"
#include <limits>

using namespace std;

//const int coverlength = 1000000;
//对数组a中的内容快速排序
void Qsort(short a[], int low, int high)
{
	if (low >= high) return;
	int first = low;
	int last = high;
	int key = a[first];
	while (first < last)
	{
		while (first < last&&a[last] >= key) --last;
		a[first] = a[last];
		while (first < last&&a[first] <= key) ++first;
		a[last] = a[first];
	}
	a[first] = key;
	Qsort(a, low, first - 1);
	Qsort(a, first + 1, high);
}

double entropy(short dct[])
{
	short arr[63] = { 0 };
	int i;
	double k = 0;//统计arr[i]==0个数
	int temp = 0;
	double sum = 0;
	double b[63] = { 0 };
	for (i = 0; i < 63; i++){
		b[i] = 1;
	}
	//arr的0到62位赋值为dct的1到63位，
	//arr的63位已赋值为0
	for (i = 0; i + 1 < 64; i++){
		arr[i] = dct[i + 1];
	}
	//对aar数组的前63位从小到大快速排序
	Qsort(arr, 0, 62);
	//k:统计arr数组中0的个数
	for (i = 0; i < 62; i++){
		if (arr[i] == 0)
			k++;
	}
	//数组b统计arr数组中前后相同的数的个数
	for (i = 0; i + 1 < 63; i++){
		if (arr[i + 1] - arr[i] == 0){
			b[temp]++;
		}
		else{ temp++; }
	}

	for (i = 0; i <= temp; i++){
		//b[i]/63得到的小数乘以以2为底的对数所有的相加
		sum = sum - (b[i] / 63)*log(b[i] / 63) / log((double)2);
	}
	//将arr数组中去掉对于0的运算。
	if (k != 0){
		sum = sum + (k / 63)*log(k / 63) / log((double)2);
	}
	return sum;
}

static void rand_permutation(u8 cover[], double profile[], int perm[], int t, unsigned int seed)
{
	int i, j;
	int n = t;
	u8 temp1;
	double temp2;
	int temp3;
	srand(seed);
	for (i = 0; i < n; i++){
		j = rand() % (n - i);
		temp1 = cover[i];
		temp2 = profile[i];
		temp3 = perm[i];
		cover[i] = cover[i + j];
		profile[i] = profile[i + j];
		perm[i] = perm[i + j];
		cover[i + j] = temp1;
		profile[i + j] = temp2;
		perm[i + j] = temp3;
	}
}

void restore(u8 stego[], int perm[], int t)
{
	int i;
	u8 *arr = new u8[t];
	for (i = 0; i < t; i++){
		arr[i] = stego[i];
	}
	for (i = 0; i < t; i++){
		stego[perm[i]] = arr[i];
	}
	delete[] arr;
}

//复制文件到指定位置
bool CopyFile(const char *src, const char *des)
{
	FILE * fSrc = fopen(src, "rb");
	if (!fSrc)
	{
		cout << "打开文件" << src << "失败";
		return false;
	}
	FILE * fDes = fopen(des, "wb");
	if (!fDes)
	{
		cout << "创建文件" << des << "失败" << endl;
		return false;
	}
	unsigned char * buf;
	unsigned int length;
	fseek(fSrc, 0, SEEK_END);
	length = ftell(fSrc);
	buf = new unsigned char[length + 1];
	memset(buf, 0, length + 1);
	fseek(fSrc, 0, SEEK_SET);
	fread(buf, length, 1, fSrc);    fwrite(buf, length, 1, fDes);    fclose(fSrc);
	fclose(fDes);
	delete[] buf;
	return true;
}


//获取文件路径下所有的文件名路径，保存在files中，
void getFiles(string path, vector<string>& files)
{
	//文件句柄  
	long   hFile = 0;
	//文件信息，声明一个存储文件信息的结构体  
	struct _finddata_t fileinfo;
	string p;//字符串，存放路径
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)//若查找成功，则进入
	{
		do
		{
			//如果是目录,迭代之（即文件夹内还有文件夹）  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0){
					printf("there is a directory in the dir!!!");
					return;
				}
			}
			//如果不是,加入列表  
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		//_findclose函数结束查找
		_findclose(hFile);
	}
}

int getmsglen(const char * file_path){
	int sum = 0, temp;
	//分析文件夹下图片的数量
	vector<string> files;
	//获取每个图片的文件名
	getFiles((string)file_path, files);
	//遍历每个文件，分析每个图片可隐藏容量的大小，统计总的文件容量大小
	int size = files.size();
	for (int i = 0; i < size; i++){
		temp = getcoverlength(files[i].data());
		sum = sum + temp;
	}
	return sum;
}


//分析一个文件夹下所有的图片可隐藏的信息量，最小为s,最大为b
void analyzesize(const char * file_path, int &s, int &b){
	int smallsum = 0, bigsum = 0;
	int smalltemp, bigtemp;
	//分析文件夹下图片的数量
	vector<string> files;
	//获取每个图片的文件名
	getFiles((string)file_path, files);
	//遍历每个文件，分析每个图片可隐藏容量的大小，统计总的文件容量大小

	int size = files.size();
	for (int i = 0; i < size; i++){
		getrange(files[i].data(), smalltemp, bigtemp);
		printf("第%d个图片隐藏容量区间为：(%d,%d]\n", i, smalltemp, bigtemp);
		smallsum = smallsum + smalltemp;
		bigsum = bigsum + bigtemp;
	}
	s = smallsum;
	b = bigsum;


}


//输入图片路径file_path
//输出图片路径stego_path
//输入的信息msg
//输入的信息长度meglength
//输入的秘钥seed

int stc_embed_ms(const char * file_path, const char * stego_jpeg, u8 *msg, int msglength, int seed){
	u8 *cover;
	u8 *stego;
	char str[1] = { 0x00 };

	short* stego_1;//修改过后的一维的数组
	double *profile;//每个像素点的代价
	short *restego;//将stego图像DCT系数存储成一位数组
	unsigned short quality_table[DCTSIZE2];
	int i, j, k, set_coverlength;
	int t = 0;
	int m = 0;//stego计数使用
	int num = 0;
	double dist;
	double limit = 0;//1
	short*** return_buf = NULL;
	unsigned int height_in_blocks, width_in_blocks;
	int flag = 0;//是否第一次分配空间

	if (read_coeff(file_path, &return_buf, &height_in_blocks, &width_in_blocks, &flag, quality_table) == -1){
		return -1;
	}//读取原始图像DCT系数

	cover = new u8[height_in_blocks*width_in_blocks*(DCTSIZE2 - 1)];
	set_coverlength = height_in_blocks*width_in_blocks*(DCTSIZE2 - 1);
	stego = new u8[height_in_blocks*width_in_blocks*(DCTSIZE2 - 1)];
	profile = new double[height_in_blocks*width_in_blocks*(DCTSIZE2 - 1)];//2
	restego = new short[height_in_blocks*width_in_blocks*(DCTSIZE2 - 1)];//stego图像的DCT系数形成一维数组
	stego_1 = new short[height_in_blocks*width_in_blocks*(DCTSIZE2 - 1)];

	//二维数组，存储空间信息熵。
	double** ent = new double*[height_in_blocks];//申请空间存储信息熵
	for (i = 0; i < height_in_blocks; i++){
		ent[i] = new double[width_in_blocks];
	}

	//return_buf是一个三维数组，return_buf[i][j]表示这个8*8块中的内容，包含有63个元素。
	for (i = 0; i < height_in_blocks; i++){//存储每个block的信息熵
		for (j = 0; j < width_in_blocks; j++){
			ent[i][j] = entropy(return_buf[i][j])*entropy(return_buf[i][j]);
		}
	}

	//DCTSIZE2=64，赋值每个块的代价
	for (i = 0; i < height_in_blocks; i++){//计算profile 只计算相对于每个BLOCK的AC系数的值
		for (j = 0; j < width_in_blocks; j++){
			for (k = 1; k < DCTSIZE2; k++){
				//每个8*8的块中只取1到63,位置0为DCT系数，所以每个8*8的块中只取63个，并且将对应的代价profile中赋值。
				profile[(i*width_in_blocks + j)*(DCTSIZE2 - 1) + k - 1] = (1 / ent[i][j])*((double)(quality_table[k])*(double)(quality_table[k]));
			}
		}
	}

	for (i = 0; i < height_in_blocks; i++) {
		for (j = 0; j < width_in_blocks; j++){
			for (k = 1; k < DCTSIZE2; k++){
				//k为1到63的AC系数值，和1进行与运算，
				//cover载体存储AC系数值的同时，编码为utf-8
				//restego存储具体的值
				cover[(i*width_in_blocks + j)*(DCTSIZE2 - 1) + k - 1] = u8(return_buf[i][j][k] & 1);
				restego[(i*width_in_blocks + j)*(DCTSIZE2 - 1) + k - 1] = return_buf[i][j][k];
			}
		}
	}

	t = set_coverlength;

	//limit表示整个图片的非零位的值
	for (i = 0; i < height_in_blocks; i++){//计算profile 只计算相对于每个BLOCK的AC系数的值
		for (j = 0; j < width_in_blocks; j++){
			for (k = 1; k < DCTSIZE2; k++){
				if (return_buf[i][j][k] != 0)
					limit += profile[(i*width_in_blocks + j)*(DCTSIZE2 - 1) + k - 1];
			}
		}
	}

	//对于为0的AC系数值的代价不进行嵌入,赋值为limit。
	for (i = 0; i < height_in_blocks; i++) {
		for (j = 0; j < width_in_blocks; j++){
			for (k = 1; k < DCTSIZE2; k++){
				if (return_buf[i][j][k] == 0)
					profile[(i*width_in_blocks + j)*(DCTSIZE2 - 1) + k - 1] = limit;
			}
		}
	}

	//取cover下标，赋值给perm置乱。
	int* perm = new int[t];
	for (i = 0; i < t; i++){
		perm[i] = i;
	}
	//置乱
	rand_permutation(cover, profile, perm, t, seed);//对于所得的cover和profile一位数组进行置乱保存置乱顺序

	//加一个判断，判断是否隐藏的信息是否适合嵌入

	//int biglimit = (int)ceil((double)(set_coverlength - 32) / 2) / 8;
	//int smalllimit = (int)floor((double)(set_coverlength - 32) / 256) / 8;
	//调用getrange函数查看图像隐藏信息的范围，smalllimit表示最小的嵌入的字节量，biglimit表示最大的嵌入字节量
	int smalllimit, biglimit;
	if (getrange(file_path, smalllimit, biglimit) == -1){
		printf("the filepath is not valid");
		return -1;
	}

	printf("the size of message is (%d,%d] byte \n", smalllimit, biglimit);
	//判断下msglength和biglimit、smalllimit之间的关系。
	//在范围内，则直接嵌入。
	//判断，先留着。
	if (msglength / 8 > smalllimit&&msglength / 8 <= biglimit){
		//用到了第4个参数msglength和constr_height
		dist = stc_embed(cover + 32, set_coverlength - 32, msg + 32, msglength, (void*)(profile + 32), true, stego + 32);//STC编码进行嵌入到置乱的一维数组
		printf("Distortion: %lf\n", dist);
		//隐藏的信息量存在stego的前32位中，
		for (i = 0; i < 32; i++){
			stego[i] = msg[i];
		}
	}//不在范围内，分段嵌入。
	else{
		printf("the Message length is not in range!");
	}

	restore(stego, perm, t);
	//后续处理，将修改后的图像嵌入回原来的位置
	for (i = 0; i < height_in_blocks; i++) {
		for (j = 0; j < width_in_blocks; j++){
			for (k = 1; k < DCTSIZE2; k++){
				stego_1[(i*width_in_blocks + j)*(DCTSIZE2 - 1) + k - 1] = (return_buf[i][j][k] & 0xfffe) | ((short)(stego[(i*width_in_blocks + j)*(DCTSIZE2 - 1) + k - 1]));
			}
		}
	}
	//将jpeg相关系数写回去。
	if (write_coeff(file_path, stego_jpeg, stego_1) == -1){//将完成嵌入的dct系数恢复成JPEG图像以stego_jpg名称存储存储
		return -1;
	}

	for (i = 0; i < height_in_blocks; i++){
		for (j = 0; j < width_in_blocks; j++){
			free(return_buf[i][j]);
		}
		free(return_buf[i]);
	}
	free(return_buf);

	for (i = 0; i < height_in_blocks; i++){
		delete[] ent[i];
	}
	delete[] ent;
	delete[] profile;
	delete[] stego;
	delete[] cover;
	delete[] perm;
	delete[] stego_1;
	delete[] restego;
	return 0;

}


//将总的文件流分段，分别隐藏在每一个图片中
//directorypath表示文件夹路径
//txt_path表示信息位置
void hidemsg(const char * dir_path, const char * msg_path,const char * stego_dir, int seed){
	int smalltemp, bigtemp, allmsglength, msglength;
	//rest记录是否需要嵌入，sum记录嵌入位置
	int rest, sum = 0,record=0;
	//string stego_dir = "C:\\Users\\luSir\\Desktop\\new\\";
	string imageFormat = ".jpeg";
	u8 * allmsgtemp ;
	u8 * msgtemp;
	allmsgtemp = new u8[getmsglen(dir_path)];
	FILE* fp;
	char str[1] = { 0x00 };
	int num = 0;

	//获取中的隐藏信息大小
	if ((fp = fopen(msg_path, "rb")) == NULL){
		printf("文件不可读");
		return ;
	}
	else{
		while (!feof(fp)){//feof的特性导致最后实际取得num要大1，产生了误差，使产生的文件比原始文件长
			fread(str, sizeof(char), 1, fp);
			for (int i = 0; i < 8; i++){
				allmsgtemp[num * 8 + i] = u8((str[0] & (0x01 << (7 - i))) ? 1 : 0);
			}
			num++;
		}
		num--;
		//rest表示需要嵌入的信息
		rest = num * 8;
		
		//分析文件夹下图片的数量
		vector<string> files;
		//获取每个图片的文件名
		getFiles((string)dir_path, files);
		int size = files.size();
		printf("allbitlength=%d\n",rest);
		for (int i = 0; i < size; i++){
			char szBuffer[20];
			sprintf_s(szBuffer, _countof(szBuffer), "%05d", i+1);
			string imageNum = szBuffer;
			string str = stego_dir + imageNum + imageFormat;
			const char * stego_jpeg = str.data();
			//给每个图片隐藏信息，每个图片大小隐藏容量选择中间值
			getrange(files[i].data(), smalltemp, bigtemp);
			msglength = ((smalltemp + bigtemp) / 2)*8;
			//需要嵌入的信息比可嵌入的容量小，但大于最低嵌入量，则直接嵌入rest，达到结束要求
			if (rest <= msglength && rest>smalltemp){
				msgtemp = new u8[getcoverlength(files[i].data())];
				//先对图像的信息写入msgtemp
				for (int i = 0; i < rest; i++){
					msgtemp[i + 32] = allmsgtemp[sum + i];
				}
				//将信息的大小写入msgtemp中
				for (int i = 0; i < 32; i++){
					msgtemp[i] = u8((rest&(0x0001 << (31 - i))) ? 1 : 0);
				}
				

				stc_embed_ms(files[i].data(), stego_jpeg, msgtemp, rest, seed);
				//执行完后，设置一下变量
				sum = sum + rest;
				rest = rest - rest;
				record = i;
				printf("the number of embed image is %d images\n",i+1);
			}
			else if (rest > msglength){//剩余需要嵌入的信息大于可嵌入的信息，则嵌入可嵌入的msglength信息
				
				msgtemp = new u8[getcoverlength(files[i].data())];
				//先对图像的信息写入msgtemp
				for (int i = 0; i < msglength; i++){
					msgtemp[i + 32] = allmsgtemp[sum + i];
				}
				//将信息的大小写入msgtemp中
				for (int i = 0; i < 32; i++){
					msgtemp[i] = u8((msglength&(0x0001 << (31 - i))) ? 1 : 0);
				}
				
				stc_embed_ms(files[i].data(), stego_jpeg, msgtemp, msglength,seed);
				//执行完后，设置一下变量
				rest = rest - msglength;
				sum = sum + msglength;
			}
			else{
				if (rest==0){
					CopyFile(files[i].data(), stego_jpeg);

				}
				else{
					//嵌不进去，先给个提示下。
					printf("最后一次嵌入的信息%d不在图片可嵌入的范围中\n", rest);
				}
				
				
			}

			printf("%d:(msglength=%d,rest=%d,sum=%d)\n",i+1,msglength,rest,sum);

		}
		if (rest == 0)
			printf("所有信息嵌入成功\n");
		num = 0;
	}
}






int main(int argc , char ** argv)
{

	//测试分析函数是否OK
	
	char * dir_path = argv[1];
	char * msg_path = argv[2];
	char * stego_dir= argv[3];
	int seed = (unsigned int)atoi(argv[4]);
	//printf("hh");
	hidemsg(dir_path, msg_path, stego_dir,seed);
	
	//STC_extract.exe C:\Users\luSir\Desktop\azw6_images C:\Users\luSir\Desktop\hh.pptx C:\Users\luSir\Desktop\stego_images 100



	/*
	//char * ttp = "C:\\sUsers\\luSir\\Desktop\hide\\山海经_B00AA7KEOU.azw6";
	char* dir_path = "C:\\Users\\luSir\\Desktop\\lybjpeg";
	int x, y;
	analyzesize(dir_path, x, y);
	cout << x << endl;
	cout << y << endl;
	*/

	

	/*
	char* dir_path = "C:\\Users\\luSir\\Desktop\\lybjpeg";
	vector<string> files;
	//获取每个图片的文件名
	getFiles((string)dir_path, files);
	//遍历每个文件，分析每个图片可隐藏容量的大小，统计总的文件容量大小

	
	u8 * msg;
	int msglength;
	char str[1] = { 0x00 };
	FILE* fp;
	int num = 0;
	char * file_path = "C:\\Users\\luSir\\Desktop\\123456.txt";
	string stego_dir = "C:\\Users\\luSir\\Desktop\\hhccqr\\newHDPhoto";
	string imageFormat = ".jpeg";
	int size = files.size();
	for (int i = 0; i < size-1; i++){
		
		int set_coverlength = getcoverlength(files[i].data());
		msg = new u8[set_coverlength];
		string imageNum = to_string(i);
		string strr = stego_dir + imageNum + imageFormat;
		const char * stego_jpeg = strr.data();
		if ((fp = fopen(file_path, "rb")) == NULL){
			return false;
		}
		else{
			while (!feof(fp)){//feof的特性导致最后实际取得num要大1，产生了误差，使产生的文件比原始文件长
				fread(str, sizeof(char), 1, fp);
				for (int i = 0; i < 8; i++){
					msg[32 + num * 8 + i] = u8((str[0] & (0x01 << (7 - i))) ? 1 : 0);
				}
				num++;
			}
			num--;
			msglength = num * 8;
			printf("%d\n", msglength);

			//msglength已经读出来了，接下来根据图片大小分配隐藏信息。
			for (int i = 0; i < 32; i++){
				msg[i] = u8((msglength&(0x0001 << (31 - i))) ? 1 : 0);
			}
			int dist=stc_embed_ms(files[i].data(), stego_jpeg, msg, msglength, 100);
			msglength = 0;
			num = 0;
			//printf("%d\n",dist);
		}
	}
	*/







		//测试stc_embed_ms,ok;
	/*
		u8 * msg;
		int msglength;
		char str[1] = { 0x00 };
		FILE* fp;
		int num = 0;
		char * file_path = "C:\\Users\\luSir\\Desktop\\1234.txt";
		char * photo_path = "C:\\Users\\luSir\\Desktop\\lybjpeg\\HDimage00049.jpg";
		char * stego_path = "C:\\Users\\luSir\\Desktop\\stego000049.jpg";
		int set_coverlength = getcoverlength(photo_path);
		msg = new u8[set_coverlength];
		//msg = new u8[20320];
		if ((fp = fopen(file_path, "rb")) == NULL){
		return false;
		}else{
		while (!feof(fp)){//feof的特性导致最后实际取得num要大1，产生了误差，使产生的文件比原始文件长
		fread(str, sizeof(char), 1, fp);
		for (int i = 0; i < 8; i++){
		msg[32 + num * 8 + i] = u8((str[0] & (0x01 << (7 - i))) ? 1 : 0);
		}
		num++;
		}
		num--;
		msglength = num * 8;
		printf("%d",msglength);

		//msglength已经读出来了，接下来根据图片大小分配隐藏信息。
		for (int i = 0; i < 32; i++){
		msg[i] = u8((msglength&(0x0001 << (31 - i))) ? 1 : 0);
		}
		stc_embed_ms(photo_path, stego_path, msg, msglength, 100);
		}
		*/




		return 0;



}
