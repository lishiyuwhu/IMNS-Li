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
//������a�е����ݿ�������
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
	double k = 0;//ͳ��arr[i]==0����
	int temp = 0;
	double sum = 0;
	double b[63] = { 0 };
	for (i = 0; i < 63; i++){
		b[i] = 1;
	}
	//arr��0��62λ��ֵΪdct��1��63λ��
	//arr��63λ�Ѹ�ֵΪ0
	for (i = 0; i + 1 < 64; i++){
		arr[i] = dct[i + 1];
	}
	//��aar�����ǰ63λ��С�����������
	Qsort(arr, 0, 62);
	//k:ͳ��arr������0�ĸ���
	for (i = 0; i < 62; i++){
		if (arr[i] == 0)
			k++;
	}
	//����bͳ��arr������ǰ����ͬ�����ĸ���
	for (i = 0; i + 1 < 63; i++){
		if (arr[i + 1] - arr[i] == 0){
			b[temp]++;
		}
		else{ temp++; }
	}

	for (i = 0; i <= temp; i++){
		//b[i]/63�õ���С��������2Ϊ�׵Ķ������е����
		sum = sum - (b[i] / 63)*log(b[i] / 63) / log((double)2);
	}
	//��arr������ȥ������0�����㡣
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

//�����ļ���ָ��λ��
bool CopyFile(const char *src, const char *des)
{
	FILE * fSrc = fopen(src, "rb");
	if (!fSrc)
	{
		cout << "���ļ�" << src << "ʧ��";
		return false;
	}
	FILE * fDes = fopen(des, "wb");
	if (!fDes)
	{
		cout << "�����ļ�" << des << "ʧ��" << endl;
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


//��ȡ�ļ�·�������е��ļ���·����������files�У�
void getFiles(string path, vector<string>& files)
{
	//�ļ����  
	long   hFile = 0;
	//�ļ���Ϣ������һ���洢�ļ���Ϣ�Ľṹ��  
	struct _finddata_t fileinfo;
	string p;//�ַ��������·��
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)//�����ҳɹ��������
	{
		do
		{
			//�����Ŀ¼,����֮�����ļ����ڻ����ļ��У�  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0){
					printf("there is a directory in the dir!!!");
					return;
				}
			}
			//�������,�����б�  
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		//_findclose������������
		_findclose(hFile);
	}
}

int getmsglen(const char * file_path){
	int sum = 0, temp;
	//�����ļ�����ͼƬ������
	vector<string> files;
	//��ȡÿ��ͼƬ���ļ���
	getFiles((string)file_path, files);
	//����ÿ���ļ�������ÿ��ͼƬ�����������Ĵ�С��ͳ���ܵ��ļ�������С
	int size = files.size();
	for (int i = 0; i < size; i++){
		temp = getcoverlength(files[i].data());
		sum = sum + temp;
	}
	return sum;
}


//����һ���ļ��������е�ͼƬ�����ص���Ϣ������СΪs,���Ϊb
void analyzesize(const char * file_path, int &s, int &b){
	int smallsum = 0, bigsum = 0;
	int smalltemp, bigtemp;
	//�����ļ�����ͼƬ������
	vector<string> files;
	//��ȡÿ��ͼƬ���ļ���
	getFiles((string)file_path, files);
	//����ÿ���ļ�������ÿ��ͼƬ�����������Ĵ�С��ͳ���ܵ��ļ�������С

	int size = files.size();
	for (int i = 0; i < size; i++){
		getrange(files[i].data(), smalltemp, bigtemp);
		printf("��%d��ͼƬ������������Ϊ��(%d,%d]\n", i, smalltemp, bigtemp);
		smallsum = smallsum + smalltemp;
		bigsum = bigsum + bigtemp;
	}
	s = smallsum;
	b = bigsum;


}


//����ͼƬ·��file_path
//���ͼƬ·��stego_path
//�������Ϣmsg
//�������Ϣ����meglength
//�������Կseed

int stc_embed_ms(const char * file_path, const char * stego_jpeg, u8 *msg, int msglength, int seed){
	u8 *cover;
	u8 *stego;
	char str[1] = { 0x00 };

	short* stego_1;//�޸Ĺ����һά������
	double *profile;//ÿ�����ص�Ĵ���
	short *restego;//��stegoͼ��DCTϵ���洢��һλ����
	unsigned short quality_table[DCTSIZE2];
	int i, j, k, set_coverlength;
	int t = 0;
	int m = 0;//stego����ʹ��
	int num = 0;
	double dist;
	double limit = 0;//1
	short*** return_buf = NULL;
	unsigned int height_in_blocks, width_in_blocks;
	int flag = 0;//�Ƿ��һ�η���ռ�

	if (read_coeff(file_path, &return_buf, &height_in_blocks, &width_in_blocks, &flag, quality_table) == -1){
		return -1;
	}//��ȡԭʼͼ��DCTϵ��

	cover = new u8[height_in_blocks*width_in_blocks*(DCTSIZE2 - 1)];
	set_coverlength = height_in_blocks*width_in_blocks*(DCTSIZE2 - 1);
	stego = new u8[height_in_blocks*width_in_blocks*(DCTSIZE2 - 1)];
	profile = new double[height_in_blocks*width_in_blocks*(DCTSIZE2 - 1)];//2
	restego = new short[height_in_blocks*width_in_blocks*(DCTSIZE2 - 1)];//stegoͼ���DCTϵ���γ�һά����
	stego_1 = new short[height_in_blocks*width_in_blocks*(DCTSIZE2 - 1)];

	//��ά���飬�洢�ռ���Ϣ�ء�
	double** ent = new double*[height_in_blocks];//����ռ�洢��Ϣ��
	for (i = 0; i < height_in_blocks; i++){
		ent[i] = new double[width_in_blocks];
	}

	//return_buf��һ����ά���飬return_buf[i][j]��ʾ���8*8���е����ݣ�������63��Ԫ�ء�
	for (i = 0; i < height_in_blocks; i++){//�洢ÿ��block����Ϣ��
		for (j = 0; j < width_in_blocks; j++){
			ent[i][j] = entropy(return_buf[i][j])*entropy(return_buf[i][j]);
		}
	}

	//DCTSIZE2=64����ֵÿ����Ĵ���
	for (i = 0; i < height_in_blocks; i++){//����profile ֻ���������ÿ��BLOCK��ACϵ����ֵ
		for (j = 0; j < width_in_blocks; j++){
			for (k = 1; k < DCTSIZE2; k++){
				//ÿ��8*8�Ŀ���ֻȡ1��63,λ��0ΪDCTϵ��������ÿ��8*8�Ŀ���ֻȡ63�������ҽ���Ӧ�Ĵ���profile�и�ֵ��
				profile[(i*width_in_blocks + j)*(DCTSIZE2 - 1) + k - 1] = (1 / ent[i][j])*((double)(quality_table[k])*(double)(quality_table[k]));
			}
		}
	}

	for (i = 0; i < height_in_blocks; i++) {
		for (j = 0; j < width_in_blocks; j++){
			for (k = 1; k < DCTSIZE2; k++){
				//kΪ1��63��ACϵ��ֵ����1���������㣬
				//cover����洢ACϵ��ֵ��ͬʱ������Ϊutf-8
				//restego�洢�����ֵ
				cover[(i*width_in_blocks + j)*(DCTSIZE2 - 1) + k - 1] = u8(return_buf[i][j][k] & 1);
				restego[(i*width_in_blocks + j)*(DCTSIZE2 - 1) + k - 1] = return_buf[i][j][k];
			}
		}
	}

	t = set_coverlength;

	//limit��ʾ����ͼƬ�ķ���λ��ֵ
	for (i = 0; i < height_in_blocks; i++){//����profile ֻ���������ÿ��BLOCK��ACϵ����ֵ
		for (j = 0; j < width_in_blocks; j++){
			for (k = 1; k < DCTSIZE2; k++){
				if (return_buf[i][j][k] != 0)
					limit += profile[(i*width_in_blocks + j)*(DCTSIZE2 - 1) + k - 1];
			}
		}
	}

	//����Ϊ0��ACϵ��ֵ�Ĵ��۲�����Ƕ��,��ֵΪlimit��
	for (i = 0; i < height_in_blocks; i++) {
		for (j = 0; j < width_in_blocks; j++){
			for (k = 1; k < DCTSIZE2; k++){
				if (return_buf[i][j][k] == 0)
					profile[(i*width_in_blocks + j)*(DCTSIZE2 - 1) + k - 1] = limit;
			}
		}
	}

	//ȡcover�±꣬��ֵ��perm���ҡ�
	int* perm = new int[t];
	for (i = 0; i < t; i++){
		perm[i] = i;
	}
	//����
	rand_permutation(cover, profile, perm, t, seed);//�������õ�cover��profileһλ����������ұ�������˳��

	//��һ���жϣ��ж��Ƿ����ص���Ϣ�Ƿ��ʺ�Ƕ��

	//int biglimit = (int)ceil((double)(set_coverlength - 32) / 2) / 8;
	//int smalllimit = (int)floor((double)(set_coverlength - 32) / 256) / 8;
	//����getrange�����鿴ͼ��������Ϣ�ķ�Χ��smalllimit��ʾ��С��Ƕ����ֽ�����biglimit��ʾ����Ƕ���ֽ���
	int smalllimit, biglimit;
	if (getrange(file_path, smalllimit, biglimit) == -1){
		printf("the filepath is not valid");
		return -1;
	}

	printf("the size of message is (%d,%d] byte \n", smalllimit, biglimit);
	//�ж���msglength��biglimit��smalllimit֮��Ĺ�ϵ��
	//�ڷ�Χ�ڣ���ֱ��Ƕ�롣
	//�жϣ������š�
	if (msglength / 8 > smalllimit&&msglength / 8 <= biglimit){
		//�õ��˵�4������msglength��constr_height
		dist = stc_embed(cover + 32, set_coverlength - 32, msg + 32, msglength, (void*)(profile + 32), true, stego + 32);//STC�������Ƕ�뵽���ҵ�һά����
		printf("Distortion: %lf\n", dist);
		//���ص���Ϣ������stego��ǰ32λ�У�
		for (i = 0; i < 32; i++){
			stego[i] = msg[i];
		}
	}//���ڷ�Χ�ڣ��ֶ�Ƕ�롣
	else{
		printf("the Message length is not in range!");
	}

	restore(stego, perm, t);
	//�����������޸ĺ��ͼ��Ƕ���ԭ����λ��
	for (i = 0; i < height_in_blocks; i++) {
		for (j = 0; j < width_in_blocks; j++){
			for (k = 1; k < DCTSIZE2; k++){
				stego_1[(i*width_in_blocks + j)*(DCTSIZE2 - 1) + k - 1] = (return_buf[i][j][k] & 0xfffe) | ((short)(stego[(i*width_in_blocks + j)*(DCTSIZE2 - 1) + k - 1]));
			}
		}
	}
	//��jpeg���ϵ��д��ȥ��
	if (write_coeff(file_path, stego_jpeg, stego_1) == -1){//�����Ƕ���dctϵ���ָ���JPEGͼ����stego_jpg���ƴ洢�洢
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


//���ܵ��ļ����ֶΣ��ֱ�������ÿһ��ͼƬ��
//directorypath��ʾ�ļ���·��
//txt_path��ʾ��Ϣλ��
void hidemsg(const char * dir_path, const char * msg_path,const char * stego_dir, int seed){
	int smalltemp, bigtemp, allmsglength, msglength;
	//rest��¼�Ƿ���ҪǶ�룬sum��¼Ƕ��λ��
	int rest, sum = 0,record=0;
	//string stego_dir = "C:\\Users\\luSir\\Desktop\\new\\";
	string imageFormat = ".jpeg";
	u8 * allmsgtemp ;
	u8 * msgtemp;
	allmsgtemp = new u8[getmsglen(dir_path)];
	FILE* fp;
	char str[1] = { 0x00 };
	int num = 0;

	//��ȡ�е�������Ϣ��С
	if ((fp = fopen(msg_path, "rb")) == NULL){
		printf("�ļ����ɶ�");
		return ;
	}
	else{
		while (!feof(fp)){//feof�����Ե������ʵ��ȡ��numҪ��1����������ʹ�������ļ���ԭʼ�ļ���
			fread(str, sizeof(char), 1, fp);
			for (int i = 0; i < 8; i++){
				allmsgtemp[num * 8 + i] = u8((str[0] & (0x01 << (7 - i))) ? 1 : 0);
			}
			num++;
		}
		num--;
		//rest��ʾ��ҪǶ�����Ϣ
		rest = num * 8;
		
		//�����ļ�����ͼƬ������
		vector<string> files;
		//��ȡÿ��ͼƬ���ļ���
		getFiles((string)dir_path, files);
		int size = files.size();
		printf("allbitlength=%d\n",rest);
		for (int i = 0; i < size; i++){
			char szBuffer[20];
			sprintf_s(szBuffer, _countof(szBuffer), "%05d", i+1);
			string imageNum = szBuffer;
			string str = stego_dir + imageNum + imageFormat;
			const char * stego_jpeg = str.data();
			//��ÿ��ͼƬ������Ϣ��ÿ��ͼƬ��С��������ѡ���м�ֵ
			getrange(files[i].data(), smalltemp, bigtemp);
			msglength = ((smalltemp + bigtemp) / 2)*8;
			//��ҪǶ�����Ϣ�ȿ�Ƕ�������С�����������Ƕ��������ֱ��Ƕ��rest���ﵽ����Ҫ��
			if (rest <= msglength && rest>smalltemp){
				msgtemp = new u8[getcoverlength(files[i].data())];
				//�ȶ�ͼ�����Ϣд��msgtemp
				for (int i = 0; i < rest; i++){
					msgtemp[i + 32] = allmsgtemp[sum + i];
				}
				//����Ϣ�Ĵ�Сд��msgtemp��
				for (int i = 0; i < 32; i++){
					msgtemp[i] = u8((rest&(0x0001 << (31 - i))) ? 1 : 0);
				}
				

				stc_embed_ms(files[i].data(), stego_jpeg, msgtemp, rest, seed);
				//ִ���������һ�±���
				sum = sum + rest;
				rest = rest - rest;
				record = i;
				printf("the number of embed image is %d images\n",i+1);
			}
			else if (rest > msglength){//ʣ����ҪǶ�����Ϣ���ڿ�Ƕ�����Ϣ����Ƕ���Ƕ���msglength��Ϣ
				
				msgtemp = new u8[getcoverlength(files[i].data())];
				//�ȶ�ͼ�����Ϣд��msgtemp
				for (int i = 0; i < msglength; i++){
					msgtemp[i + 32] = allmsgtemp[sum + i];
				}
				//����Ϣ�Ĵ�Сд��msgtemp��
				for (int i = 0; i < 32; i++){
					msgtemp[i] = u8((msglength&(0x0001 << (31 - i))) ? 1 : 0);
				}
				
				stc_embed_ms(files[i].data(), stego_jpeg, msgtemp, msglength,seed);
				//ִ���������һ�±���
				rest = rest - msglength;
				sum = sum + msglength;
			}
			else{
				if (rest==0){
					CopyFile(files[i].data(), stego_jpeg);

				}
				else{
					//Ƕ����ȥ���ȸ�����ʾ�¡�
					printf("���һ��Ƕ�����Ϣ%d����ͼƬ��Ƕ��ķ�Χ��\n", rest);
				}
				
				
			}

			printf("%d:(msglength=%d,rest=%d,sum=%d)\n",i+1,msglength,rest,sum);

		}
		if (rest == 0)
			printf("������ϢǶ��ɹ�\n");
		num = 0;
	}
}






int main(int argc , char ** argv)
{

	//���Է��������Ƿ�OK
	
	char * dir_path = argv[1];
	char * msg_path = argv[2];
	char * stego_dir= argv[3];
	int seed = (unsigned int)atoi(argv[4]);
	//printf("hh");
	hidemsg(dir_path, msg_path, stego_dir,seed);
	
	//STC_extract.exe C:\Users\luSir\Desktop\azw6_images C:\Users\luSir\Desktop\hh.pptx C:\Users\luSir\Desktop\stego_images 100



	/*
	//char * ttp = "C:\\sUsers\\luSir\\Desktop\hide\\ɽ����_B00AA7KEOU.azw6";
	char* dir_path = "C:\\Users\\luSir\\Desktop\\lybjpeg";
	int x, y;
	analyzesize(dir_path, x, y);
	cout << x << endl;
	cout << y << endl;
	*/

	

	/*
	char* dir_path = "C:\\Users\\luSir\\Desktop\\lybjpeg";
	vector<string> files;
	//��ȡÿ��ͼƬ���ļ���
	getFiles((string)dir_path, files);
	//����ÿ���ļ�������ÿ��ͼƬ�����������Ĵ�С��ͳ���ܵ��ļ�������С

	
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
			while (!feof(fp)){//feof�����Ե������ʵ��ȡ��numҪ��1����������ʹ�������ļ���ԭʼ�ļ���
				fread(str, sizeof(char), 1, fp);
				for (int i = 0; i < 8; i++){
					msg[32 + num * 8 + i] = u8((str[0] & (0x01 << (7 - i))) ? 1 : 0);
				}
				num++;
			}
			num--;
			msglength = num * 8;
			printf("%d\n", msglength);

			//msglength�Ѿ��������ˣ�����������ͼƬ��С����������Ϣ��
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







		//����stc_embed_ms,ok;
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
		while (!feof(fp)){//feof�����Ե������ʵ��ȡ��numҪ��1����������ʹ�������ļ���ԭʼ�ļ���
		fread(str, sizeof(char), 1, fp);
		for (int i = 0; i < 8; i++){
		msg[32 + num * 8 + i] = u8((str[0] & (0x01 << (7 - i))) ? 1 : 0);
		}
		num++;
		}
		num--;
		msglength = num * 8;
		printf("%d",msglength);

		//msglength�Ѿ��������ˣ�����������ͼƬ��С����������Ϣ��
		for (int i = 0; i < 32; i++){
		msg[i] = u8((msglength&(0x0001 << (31 - i))) ? 1 : 0);
		}
		stc_embed_ms(photo_path, stego_path, msg, msglength, 100);
		}
		*/




		return 0;



}
