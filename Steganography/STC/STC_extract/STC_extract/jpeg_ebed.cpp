#include <cstdio>
#include <cstdlib>

#include <io.h>  
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "stc_embed_c.h"
#include "stc_extract_c.h"
#include "jpeglib.h"
#include "joint.h"
#include "math.h"
#include "float.h"
#include "time.h"
#include <limits>

using namespace std;

static void rand_permutation(u8 stego[], int t, unsigned int seed)
{
	int i, j;
	int n = t;
	u8 temp1;
	srand(seed);
	for (i = 0; i < n; i++){
		j = rand() % (n - i);
		temp1 = stego[i];
		stego[i] = stego[i + j];
		stego[i + j] = temp1;
	}
}

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




void extractms(const char* file_path, u8 * msg, int &msglength,unsigned int seed){
	u8 *stego;
	int i, j, k, set_coverlength;
	int t = 0;
	int m = 0;//stego����ʹ��
	int flag = 0;
	short*** return_buf = NULL;
	short*** stego_buf = NULL;//�洢stegoͼ���DCTϵ��
	unsigned int height_in_blocks, width_in_blocks;
	int num = 0;//�洢���ܳ���
	//unsigned int seed = (unsigned int)atoi(argv[3]);//��ȡ�������
	if (read_coeff(file_path, &return_buf, &height_in_blocks, &width_in_blocks, &flag) == -1){
		return;
	}//��ȡԭʼͼ��DCTϵ��

	set_coverlength = height_in_blocks*width_in_blocks*(DCTSIZE2 - 1);
	//msgtemp = new u8[height_in_blocks*width_in_blocks*(DCTSIZE2 - 1) + 32];//�洢��õ������ļ�
	stego = new u8[height_in_blocks*width_in_blocks*(DCTSIZE2 - 1)];


	for (i = 0; i < height_in_blocks; i++) {
		for (j = 0; j < width_in_blocks; j++){
			for (k = 1; k < DCTSIZE2; k++){
				stego[(i*width_in_blocks + j)*(DCTSIZE2 - 1) + k - 1] = u8(return_buf[i][j][k] & 1);
			}
		}
	}
	t = set_coverlength;

	rand_permutation(stego, t, seed);//�������õ�stego����Ƕ��ʱ��ͬ�����ӽ�������

	for (i = 0; i < 32; i++){
		num += int(stego[i] << (31 - i));
	}

	msglength = num;
	stc_extract(stego + 32, set_coverlength - 32, msg, msglength);
	
	

	//�ͷ���Դ

	for (i = 0; i < height_in_blocks; i++){
		for (j = 0; j < width_in_blocks; j++){
			free(return_buf[i][j]);
		}
		free(return_buf[i]);
	}
	free(return_buf);

	delete[] stego;


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
		//printf("��%d��ͼƬ������������Ϊ��(%d,%d]\n", i, smalltemp, bigtemp);
		smallsum = smallsum + smalltemp;
		bigsum = bigsum + bigtemp;
	}
	s = smallsum;
	b = bigsum;


}

void extractallms(const char* dir_path, const char* out_path, int num,int seed){
	int i,j,small=0,big=0;
	int sum=0;
	u8 * msg;
	u8 * allmsg;
	int msglength = 0;
	analyzesize(dir_path, small, big);
	allmsg = new u8[big*8];//��ʱ������д

	vector<string> files;
	//��ȡÿ��ͼƬ���ļ���.����ط��пӣ���ȡͼƬ����0��1˳���ȡ��
	getFiles((string)dir_path, files);
	int size = files.size()>num ? num : files.size();
	
	for (int i = 0; i < size; i++){
		msg = new u8[getcoverlength(files[i].data())*8];
		extractms(files[i].data(), msg, msglength, seed);
		for (int j = 0; j < msglength; j++){
			allmsg[j + sum] = msg[j];
		}
		sum += msglength;
	}
	printf("%d", sum);
	
	//��ʼд�ļ�
	FILE*fp;
	if ((fp = fopen(out_path, "wb")) == NULL){
		printf("can't open%s\n ", out_path);
		exit(EXIT_FAILURE);
	}
	char *str_1 = new char[sum];
	for (i = 0; i < sum; i++){
		str_1[i] = 0;
	}

	for (i = 0; i < sum / 8; i++){
		for (j = 0; j < 8; j++){
			str_1[i] |= char(allmsg[i * 8 + j] << (7 - j));
		}
	}

	fwrite(str_1, sizeof(char), sum / 8, fp);
	fclose(fp);
	delete[] allmsg;
	delete[] msg;
	delete[] str_1;
	
}




int main(int argc,char ** argv)
{
	/*
	u8 * msg;
	int msglength,i,j;
	const char* file_path = "C:\\Users\\luSir\\Desktop\\new\\HDPhoto32.jpeg";
	msg = new u8[getcoverlength(file_path)];

	
	unsigned int seed = 100;
	extractms(file_path, msg, msglength, seed);
	char *str_1 = new char[msglength];
	for (i = 0; i < msglength; i++){
		str_1[i] = 0;
	}

	for (i = 0; i < msglength / 8; i++){
		for (j = 0; j < 8; j++){
			str_1[i] |= char(msg[i * 8 + j] << (7 - j));
		}
	}
	*/


	/*
	const char* dir_path = "C:\\Users\\60566\\Desktop\\other\\stego-out";
	const char* out_path = "C:\\Users\\60566\\Desktop\\other\\new.jpg";
	//num��ʾ������Ϣ��ͼƬ�ĸ���
	int num = 31;
	int seed = 100;
	extractallms(dir_path, out_path,num,seed);
	*/


	const char* dir_path = argv[1];
	const char* out_path = argv[2];
	unsigned int num = (unsigned int)atoi(argv[3]);
	unsigned int seed = (unsigned int)atoi(argv[4]);
	extractallms(dir_path, out_path, num, seed);



	/*
	const char* dir_path = "C:\\Users\\luSir\\Desktop\\azw6_images";
	string txt_dir = "C:\\Users\\luSir\\Desktop\\newtxt\\newHDPhoto";
	string txtFormat = ".txt";
	unsigned int seed = 100;
	vector<string> files;
	//��ȡÿ��ͼƬ���ļ���
	getFiles((string)dir_path, files);
	//����ÿ���ļ�������ÿ��ͼƬ�����������Ĵ�С��ͳ���ܵ��ļ�������С
	int size = files.size();
	for (int i = 0; i < size; i++){
		string imageNum = to_string(i);
		string strr = txt_dir + imageNum + txtFormat;
		const char * txt_path = strr.data();
		const char * file_path = files[i].data();
		extractms(file_path, txt_path, seed);
		printf("��%d����ȡ��ɣ�\n", i);
	}
	*/
	return 0;
}
