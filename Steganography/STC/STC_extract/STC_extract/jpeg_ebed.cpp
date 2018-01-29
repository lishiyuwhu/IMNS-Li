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




void extractms(const char* file_path, u8 * msg, int &msglength,unsigned int seed){
	u8 *stego;
	int i, j, k, set_coverlength;
	int t = 0;
	int m = 0;//stego计数使用
	int flag = 0;
	short*** return_buf = NULL;
	short*** stego_buf = NULL;//存储stego图像的DCT系数
	unsigned int height_in_blocks, width_in_blocks;
	int num = 0;//存储秘密长度
	//unsigned int seed = (unsigned int)atoi(argv[3]);//获取随机种子
	if (read_coeff(file_path, &return_buf, &height_in_blocks, &width_in_blocks, &flag) == -1){
		return;
	}//读取原始图像DCT系数

	set_coverlength = height_in_blocks*width_in_blocks*(DCTSIZE2 - 1);
	//msgtemp = new u8[height_in_blocks*width_in_blocks*(DCTSIZE2 - 1) + 32];//存储获得的秘密文件
	stego = new u8[height_in_blocks*width_in_blocks*(DCTSIZE2 - 1)];


	for (i = 0; i < height_in_blocks; i++) {
		for (j = 0; j < width_in_blocks; j++){
			for (k = 1; k < DCTSIZE2; k++){
				stego[(i*width_in_blocks + j)*(DCTSIZE2 - 1) + k - 1] = u8(return_buf[i][j][k] & 1);
			}
		}
	}
	t = set_coverlength;

	rand_permutation(stego, t, seed);//对于所得的stego按照嵌入时相同的种子进行置乱

	for (i = 0; i < 32; i++){
		num += int(stego[i] << (31 - i));
	}

	msglength = num;
	stc_extract(stego + 32, set_coverlength - 32, msg, msglength);
	
	

	//释放资源

	for (i = 0; i < height_in_blocks; i++){
		for (j = 0; j < width_in_blocks; j++){
			free(return_buf[i][j]);
		}
		free(return_buf[i]);
	}
	free(return_buf);

	delete[] stego;


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
		//printf("第%d个图片隐藏容量区间为：(%d,%d]\n", i, smalltemp, bigtemp);
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
	allmsg = new u8[big*8];//暂时先这样写

	vector<string> files;
	//获取每个图片的文件名.这个地方有坑，读取图片按从0到1顺序读取。
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
	
	//开始写文件
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
	//num表示隐藏信息的图片的个数
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
	//获取每个图片的文件名
	getFiles((string)dir_path, files);
	//遍历每个文件，分析每个图片可隐藏容量的大小，统计总的文件容量大小
	int size = files.size();
	for (int i = 0; i < size; i++){
		string imageNum = to_string(i);
		string strr = txt_dir + imageNum + txtFormat;
		const char * txt_path = strr.data();
		const char * file_path = files[i].data();
		extractms(file_path, txt_path, seed);
		printf("第%d次提取完成！\n", i);
	}
	*/
	return 0;
}
