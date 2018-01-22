#ifndef JOINT_H
#define JOINT_H
	#define DCT_SIZE 8 
	int getrange(const char * file_path, int & small_limit, int & big_limit);
	int getcoverlength(const char * file_path);
	int jpeg_crop(const char* ,unsigned*,unsigned*,unsigned*,unsigned*,const char* );
	int read_coeff(const char*,short****,unsigned* height,unsigned* width,int*);
    int write_coeff(const char*,const char* ,short*);
#endif