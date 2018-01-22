#include <stdio.h>
#include "jerror.h"
#include "jpeglib.h"
#include "joint.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <limits>

void get_quality_table(j_decompress_ptr srcinfo, unsigned short quality_table[DCTSIZE2*2]){
	short i,pos;
	pos = 0;
	for (i = 0; i < DCTSIZE2; i++){
          quality_table[pos] = srcinfo->quant_tbl_ptrs[0]->quantval[i];
		  pos ++;
    }
}

int getcoverlength(const char * file_path){
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	FILE *input_file;
	unsigned short flag_value;

	if ((input_file = fopen(file_path, "rb")) == NULL) {
		fprintf(stderr, "can't open %s\n", file_path);
		return -1;
	}
	// 判断是否为jpeg
	fread(&flag_value, 1, sizeof(flag_value), input_file);
	if (0xd8ff != flag_value){
		fprintf(stderr, "It's not a jpeg file!\n");
		fclose(input_file);
		return -1;
	}
	fseek(input_file, 0L, SEEK_SET);
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);
	/* Specify data source for decompression */
	jpeg_stdio_src(&cinfo, input_file);

	/* Read file header, set default decompression parameters */
	(void)jpeg_read_header(&cinfo, TRUE);
	unsigned int height = cinfo.comp_info->height_in_blocks;
	unsigned int width = cinfo.comp_info->width_in_blocks;
	int set_coverlength = height *width *(DCTSIZE2 - 1);
	return set_coverlength;
}

int getrange(const char * file_path, int & small_limit, int & big_limit){
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	FILE *input_file;
	unsigned short flag_value;

	if ((input_file = fopen(file_path, "rb")) == NULL) {
		fprintf(stderr, "can't open %s\n", file_path);
		return -1;
	}
	// 判断是否为jpeg
	fread(&flag_value, 1, sizeof(flag_value), input_file);
	if (0xd8ff != flag_value){
		fprintf(stderr, "It's not a jpeg file!\n");
		fclose(input_file);
		return -1;
	}
	fseek(input_file, 0L, SEEK_SET);
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);
	/* Specify data source for decompression */
	jpeg_stdio_src(&cinfo, input_file);

	/* Read file header, set default decompression parameters */
	(void)jpeg_read_header(&cinfo, TRUE);
	unsigned int height = cinfo.comp_info->height_in_blocks;
	unsigned int width = cinfo.comp_info->width_in_blocks;
	int set_coverlength = height *width *(DCTSIZE2 - 1);

	small_limit = (int)floor((double)(set_coverlength - 32) / 256) / 8;//256是2的（10-2）次方
	big_limit = (int)ceil((double)(set_coverlength - 32) / 2) / 8;


	/* close input file */
	fclose(input_file);
	return 0;
}

int read_coeff(const char*file_path,short**** return_buf,unsigned*height,unsigned* width,int* flag)
	{
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;

  FILE *input_file;

  jvirt_barray_ptr* coef_arrays;
  JBLOCKARRAY buffer;
  short*** temp_buf = *return_buf;
  int i,j;
  JDIMENSION blk_y,blk_x;
  unsigned short flag_value;
 // errno_t err;

  if ((input_file=fopen(file_path, "rb" ))==NULL) {
		fprintf(stderr, "can't open %s\n", file_path);
		return -1;
  }
  // 判断是否为jpeg
  fread(&flag_value, 1, sizeof(flag_value), input_file);
  if (0xd8ff != flag_value){
	fprintf(stderr, "It's not a jpeg file!\n");
	fclose(input_file);
	return -1;
  }
  fseek(input_file, 0L, SEEK_SET);

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  /* Specify data source for decompression */
  jpeg_stdio_src(&cinfo, input_file);

  /* Read file header, set default decompression parameters */
  (void) jpeg_read_header(&cinfo, TRUE);

  *height = cinfo.comp_info->height_in_blocks;
  *width = cinfo.comp_info->width_in_blocks;

  if (*flag == 0){
	// 为数组分配空间
	/**flag = 1;*/
	temp_buf = (short***)malloc(sizeof(short**) * cinfo.comp_info->height_in_blocks);
		for (blk_y = 0; blk_y < cinfo.comp_info->height_in_blocks; blk_y++){
		temp_buf[blk_y] = (short**)malloc(sizeof(short*) * cinfo.comp_info->width_in_blocks);
		for(blk_x = 0; blk_x < cinfo.comp_info->width_in_blocks; blk_x++){
			temp_buf[blk_y][blk_x] = (short*)malloc(sizeof(short) * DCTSIZE2);
		}
	}
  }

  coef_arrays = jpeg_read_coefficients(&cinfo);
  buffer = (cinfo.mem->access_virt_barray)
    	((j_common_ptr) &cinfo, coef_arrays[0], 0, 1, FALSE);
  for (blk_y = 0; blk_y < cinfo.comp_info->height_in_blocks; blk_y++) {
	  for (blk_x = 0; blk_x < cinfo.comp_info->width_in_blocks; blk_x++){
		  for (i = 0; i < DCTSIZE; i++){        /* for each row in block */
				for (j = 0; j < DCTSIZE; j++){      /* for each column in block */
					//printf("%d\t",buffer[blk_y][blk_x][i*DCTSIZE+j]);
					temp_buf[blk_y][blk_x][i*DCTSIZE+j] = buffer[blk_y][blk_x][i*DCTSIZE+j];
				}
				//printf("\n");
		  }
		  //system("pause");
	  }
  }
  *return_buf = temp_buf;

  /* done with cinfo */
  (void)jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  /* close input file */
  fclose(input_file);
  
  return 0;
}










