#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "jpeglib.h"
#include "jerror.h"


int get_quality_table(j_decompress_ptr srcinfo, unsigned short quality_table[DCTSIZE2*2]){
	short i,pos;
	pos = 0;
	for (i = 0; i < DCTSIZE2; i++){
          quality_table[pos] = srcinfo->quant_tbl_ptrs[0]->quantval[i];
		  pos ++;
    }

	if(srcinfo->quant_tbl_ptrs[1] != NULL){
		for (i = 0; i < DCTSIZE2; i++){
			  quality_table[pos] = srcinfo->quant_tbl_ptrs[1]->quantval[i];
			  pos ++;
		}
		return 2;
	}
	else {
		return 1;
	}
}

void get_table(unsigned short quality_table[DCTSIZE2*2], unsigned short quality_value){
	unsigned short pos;
	static const unsigned int std_quant_tbl[128] = {  
	  16,  11,  10,  16,  24,  40,  51,  61,  
	  12,  12,  14,  19,  26,  58,  60,  55,  
	  14,  13,  16,  24,  40,  57,  69,  56,  
	  14,  17,  22,  29,  51,  87,  80,  62,  
	  18,  22,  37,  56,  68, 109, 103,  77, 
	  24,  35,  55,  64,  81, 104, 113,  92,
	  49,  64,  78,  87, 103, 121, 120, 101,  
	  72,  92,  95,  98, 112, 100, 103,  99,

	  17,  18,  24,  47,  99,  99,  99,  99,  
	  18,  21,  26,  66,  99,  99,  99,  99,  
	  24,  26,  56,  99,  99,  99,  99,  99,  
	  47,  66,  99,  99,  99,  99,  99,  99,  
	  99,  99,  99,  99,  99,  99,  99,  99,  
	  99,  99,  99,  99,  99,  99,  99,  99,  
	  99,  99,  99,  99,  99,  99,  99,  99,  
	  99,  99,  99,  99,  99,  99,  99,  99  
	};  
	if(quality_value <= 0){
		quality_value = 1;
	}
	if (quality_value > 100){
		quality_value = 100;
	}
	if (quality_value < 50){
		quality_value = 5000 / quality_value;
	}else{
		quality_value = 200 - quality_value*2;
	}

	for (pos = 0; pos < 128; pos++){
		quality_table[pos] = (std_quant_tbl[pos] * quality_value + 50)/100;
		if (quality_table[pos] == 0) quality_table[pos] = 1;
		else if(quality_table[pos] > 32767) quality_table[pos] = 32767;
	}

}

int cmp_table(unsigned short quality_table[DCTSIZE2*2], unsigned short temp_table[DCTSIZE2*2], int num_qt){
	unsigned short pos = 0;
	for (pos; pos < DCTSIZE2; pos++){
		if(quality_table[pos] != temp_table[pos]){
			return 0;
		}
	}
	if(num_qt == 2){
		for (pos; pos < DCTSIZE2*2; pos++){
			if(quality_table[pos] != temp_table[pos]){
				return 1;
			}
		}
	}
	return 1;
}

void disp_table(unsigned short quality_table[DCTSIZE2*2]){
	unsigned short i,j;
	for(i = 0; i < DCTSIZE; i++){
	  for(j = 0; j < DCTSIZE; j++){
		printf("%d\t", quality_table[i*DCTSIZE+j]);
	  }
	  printf("\n");
  }
  printf("\n");
  for(i = 0; i < DCTSIZE; i++){
	  for(j = 0; j < DCTSIZE; j++){
		printf("%d\t", quality_table[DCTSIZE2+i*DCTSIZE+j]);
	  }
	  printf("\n");
  }
}
unsigned short calc_quality(unsigned short quality_table[DCTSIZE2*2], int num_qt){
	unsigned short i;
	unsigned short quality_value = 0;
	unsigned short temp_table[DCTSIZE2*2];
	for (i = 100; i > 0 ; i--){
		get_table(temp_table, i);
		/*if (i == 63){
			disp_table(temp_table);
		}*/
		
		if( cmp_table(quality_table, temp_table, num_qt) == TRUE ){
			quality_value = i;
			break;
		}
	}
	return quality_value;
}

int jpeg_crop(const char* file_path,unsigned* height,unsigned* width,unsigned*dct_height,unsigned* dct_width,const char* crop_jpeg){

  JDIMENSION X_BEGIN = 4;	//裁剪时水平方向开始像素（从０算起）
  JDIMENSION Y_BEGIN = 4;	//垂直方向开始像素
  unsigned short quality_table[DCTSIZE2*2];  // 量化表
  int num_qt;	// 量化表数量：1或2
  unsigned short quality_value;	// 量化因子
  static const char * outfilename;
  struct jpeg_decompress_struct srcinfo;
  struct jpeg_compress_struct dstinfo;
  struct jpeg_error_mgr jsrcerr, jdsterr;
 // unsigned xxxxxx;


  unsigned char *jpgbuf;      //存放解压后一行图像数据  
  unsigned int row_stride;    //每行图像的字节数
  //unsigned int i,j;
  unsigned char * sub_buf;
  //djpeg_dest_ptr dest_mgr = NULL;
  FILE * input_file;
  FILE * output_file;
  errno_t err;
  JDIMENSION num_scanlines;

  unsigned short flag_value;

  if ((err= fopen_s(&input_file,file_path, "rb")) != 0) {
		fprintf(stderr, "can't open %s\n", file_path);
		return 0;
  }
  // 判断是否为jpeg
  fread(&flag_value, 1, sizeof(flag_value), input_file);
  if (0xd8ff != flag_value){
	fprintf(stderr, "It's not a jpeg file!\n");
	fclose(input_file);
	return 0;
  }
  fseek(input_file, 0L, SEEK_SET);

  srcinfo.err = jpeg_std_error(&jsrcerr);
  jpeg_create_decompress(&srcinfo);
  dstinfo.err = jpeg_std_error(&jdsterr);
  jpeg_create_compress(&dstinfo);

  outfilename = crop_jpeg;
  if ((err= fopen_s(&output_file,outfilename, "wb")) != 0) {
      fprintf(stderr,"can't open %s\n", outfilename);
	  fclose(input_file);
      exit(EXIT_FAILURE);
  }

  /* Specify data source for decompression */
  jpeg_stdio_src(&srcinfo, input_file);

  /* Read file header, set default decompression parameters */
  (void) jpeg_read_header(&srcinfo, TRUE);

  if(srcinfo.image_height < 5 || srcinfo.image_width < 5){
	fprintf(stderr, "the jpeg's size is too small!");
	fclose(input_file);
	return 0;
  }

  num_qt = get_quality_table(&srcinfo, quality_table);	//获取量化表  
  
  quality_value = calc_quality(quality_table, num_qt);	// 计算质量因子
  if (quality_value == 0) {
	fprintf(stderr, "quality calc error!");
	return 0;	
  }		

   /* Start decompressor */
  (void) jpeg_start_decompress(&srcinfo);

  *height=srcinfo.output_height-4;
  *width=srcinfo.output_width-4;
 *dct_width=srcinfo.comp_info->downsampled_width;
 *dct_height=srcinfo.comp_info->downsampled_height;
  // 存储图像数据
  row_stride = srcinfo.output_width * srcinfo.output_components;
  jpgbuf = (unsigned char *) malloc(sizeof(unsigned char) * row_stride);
  memset(jpgbuf, 0, sizeof(unsigned char) * row_stride);

  /* Specify data destination for compression */
  jpeg_stdio_dest(&dstinfo, output_file);

  dstinfo.image_width = srcinfo.image_width - X_BEGIN;             
  dstinfo.image_height =  srcinfo.image_height - Y_BEGIN;   
  dstinfo.input_components = srcinfo.output_components;          
  dstinfo.in_color_space = srcinfo.out_color_space;     
  jpeg_set_defaults(&dstinfo);    
  jpeg_set_quality(&dstinfo, quality_value , TRUE );           // 设置质量因子
  dstinfo.optimize_coding = TRUE;
  /* Start compressor */
  jpeg_start_compress(&dstinfo, TRUE);

  // 读取像素值并写入到压缩图像
  /* 去掉前面不要的Y_BEGIN行数据 */
  while (srcinfo.output_scanline < Y_BEGIN) {
     num_scanlines = jpeg_read_scanlines(&srcinfo, &jpgbuf, 1);
   }

  while (srcinfo.output_scanline < srcinfo.output_height) {
     num_scanlines = jpeg_read_scanlines(&srcinfo, &jpgbuf, 1);
	 sub_buf = jpgbuf + X_BEGIN*srcinfo.output_components;  // 去掉前X_BEGIN列
	(void) jpeg_write_scanlines(&dstinfo, &sub_buf, num_scanlines);
  }
   

  jpeg_finish_compress(&dstinfo);
  jpeg_destroy_compress(&dstinfo);
  (void) jpeg_finish_decompress(&srcinfo);
  jpeg_destroy_decompress(&srcinfo);
  
  free(jpgbuf);
  /* close file */
  fclose(input_file);
  fclose(output_file);
  
  return 1;
}