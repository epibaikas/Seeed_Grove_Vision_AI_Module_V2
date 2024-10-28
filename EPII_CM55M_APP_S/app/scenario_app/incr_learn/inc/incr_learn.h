/*
 * incr_learn.h
 *
 *  Created on: Han 10, 2022
 *      Author: 904207
 */

#ifndef SCENARIO_APP_TFLM_PL_H_
#define SCENARIO_APP_TFLM_PL_H_


#define APP_BLOCK_FUNC() do{ \
	__asm volatile("b    .");\
	}while(0)

#define LINE_BUFFER_LEN 200
#define COMMAND_NAME_LEN 30

#define NUM_OF_IMGS_IN_RAM_BUFFER 400
#define NUM_OF_IMGS_IN_EEPROM_BUFFER 800
#define NUM_OF_IMGS_TOTAL NUM_OF_IMGS_IN_RAM_BUFFER + NUM_OF_IMGS_IN_EEPROM_BUFFER
#define BYTES_PER_IMG 785
#define DATA_BYTES_PER_IMG BYTES_PER_IMG - 1
#define FLASH_BASE_ADDRESS 0x00201000

#define NUM_OF_CLASSES 10
#define kNN_k 3

int app_main(void);

#endif /* SCENARIO_APP_TFLM_PL_H_ */