/*
 * incr_learn.h
 *
 *  Created on: Han 10, 2022
 *      Author: 904207
 */

#ifndef INCR_LEARN_H
#define INCR_LEARN_H


#define APP_BLOCK_FUNC() do{ \
	__asm volatile("b    .");\
	}while(0)

#define LINE_BUFFER_LEN 200
#define COMMAND_NAME_LEN 30

#define NUM_OF_IMGS_IN_RAM_BUFFER 400
#define NUM_OF_IMGS_IN_EEPROM_BUFFER 800
#define NUM_OF_IMGS_TOTAL (NUM_OF_IMGS_IN_RAM_BUFFER + NUM_OF_IMGS_IN_EEPROM_BUFFER)
#define BYTES_PER_IMG 785
#define DATA_BYTES_PER_IMG BYTES_PER_IMG - 1

#define FLASH_BASE_ADDRESS 0x00201000
#define FLASH_SECTOR_SIZE 4096
#define EXAMPLES_PER_FLASH_SECTOR (FLASH_SECTOR_SIZE / BYTES_PER_IMG)

#define NUM_OF_CLASSES 10
#define kNN_k 3

int app_main(void);

#endif