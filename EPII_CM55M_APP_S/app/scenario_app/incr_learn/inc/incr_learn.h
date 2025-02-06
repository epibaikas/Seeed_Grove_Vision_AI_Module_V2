/*
 * incr_learn.h
 *
 *  Created on: Han 10, 2022
 *      Author: 904207
 */

#ifndef INCR_LEARN_H
#define INCR_LEARN_H

#ifdef GROVE_VISION_WE2
	#define APP_BLOCK_FUNC() do{ \
		__asm volatile("b    .");\
		}while(0)
#elif HOST_PLATFORM
	#define SERIAL_PORT "/dev/ttys008"
	#define BAUDRATE 115200
#endif

#define EEPROM_TEMP_BUFFER_SIZE 1024
#define EEPROM_BASE_ADDRESS 0x00201000
#define EEPROM_SECTOR_SIZE 4096

#define LINE_BUFFER_LEN 200
#define COMMAND_NAME_LEN 30
#define PARAM_LEN 100

#define kNN_k 3

int app_main(void);

#endif