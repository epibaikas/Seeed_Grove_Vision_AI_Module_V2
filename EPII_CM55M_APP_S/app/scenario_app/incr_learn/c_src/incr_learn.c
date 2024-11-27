#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "powermode_export.h"

#define WATCH_DOG_TIMEOUT_TH	(500) //ms

#ifdef TRUSTZONE_SEC
#ifdef FREERTOS
/* Trustzone config. */
//
/* FreeRTOS includes. */
//#include "secure_port_macros.h"
#else
#if (__ARM_FEATURE_CMSE & 1) == 0
#error "Need ARMv8-M security extensions"
#elif (__ARM_FEATURE_CMSE & 2) == 0
#error "Compile with --cmse"
#endif
#include "arm_cmse.h"
//#include "veneer_table.h"
//
#endif
#endif

#include "WE2_device.h"

#include "spi_master_protocol.h"
#include "hx_drv_spi.h"
#include "spi_eeprom_comm.h"
#include "board.h"
#include "xprintf.h"
#include "incr_learn.h"
#include "board.h"
#include "WE2_core.h"
#include "hx_drv_scu.h"
#include "hx_drv_swreg_aon.h"
#include "hx_drv_uart.h"
#ifdef IP_sensorctrl
#include "hx_drv_sensorctrl.h"
#endif
#ifdef IP_xdma
#include "hx_drv_xdma.h"
#include "sensor_dp_lib.h"
#endif
#ifdef IP_cdm
#include "hx_drv_cdm.h"
#endif
#ifdef IP_gpio
#include "hx_drv_gpio.h"
#endif
#include "hx_drv_pmu_export.h"
#include "hx_drv_pmu.h"
#include "powermode.h"
//#include "dp_task.h"
#include "BITOPS.h"


#include "event_handler.h"
#include "memory_manage.h"
#include "hx_drv_watchdog.h"
#include <arm_math_types.h>
#include <arm_math.h>
#include "protocol_functions.h"
#include "util_functions.h"

#ifdef EPII_FPGA
#define DBG_APP_LOG             (1)
#else
#define DBG_APP_LOG             (0)
#endif
#if DBG_APP_LOG
    #define dbg_app_log(fmt, ...)       xprintf(fmt, ##__VA_ARGS__)
#else
    #define dbg_app_log(fmt, ...)
#endif

#define TOTAL_STEP_TICK 1
#define TOTAL_STEP_TICK_DBG_LOG 0

#if TOTAL_STEP_TICK
#define CPU_CLK	0xffffff+1
#endif

void pinmux_init();

/* Init SPI master pin mux (share with SDIO) */
void spi_m_pinmux_cfg(SCU_PINMUX_CFG_T *pinmux_cfg)
{
	pinmux_cfg->pin_pb2 = SCU_PB2_PINMUX_SPI_M_DO_1;        /*!< pin PB2*/
	pinmux_cfg->pin_pb3 = SCU_PB3_PINMUX_SPI_M_DI_1;        /*!< pin PB3*/
	pinmux_cfg->pin_pb4 = SCU_PB4_PINMUX_SPI_M_SCLK_1;      /*!< pin PB4*/
	pinmux_cfg->pin_pb11 = SCU_PB11_PINMUX_SPI_M_CS;        /*!< pin PB11*/
}

void uart_pinmux_cfg()
{
	// UART0 pin mux configuration
	hx_drv_scu_set_PB0_pinmux(SCU_PB0_PINMUX_UART0_RX_1, 1);
	hx_drv_scu_set_PB1_pinmux(SCU_PB1_PINMUX_UART0_TX_1, 1);
}

void pinmux_init()
{
	SCU_PINMUX_CFG_T pinmux_cfg;

	hx_drv_scu_get_all_pinmux_cfg(&pinmux_cfg);

	/* Init SPI master pin mux (share with SDIO) */
	spi_m_pinmux_cfg(&pinmux_cfg);

	hx_drv_scu_set_all_pinmux_cfg(&pinmux_cfg, 1);

	/* Configure UART pin mux */
	uart_pinmux_cfg();
}

/*!
 * @brief Main function
 */
int app_main(void) {

	uint32_t wakeup_event;
	uint32_t wakeup_event1;
	uint32_t freq=0;

	hx_drv_pmu_get_ctrl(PMU_pmu_wakeup_EVT, &wakeup_event);
	hx_drv_pmu_get_ctrl(PMU_pmu_wakeup_EVT1, &wakeup_event1);

    hx_drv_swreg_aon_get_pllfreq(&freq);
    xprintf("wakeup_event=0x%x,WakeupEvt1=0x%x, freq=%d\n", wakeup_event, wakeup_event1, freq);

    pinmux_init();

#ifdef __GNU__
	xprintf("__GNUC \n");
	extern char __mm_start_addr__;
	xprintf("__mm_start_addr__ address: %x\r\n",&__mm_start_addr__);
	mm_set_initial((int)(&__mm_start_addr__), 0x00200000-((int)(&__mm_start_addr__)-0x34000000));
#else
	static uint8_t mm_start_addr __attribute__((section(".bss.mm_start_addr")));
	xprintf("mm_start_addr address: %x \r\n",&mm_start_addr);
	mm_set_initial((int)(&mm_start_addr), 0x00200000-((int)(&mm_start_addr)-0x34000000));
#endif

	uint8_t id_info = 2;
	char line_buf[LINE_BUFFER_LEN];
	char command_name[COMMAND_NAME_LEN];
	int sscanf_ret_value;

	int seq_num;
	int seq_num_begin;
	char *param;

	// Initialize eeprom
	printf("Init EEPROM...\r\n");
	hx_lib_spi_eeprom_open(USE_DW_SPI_MST_Q);
	hx_lib_spi_eeprom_read_ID(USE_DW_SPI_MST_Q, &id_info);
	printf("SPI ID info: %u\r\n", id_info);
	xprintf("Init complete\r\n");

	// Allocate memory for RAM buffer
	uint8_t **ram_buffer = (uint8_t **)calloc(NUM_OF_IMGS_IN_RAM_BUFFER, sizeof(uint8_t *));
	if (ram_buffer == NULL) {
		xprintf("mem_error: memory allocation for ram_buffer failed\r\n");
		exit(1);
	}

	for (int i = 0; i < NUM_OF_IMGS_IN_RAM_BUFFER; i++) {
		ram_buffer[i] = (uint8_t *)calloc(BYTES_PER_IMG, sizeof(uint8_t));
		if (ram_buffer[i] == NULL) {
			xprintf("mem_error: memory allocation for ram_buffer[%d] failed\r\n", i);
			exit(1);
		}
	}

	// Allocate memory for distance matrix
	uint16_t* dist_matrix = allocate_symmetric_2D_array(NUM_OF_IMGS_TOTAL);

	// Create eeprom buffers;
	static uint8_t eeprom_buffer[BYTES_PER_IMG] = {0};
	static uint8_t eeprom_buffer_2[BYTES_PER_IMG] = {0};
	static uint8_t eeprom_sector_buffer[FLASH_SECTOR_SIZE] = {0};
	static uint8_t labels[NUM_OF_IMGS_TOTAL] = {0};

	struct FunctionArguments fun_args;
	fun_args.ram_buffer = ram_buffer;
	fun_args.eeprom_buffer = eeprom_buffer;
	fun_args.eeprom_buffer_2 = eeprom_buffer_2;
	fun_args.eeprom_sector_buffer = eeprom_sector_buffer;
	fun_args.dist_matrix = dist_matrix;
	fun_args.labels = labels;
	fun_args.random_seed = 1;

	xprintf("Addr of dist_matrix: 0x%08x\r\n", dist_matrix);
	xprintf("Addr of eeprom_buffer: 0x%08x\r\n", eeprom_buffer);
	xprintf("Addr of eeprom_buffer_2: 0x%08x\r\n", eeprom_buffer_2);
	xprintf("Addr of eeprom_sector_buffer: 0x%08x\r\n", eeprom_sector_buffer);
	xprintf("Addr of labels buffer: 0x%08x\r\n", labels);
	xprintf("RAND_MAX: 0x%08x\r\n", RAND_MAX);

	xprintf("Board initialisation complete\r\n");
	//-----------------------------------------------------

	while(1) {
		// Get new line
		xgets(line_buf, LINE_BUFFER_LEN);
		
		// Check for errors in begin statement
		if(strncmp(line_buf, "begin ", 6) != 0) {
			xprintf("ack_error: missing begin\r\n");
			exit(1);
		}

		// Check for errors in parsing the seq_num and command_name
		sscanf_ret_value = sscanf(line_buf, "begin %d", &seq_num);
		if (sscanf_ret_value <= 0) {
			xprintf("ack_error: seq_num not parsed correctly\r\n");
			exit(1);
		}

		sscanf_ret_value = sscanf(line_buf, "begin %*d %s", command_name);
		if (sscanf_ret_value <= 0) {
			xprintf("ack_error: command_name not parsed correctly\r\n");
			exit(1);
		}

		// Get function parameters, checks for parsing errors take place inside the function
		sscanf(line_buf, "begin %*d %*s %[^\r]", param);

		seq_num_begin = seq_num;
		fun_args.seq_num = seq_num;
		fun_args.param = param;

		// Call the function specified by command_name
		function_pointer func_ptr;
		func_ptr = lookup_function(command_name);
		func_ptr(&fun_args);

		xgets(line_buf, LINE_BUFFER_LEN);
		// Check for errors in end statement
		if(strncmp(line_buf, "end ", 3) != 0) {
			xprintf("ack_error: missing end\r\n");
			exit(1);
		}

		sscanf_ret_value = sscanf(line_buf, "end %d", &seq_num);
		if (sscanf_ret_value <= 0) {
			xprintf("ack_error: seq_num not parsed correctly in end statement\r\n");
			exit(1);
		} else if (seq_num != seq_num_begin) {
			xprintf("ack_error: seq_num not matching seq_number from begin statement\r\n");
			exit(1);
		}

		xprintf("ack_end %d\r\n", seq_num);
	};

	return 0;
}
