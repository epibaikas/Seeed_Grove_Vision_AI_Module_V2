#include <stdio.h>
#include "xprintf.h"
#include "spi_eeprom_comm.h"

#include "incr_learn.h"
#include "protocol_functions.h"
#include "util_functions.h"


void write_eeprom(struct FunctionArguments *fun_args) {
    int example_num = 0;
    int num_per_line = 8;
    int sscanf_ret_value = 0;

    sscanf_ret_value = sscanf(fun_args->param, "%d %d", &example_num, &num_per_line);
    if (sscanf_ret_value <= 1) {
        xprintf("ack_error: write_eeprom() parameters not parsed correctly\r\n");
        exit(1);
    }
    if (example_num < 0) {
        xprintf("ack_error: example_num cannot be < 0\r\n");
        exit(1);
    }

    // Determine flash_sector_num based on example_num
    int flash_sector_num = 0;
    uint32_t flash_sector_start_addr = EEPROM_BASE_ADDRESS;
    int flash_sector_idx = 0;

    get_example_flash_addr(example_num, &flash_sector_num, &flash_sector_start_addr, &flash_sector_idx, fun_args);

    // Read contents from flash sector to eeprom_sector_buffer 
	hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, flash_sector_start_addr, &(fun_args->eeprom_sector_buffer[0]), EEPROM_SECTOR_SIZE);

    // Erase flash sector
    hx_lib_spi_eeprom_erase_sector(USE_DW_SPI_MST_Q, flash_sector_start_addr, FLASH_SECTOR);

    xprintf("ack_begin %d\r\n", fun_args->seq_num);

    // Write new data to eeprom_sector_buffer
    write_buffer(&(fun_args->eeprom_sector_buffer[flash_sector_idx]), fun_args->bytes_per_example, num_per_line);

    // Write data in eeprom_sector_buffer to flash
    hx_lib_spi_eeprom_write(USE_DW_SPI_MST_Q, flash_sector_start_addr, &(fun_args->eeprom_sector_buffer[0]), EEPROM_SECTOR_SIZE, 0);
}

void read_eeprom(struct FunctionArguments *fun_args) {
    int example_num = 0;
    int num_per_line = 8;
    int sscanf_ret_value = 0;

    sscanf_ret_value = sscanf(fun_args->param, "%d %d", &example_num, &num_per_line);
    if (sscanf_ret_value <= 1) {
        xprintf("ack_error: read_eeprom() parameters not parsed correctly\r\n");
        exit(1);
    }
    if (example_num < 0) {
        xprintf("ack_error: example_num cannot be < 0\r\n");
        exit(1);
    }

    xprintf("ack_begin %d\r\n", fun_args->seq_num);

    // Determine flash_sector_num based on example_num
    int flash_sector_num = 0;
    uint32_t flash_sector_start_addr = EEPROM_BASE_ADDRESS;
    int flash_sector_idx = 0;

    get_example_flash_addr(example_num, &flash_sector_num, &flash_sector_start_addr, &flash_sector_idx, fun_args);

    // Read data to eeprom_buffer from flash
    hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, flash_sector_start_addr + (uint32_t)flash_sector_idx, &(fun_args->eeprom_buffer[0]), fun_args->bytes_per_example);
    read_buffer(&(fun_args->eeprom_buffer[0]), fun_args->bytes_per_example, sizeof(uint8_t), num_per_line);
}

void compute_dist_matrix(struct FunctionArguments *fun_args) {
    uint32_t bitshift = 0;
    int sscanf_ret_value = 0;

    sscanf_ret_value = sscanf(fun_args->param, "%lu", &bitshift);
    if (sscanf_ret_value <= 0) {
        xprintf("ack_error: compute_dist_matrix() parameters not parsed correctly\r\n");
        exit(1);
    }

    xprintf("ack_begin %d\r\n", fun_args->seq_num);

    int example_num = 0;
    int flash_sector_num = 0;
    uint32_t flash_sector_start_addr = EEPROM_BASE_ADDRESS;
    int flash_sector_idx = 0;

    uint32_t *self_dot_prod = calloc(fun_args->max_num_examples, sizeof(uint32_t));
    if (self_dot_prod == NULL) {
		xprintf("mem_error: memory allocation for self_dot_prod buffer failed\r\n");
		exit(1);
	}

    // Compute self-dot products
    for (int i = 0; i < fun_args->max_num_examples; i++) {
        if (i < fun_args->ram_buffer_size)
            self_dot_prod[i] = dot_prod_uint8_vect(fun_args->ram_buffer[i], fun_args->ram_buffer[i], fun_args->data_bytes_per_example);
        else {
            // Read data to eeprom_buffer from flash
            example_num = i - fun_args->ram_buffer_size;
            get_example_flash_addr(example_num, &flash_sector_num, &flash_sector_start_addr, &flash_sector_idx, fun_args);
            hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, flash_sector_start_addr + (uint32_t)flash_sector_idx, &(fun_args->eeprom_buffer[0]), fun_args->data_bytes_per_example);
            
            self_dot_prod[i] = dot_prod_uint8_vect(&(fun_args->eeprom_buffer[0]), &(fun_args->eeprom_buffer[0]), fun_args->data_bytes_per_example);
        }
        // xprintf("self_dot_prod[%u] = %u\r\n", i, self_dot_prod[i]);
    }

    // Compute distances
    uint32_t dist = 0;
    for (int i = 0; i < fun_args->max_num_examples; i++) {
        for (int j = i + 1; j < fun_args->max_num_examples; j++) {
            dist = self_dot_prod[i] + self_dot_prod[j];            
            if (i < fun_args->ram_buffer_size && j < fun_args->ram_buffer_size) {
                dist -= 2 * dot_prod_uint8_vect(fun_args->ram_buffer[i], fun_args->ram_buffer[j], fun_args->data_bytes_per_example);
                // xprintf("i = %d, j = %d, Cond 1\r\n", i, j);
            }

            if (i >= fun_args->ram_buffer_size && j < fun_args->ram_buffer_size) {
                example_num = i - fun_args->ram_buffer_size;
                get_example_flash_addr(example_num, &flash_sector_num, &flash_sector_start_addr, &flash_sector_idx, fun_args);
                hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, flash_sector_start_addr + (uint32_t)flash_sector_idx, &(fun_args->eeprom_buffer[0]), fun_args->data_bytes_per_example);

                dist -= 2 * dot_prod_uint8_vect(&(fun_args->eeprom_buffer[0]), fun_args->ram_buffer[j], fun_args->data_bytes_per_example);
                // xprintf("i = %d, j = %d, Cond 2\r\n", i, j);
            }

            if (i < fun_args->ram_buffer_size && j >= fun_args->ram_buffer_size) {
                example_num = j - fun_args->ram_buffer_size;
                get_example_flash_addr(example_num, &flash_sector_num, &flash_sector_start_addr, &flash_sector_idx, fun_args);
                hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, flash_sector_start_addr + (uint32_t)flash_sector_idx, &(fun_args->eeprom_buffer[0]), fun_args->data_bytes_per_example);

                dist -= 2 * dot_prod_uint8_vect(fun_args->ram_buffer[i], &(fun_args->eeprom_buffer[0]), fun_args->data_bytes_per_example);
                // xprintf("i = %d, j = %d, Cond 3\r\n", i, j);
            }

            if (i >= fun_args->ram_buffer_size && j >= fun_args->ram_buffer_size) {
                example_num = i - fun_args->ram_buffer_size;
                get_example_flash_addr(example_num, &flash_sector_num, &flash_sector_start_addr, &flash_sector_idx, fun_args);
                hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, flash_sector_start_addr + (uint32_t)flash_sector_idx, &(fun_args->eeprom_buffer[0]), fun_args->data_bytes_per_example);

                example_num = j - fun_args->ram_buffer_size;
                get_example_flash_addr(example_num, &flash_sector_num, &flash_sector_start_addr, &flash_sector_idx, fun_args);
                hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, flash_sector_start_addr + (uint32_t)flash_sector_idx, &(fun_args->eeprom_buffer_2[0]), fun_args->data_bytes_per_example);

                dist -= 2 * dot_prod_uint8_vect(&(fun_args->eeprom_buffer[0]), &(fun_args->eeprom_buffer_2[0]), fun_args->data_bytes_per_example);
                // xprintf("i = %d, j = %d, Cond 4\r\n", i, j);
            }

            // xprintf("i = %d, j = %d, %010u ", i, j, dist >> bitshift);
            set_symmetric_2D_array_value(&(fun_args->dist_matrix[0]), fun_args->max_num_examples, i, j, dist >> bitshift);
            // xprintf("%010u \r\n", get_symmetric_2D_array_value(&(fun_args->dist_matrix[0]), fun_args->max_num_examples, i, j));
        }
        // xprintf("\r\n");
    }

    // Set every cell on the diagonal equal to 0xFFFF
    for (int i = 0; i < fun_args->max_num_examples; i++) {
        set_symmetric_2D_array_value(&(fun_args->dist_matrix[0]), fun_args->max_num_examples, i, i, 0xFFFF);
    }

    free(self_dot_prod);
    xprintf("done\r\n");
}