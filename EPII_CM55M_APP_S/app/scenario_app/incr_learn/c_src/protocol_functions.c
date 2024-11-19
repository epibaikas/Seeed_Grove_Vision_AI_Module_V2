#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xprintf.h"
#include "spi_eeprom_comm.h"
#include "incr_learn.h"
#include "protocol_functions.h"
#include "util_functions.h"

void write_ram_buffer(struct FunctionArguments *fun_args) {
    int example_num = 0;
    int num_per_line = 8;
    int sscanf_ret_value = 0;

    sscanf_ret_value = sscanf(fun_args->param, "%d %d", &example_num, &num_per_line);
    if (sscanf_ret_value <= 1) {
        xprintf("ack_error: write_ram_buffer() parameters not parsed correctly\r\n");
        exit(1);
    }

    xprintf("ack_begin %d\r\n", fun_args->seq_num);
    write_buffer(&(fun_args->ram_buffer[example_num][0]), BYTES_PER_IMG, num_per_line);
}

void read_ram_buffer(struct FunctionArguments *fun_args) {
    int example_num = 0;
    int num_per_line = 8;
    int sscanf_ret_value = 0;

    sscanf_ret_value = sscanf(fun_args->param, "%d %d", &example_num, &num_per_line);
    if (sscanf_ret_value <= 1) {
        xprintf("ack_error: read_ram_buffer() parameters not parsed correctly\r\n");
        exit(1);
    }

    xprintf("ack_begin %d\r\n", fun_args->seq_num);

    read_buffer(&(fun_args->ram_buffer[example_num][0]), BYTES_PER_IMG, sizeof(uint8_t), num_per_line);
}

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
    uint32_t flash_sector_start_addr = FLASH_BASE_ADDRESS;
    int flash_sector_idx = 0;

    get_example_flash_addr(example_num, &flash_sector_num, &flash_sector_start_addr, &flash_sector_idx);

    // Read contents from flash sector to eeprom_sector_buffer 
	hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, flash_sector_start_addr, &(fun_args->eeprom_sector_buffer[0]), FLASH_SECTOR_SIZE);

    // Erase flash sector
    hx_lib_spi_eeprom_erase_sector(USE_DW_SPI_MST_Q, flash_sector_start_addr, FLASH_SECTOR);

    xprintf("ack_begin %d\r\n", fun_args->seq_num);

    // Write new data to eeprom_sector_buffer
    write_buffer(&(fun_args->eeprom_sector_buffer[flash_sector_idx]), BYTES_PER_IMG, num_per_line);

    // Write data in eeprom_sector_buffer to flash
    hx_lib_spi_eeprom_write(USE_DW_SPI_MST_Q, flash_sector_start_addr, &(fun_args->eeprom_sector_buffer[0]), FLASH_SECTOR_SIZE, 0);
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
    uint32_t flash_sector_start_addr = FLASH_BASE_ADDRESS;
    int flash_sector_idx = 0;

    get_example_flash_addr(example_num, &flash_sector_num, &flash_sector_start_addr, &flash_sector_idx);

    // Read data to eeprom_buffer from flash
    hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, flash_sector_start_addr + (uint32_t)flash_sector_idx, &(fun_args->eeprom_buffer[0]), BYTES_PER_IMG);
    read_buffer(&(fun_args->eeprom_buffer[0]), BYTES_PER_IMG, sizeof(uint8_t), num_per_line);
}

void read_labels_buffer(struct FunctionArguments *fun_args) {
    int num_per_line = 8;
    int sscanf_ret_value = 0;

    sscanf_ret_value = sscanf(fun_args->param, "%d", &num_per_line);
    if (sscanf_ret_value <= 0) {
        xprintf("ack_error: read_labels_buffer() parameters not parsed correctly\r\n");
        exit(1);
    }

    xprintf("ack_begin %d\r\n", fun_args->seq_num);

    update_labels_buffer(fun_args);
    read_buffer(&(fun_args->labels[0]), NUM_OF_IMGS_TOTAL, sizeof(uint8_t), num_per_line);
}

void compute_dist_matrix(struct FunctionArguments *fun_args) {
    xprintf("ack_begin %d\r\n", fun_args->seq_num);

    int example_num = 0;
    int flash_sector_num = 0;
    uint32_t flash_sector_start_addr = FLASH_BASE_ADDRESS;
    int flash_sector_idx = 0;

    uint32_t *self_dot_prod = calloc(NUM_OF_IMGS_TOTAL, sizeof(uint32_t));
    if (self_dot_prod == NULL) {
		xprintf("mem_error: memory allocation for self_dot_prod buffer failed\r\n");
		exit(1);
	}

    // Compute self-dot products
    for (int i = 0; i < NUM_OF_IMGS_TOTAL; i++) {
        if (i < NUM_OF_IMGS_IN_RAM_BUFFER)
            self_dot_prod[i] = dot_prod_uint8_vect(fun_args->ram_buffer[i], fun_args->ram_buffer[i], DATA_BYTES_PER_IMG);
        else {
            // Read data to eeprom_buffer from flash
            example_num = i - NUM_OF_IMGS_IN_RAM_BUFFER;
            get_example_flash_addr(example_num, &flash_sector_num, &flash_sector_start_addr, &flash_sector_idx);
            hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, flash_sector_start_addr + (uint32_t)flash_sector_idx, &(fun_args->eeprom_buffer[0]), DATA_BYTES_PER_IMG);
            
            self_dot_prod[i] = dot_prod_uint8_vect(&(fun_args->eeprom_buffer[0]), &(fun_args->eeprom_buffer[0]), DATA_BYTES_PER_IMG);
        }
        // xprintf("self_dot_prod[%u] = %u\r\n", i, self_dot_prod[i]);
    }

    // Compute distances
    uint32_t dist = 0;
    for (int i = 0; i < NUM_OF_IMGS_TOTAL; i++) {
        for (int j = i + 1; j < NUM_OF_IMGS_TOTAL; j++) {
            dist = self_dot_prod[i] + self_dot_prod[j];            
            if (i < NUM_OF_IMGS_IN_RAM_BUFFER && j < NUM_OF_IMGS_IN_RAM_BUFFER) {
                dist -= 2 * dot_prod_uint8_vect(fun_args->ram_buffer[i], fun_args->ram_buffer[j], DATA_BYTES_PER_IMG);
                // xprintf("Cond 1 ");
            }

            if (i >= NUM_OF_IMGS_IN_RAM_BUFFER && j < NUM_OF_IMGS_IN_RAM_BUFFER) {
                example_num = i - NUM_OF_IMGS_IN_RAM_BUFFER;
                get_example_flash_addr(example_num, &flash_sector_num, &flash_sector_start_addr, &flash_sector_idx);
                hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, flash_sector_start_addr + (uint32_t)flash_sector_idx, &(fun_args->eeprom_buffer[0]), DATA_BYTES_PER_IMG);

                dist -= 2 * dot_prod_uint8_vect(&(fun_args->eeprom_buffer[0]), fun_args->ram_buffer[j], DATA_BYTES_PER_IMG);
                // xprintf("Cond 2 ");
            }

            if (i < NUM_OF_IMGS_IN_RAM_BUFFER && j >= NUM_OF_IMGS_IN_RAM_BUFFER) {
                example_num = j - NUM_OF_IMGS_IN_RAM_BUFFER;
                get_example_flash_addr(example_num, &flash_sector_num, &flash_sector_start_addr, &flash_sector_idx);
                hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, flash_sector_start_addr + (uint32_t)flash_sector_idx, &(fun_args->eeprom_buffer[0]), DATA_BYTES_PER_IMG);

                dist -= 2 * dot_prod_uint8_vect(fun_args->ram_buffer[i], &(fun_args->eeprom_buffer[0]), DATA_BYTES_PER_IMG);
                // xprintf("Cond 3 ");
            }

            if (i >= NUM_OF_IMGS_IN_RAM_BUFFER && j >= NUM_OF_IMGS_IN_RAM_BUFFER) {
                example_num = i - NUM_OF_IMGS_IN_RAM_BUFFER;
                get_example_flash_addr(example_num, &flash_sector_num, &flash_sector_start_addr, &flash_sector_idx);
                hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, flash_sector_start_addr + (uint32_t)flash_sector_idx, &(fun_args->eeprom_buffer[0]), DATA_BYTES_PER_IMG);

                example_num = j - NUM_OF_IMGS_IN_RAM_BUFFER;
                get_example_flash_addr(example_num, &flash_sector_num, &flash_sector_start_addr, &flash_sector_idx);
                hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, flash_sector_start_addr + (uint32_t)flash_sector_idx, &(fun_args->eeprom_buffer_2[0]), DATA_BYTES_PER_IMG);

                dist -= 2 * dot_prod_uint8_vect(&(fun_args->eeprom_buffer[0]), &(fun_args->eeprom_buffer_2[0]), DATA_BYTES_PER_IMG);
                // xprintf("Cond 4 ");
            }

            // xprintf("%010u ", dist);
            set_symmetric_2D_array_value(&(fun_args->dist_matrix[0]), NUM_OF_IMGS_TOTAL, i, j, dist >> 12);
            // xprintf("%010u ", get_symmetric_2D_array_value(&(fun_args->dist_matrix[0]), NUM_OF_IMGS_TOTAL, i, j));
        }
        // xprintf("\r\n");
    }

    // Set every cell on the diagonal equal to 0xFFFF
    for (int i = 0; i < NUM_OF_IMGS_TOTAL; i++) {
        set_symmetric_2D_array_value(&(fun_args->dist_matrix[0]), NUM_OF_IMGS_TOTAL, i, i, 0xFFFF);
    }

    free(self_dot_prod);
    xprintf("done\r\n");
}

void read_dist_matrix(struct FunctionArguments *fun_args) {
    int num_per_line = 8;
    int sscanf_ret_value = 0;

    sscanf_ret_value = sscanf(fun_args->param, "%d", &num_per_line);
    if (sscanf_ret_value <= 0) {
        xprintf("ack_error: read_dist_matrix() parameters not parsed correctly\r\n");
        exit(1);
    }

    xprintf("ack_begin %d\r\n", fun_args->seq_num);

    uint32_t N = NUM_OF_IMGS_TOTAL;
    uint32_t size = (N * (N + 1)) / 2;

    read_buffer(&(fun_args->dist_matrix[0]), size, sizeof(uint16_t), num_per_line);
}

void rand_subset_selection(struct FunctionArguments *fun_args) {
    int balanced_subset = 0; 
    int num_per_line = 8;
    int sscanf_ret_value = 0;

    sscanf_ret_value = sscanf(fun_args->param, "%d %d",  &balanced_subset, &num_per_line);
    if (sscanf_ret_value <= 0) {
        xprintf("ack_error: rand_subset_selection() parameters not parsed correctly\r\n");
        exit(1);
    }
    
    xprintf("ack_begin %d\r\n", fun_args->seq_num);

    // Generate subset
    uint16_t* subset_idxs = calloc(NUM_OF_IMGS_IN_EEPROM_BUFFER, sizeof(uint16_t));
    uint8_t* predicted_labels = calloc(NUM_OF_IMGS_TOTAL, sizeof(uint8_t));
    if (subset_idxs == NULL || predicted_labels == NULL) {
        xprintf("mem_error: memory allocation for subset_idxs or predicted_labels failed\r\n");
		exit(1);
    }

    // Update the labels buffer
    update_labels_buffer(fun_args);
    
    if (balanced_subset == 1) {
        get_random_bal_subset(fun_args->labels, subset_idxs);
    } else {
        get_random_subset(NUM_OF_IMGS_IN_EEPROM_BUFFER, NUM_OF_IMGS_TOTAL, subset_idxs);
    }

    // Classify all examples using the subset
    classify_training_set(fun_args, subset_idxs, predicted_labels);

    // Output generated subset and label predictions
    read_buffer(subset_idxs, NUM_OF_IMGS_IN_EEPROM_BUFFER, sizeof(uint16_t), num_per_line);
    xprintf("subset_idxs_read_done\r\n");
    read_buffer(predicted_labels, NUM_OF_IMGS_TOTAL, sizeof(uint8_t), num_per_line);
    xprintf("predicted_labels_read_done\r\n");

    // Move subset data examples located in RAM to EEPROM
    move_subset_to_eeprom(subset_idxs, NUM_OF_IMGS_IN_EEPROM_BUFFER, fun_args);

    free(subset_idxs);
    free(predicted_labels);
}

void set_random_seed(struct FunctionArguments *fun_args) {
    unsigned int random_seed = 1;
    int sscanf_ret_value = 0;

    sscanf_ret_value = sscanf(fun_args->param, "%u", &random_seed);
    if (sscanf_ret_value <= 0) {
        xprintf("ack_error: set_random_seed() parameters not parsed correctly\r\n");
        exit(1);
    }
    
    xprintf("ack_begin %d\r\n", fun_args->seq_num);
    srand(random_seed);

    fun_args->random_seed = random_seed;

    xprintf("random seed set to: %u\r\n", fun_args->random_seed);
}

function_pointer lookup_function(char *command_name) {
    if (strncmp(command_name, "write_ram_buffer", 17) == 0) {
        return &write_ram_buffer;
    } else if (strncmp(command_name, "read_ram_buffer", 16) == 0) {
        return &read_ram_buffer;
    } else if (strncmp(command_name, "write_eeprom", 13) == 0) {
        return &write_eeprom;
    } else if (strncmp(command_name, "read_eeprom", 12) == 0) {
        return &read_eeprom;
    } else if (strncmp(command_name, "read_labels_buffer", 19) == 0) {
        return &read_labels_buffer;
    } else if (strncmp(command_name, "compute_dist_matrix", 20) == 0) {
        return &compute_dist_matrix;
    } else if (strncmp(command_name, "read_dist_matrix", 17) == 0) {
        return &read_dist_matrix;
    } else if (strncmp(command_name, "rand_subset_selection", 22) == 0) {
       return &rand_subset_selection;
    } else if (strncmp(command_name, "set_random_seed", 16) == 0) {
       return &set_random_seed;
    } else {
        xprintf("ack_error: command_name not recognised\r\n");
        xprintf("command_name %s\r\n", command_name);
		exit(1);
    }
}