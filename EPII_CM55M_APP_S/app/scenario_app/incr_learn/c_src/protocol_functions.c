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
    int sscanf_ret_value = 0;
    int num_per_line = 8;
    uint32_t flash_address = FLASH_BASE_ADDRESS;

    sscanf_ret_value = sscanf(fun_args->param, "0x%08x %d", &flash_address, &num_per_line);
    if (sscanf_ret_value <= 1) {
        xprintf("ack_error: write_eeprom() parameters not parsed correctly\r\n");
        exit(1);
    }
    if (flash_address < FLASH_BASE_ADDRESS) {
        xprintf("ack_error: cannot write to flash_address < 0x00201000\r\n");
        exit(1);
    }

    xprintf("ack_begin %d\r\n", fun_args->seq_num);
    write_buffer(&(fun_args->eeprom_buffer[0]), BYTES_PER_IMG, num_per_line);

    // Write data in eeprom_buffer to flash
    hx_lib_spi_eeprom_write(USE_DW_SPI_MST_Q, flash_address, &(fun_args->eeprom_buffer[0]), BYTES_PER_IMG, 0);
}

void read_eeprom(struct FunctionArguments *fun_args) {
    int sscanf_ret_value = 0;
    uint32_t flash_address = FLASH_BASE_ADDRESS;
    int num_per_line = 8;

    sscanf_ret_value = sscanf(fun_args->param, "0x%08x %d", &flash_address, &num_per_line);
    if (sscanf_ret_value <= 1) {
        xprintf("ack_error: read_eeprom() parameters not parsed correctly\r\n");
        exit(1);
    }

    xprintf("ack_begin %d\r\n", fun_args->seq_num);

    // Read data to eeprom_buffer from flash
    hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, flash_address, &(fun_args->eeprom_buffer[0]), BYTES_PER_IMG);

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

    uint32_t base_flash_address = FLASH_BASE_ADDRESS;
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
            hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, base_flash_address + (i - NUM_OF_IMGS_IN_RAM_BUFFER) * (BYTES_PER_IMG), &(fun_args->eeprom_buffer[0]), DATA_BYTES_PER_IMG);
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
                hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, base_flash_address + (i - NUM_OF_IMGS_IN_RAM_BUFFER) * (BYTES_PER_IMG), &(fun_args->eeprom_buffer[0]), DATA_BYTES_PER_IMG);
                dist -= 2 * dot_prod_uint8_vect(&(fun_args->eeprom_buffer[0]), fun_args->ram_buffer[j], DATA_BYTES_PER_IMG);
                // xprintf("Cond 2 ");
            }

            if (i < NUM_OF_IMGS_IN_RAM_BUFFER && j >= NUM_OF_IMGS_IN_RAM_BUFFER) {
                hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, base_flash_address + (j - NUM_OF_IMGS_IN_RAM_BUFFER) * (BYTES_PER_IMG), &(fun_args->eeprom_buffer[0]), DATA_BYTES_PER_IMG);
                dist -= 2 * dot_prod_uint8_vect(fun_args->ram_buffer[i], &(fun_args->eeprom_buffer[0]), DATA_BYTES_PER_IMG);
                // xprintf("Cond 3 ");
            }

            if (i >= NUM_OF_IMGS_IN_RAM_BUFFER && j >= NUM_OF_IMGS_IN_RAM_BUFFER) {
                hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, base_flash_address + (i - NUM_OF_IMGS_IN_RAM_BUFFER) * (BYTES_PER_IMG), &(fun_args->eeprom_buffer[0]), DATA_BYTES_PER_IMG);
                hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, base_flash_address + (j - NUM_OF_IMGS_IN_RAM_BUFFER) * (BYTES_PER_IMG), &(fun_args->eeprom_buffer_2[0]), DATA_BYTES_PER_IMG);
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
    int num_per_line = 8;
    int sscanf_ret_value = 0;

    sscanf_ret_value = sscanf(fun_args->param, "%d", &num_per_line);
    if (sscanf_ret_value <= 0) {
        xprintf("ack_error: rand_subset_selection() parameters not parsed correctly\r\n");
        exit(1);
    }
    
    xprintf("ack_begin %d\r\n", fun_args->seq_num);

    // Generate random subset
    uint16_t* subset_idxs = calloc(NUM_OF_IMGS_IN_EEPROM_BUFFER, sizeof(uint16_t));
    uint16_t* temp_dist_buf = calloc(NUM_OF_IMGS_IN_EEPROM_BUFFER, sizeof(uint16_t));
    uint16_t* indices = calloc(NUM_OF_IMGS_IN_EEPROM_BUFFER, sizeof(uint16_t));
    uint8_t* predicted_labels = calloc(NUM_OF_IMGS_TOTAL, sizeof(uint8_t));
    if (subset_idxs == NULL || temp_dist_buf == NULL || indices == NULL) {
        xprintf("mem_error: memory allocation for subset_idxs, temp_dist_buf, indices or predicted_labels failed\r\n");
		exit(1);
    }

    get_random_subset(NUM_OF_IMGS_IN_EEPROM_BUFFER, NUM_OF_IMGS_TOTAL, subset_idxs);

    // Update the labels buffer
    update_labels_buffer(fun_args);

    // Classify all examples using the subset
    for (int i = 0; i < NUM_OF_IMGS_TOTAL; i++) {
        // Load temporary buffer with the distances between the i-th examples and all the examples in the subset
        for (int j = 0; j < NUM_OF_IMGS_IN_EEPROM_BUFFER; j++) {
            temp_dist_buf[j] = get_symmetric_2D_array_value(fun_args->dist_matrix, NUM_OF_IMGS_TOTAL, i, subset_idxs[j]);
        }

        for (size_t i = 0; i < NUM_OF_IMGS_IN_EEPROM_BUFFER; i++) {
            indices[i] = i;
        }
        // Get the indices that short temp_dist_buf in ascending distance order
        qsort_r(indices, NUM_OF_IMGS_IN_EEPROM_BUFFER, sizeof(uint16_t), (void *) temp_dist_buf, compare_indices);

        // Convert the indices to the corresponding subset_idxs
        for (int j = 0; j < NUM_OF_IMGS_IN_EEPROM_BUFFER; j++) {
            indices[j] = subset_idxs[indices[j]];
        }

        xprintf("Example %d, Nearest Neibhours: [%u, %u, %u, %u, %u] ", i, indices[0], indices[1], indices[2], indices[3], indices[4]);
        predicted_labels[i] = predict_label(indices, fun_args->labels, kNN_k);
    }

    // Output generated subset and label predictions
    read_buffer(subset_idxs, NUM_OF_IMGS_IN_EEPROM_BUFFER, sizeof(uint16_t), num_per_line);
    xprintf("subset_idxs_read_done\r\n");
    read_buffer(predicted_labels, NUM_OF_IMGS_TOTAL, sizeof(uint8_t), num_per_line);
    xprintf("predicted_labels_read_done\r\n");

    free(subset_idxs);
    free(temp_dist_buf);
    free(indices);
    free(predicted_labels);
}

void write_buffer(uint8_t* buffer, uint32_t buffer_size, int num_per_line) {
    char line_buf[LINE_BUFFER_LEN];
    char *data;
    int offset;
    uint32_t num;
    int byte_idx;

    for (int i = 0; i < buffer_size; i += num_per_line) {
        xgets(line_buf, LINE_BUFFER_LEN);

        byte_idx = 0;
        data = line_buf;
        while (sscanf(data, "%u%n", &num, &offset) == 1 && byte_idx < num_per_line) {
            buffer[i + byte_idx] = num;
            data += offset;
            byte_idx++;
            xprintf("%3u ", num);
        }
        xprintf("\r\n");

        // Check that the correct number of bytes has been parsed from the line
        if ((buffer_size - i) / num_per_line >= 1) {
            if (byte_idx != num_per_line) {
                xprintf("ack_error: data line not parsed correctly\r\n");
                exit(1);
            }
        } else if (byte_idx != buffer_size % num_per_line) {
            xprintf("ack_error: last data line not parsed correctly\r\n");
            exit(1);
        }
    }
}

void read_buffer(void* buffer, uint32_t buffer_size, size_t element_size, int num_per_line) {
    char line_buf[LINE_BUFFER_LEN];
    char idx_str[12];
    char ack_str[10];

    uint8_t *buf8;
    uint16_t *buf16;
    
    if (element_size == sizeof(uint8_t)) {
        buf8 = (uint8_t *)buffer;
    } else if (element_size == sizeof(uint16_t)) {
        buf16 = (uint16_t *)buffer;
    }

    for (int i = 0; i < buffer_size; i++) {
        if (element_size == sizeof(uint8_t)) {
            xprintf("%3u ", buf8[i]);
        } else if (element_size == sizeof(uint16_t)) {
            xprintf("%5u ", buf16[i]);
        }
        if ((i+1) % num_per_line == 0 || i+1 == buffer_size) {
            xprintf("\r\n");
            xgets(line_buf, LINE_BUFFER_LEN);

            strcpy(ack_str, "ack ");
            sprintf(idx_str, "%d", i+1);
            strcat(ack_str, idx_str);
            if (strcmp(line_buf, ack_str) != 0) {
                xprintf("ack_error: read acknowledge not properly received\r\n");
                xprintf("line_buf: %s, ack_str: %s\r\n", line_buf, ack_str);
                exit(1);
            }
        }
    }
}

void update_labels_buffer(struct FunctionArguments *fun_args) {
    for (int i = 0; i < NUM_OF_IMGS_TOTAL; i++) {
        if (i < NUM_OF_IMGS_IN_RAM_BUFFER) {
            fun_args->labels[i] = fun_args->ram_buffer[i][BYTES_PER_IMG - 1];
        } else {
            // Read only the last byte that contains the label of the examples stored in EEPROM
            hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, FLASH_BASE_ADDRESS + (i - NUM_OF_IMGS_IN_RAM_BUFFER) * (BYTES_PER_IMG) + BYTES_PER_IMG - 1, &(fun_args->eeprom_buffer[0]), 1);
            fun_args->labels[i] = fun_args->eeprom_buffer[0];
        }
    }
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
    } else {
        xprintf("ack_error: command_name not recognised\r\n");
        xprintf("command_name %s\r\n", command_name);
		exit(1);
    }
}