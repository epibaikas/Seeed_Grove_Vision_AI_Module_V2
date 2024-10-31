#include <stdio.h>
#include <stdlib.h>
#include "xprintf.h"
#include "spi_eeprom_comm.h"
#include "incr_learn.h"
#include "util_functions.h"
#include "protocol_functions.h"
#include "arm_mve.h"

uint16_t* allocate_symmetric_2D_array(uint32_t N) {
    // Number of elements in the upper triangle of the symmetric matrix including diagonal
    uint32_t size = (N * (N + 1)) / 2;
    
    // Allocate memory for the upper triangle of the matrix as a 1D array
    uint16_t *array = calloc(size, sizeof(uint16_t));

    if (array == NULL) {
        xprintf("mem_error: memory allocation for symmetric matrix failed\r\n");
        exit(1);
    }

    return array;
}

void set_symmetric_2D_array_value(uint16_t *array, uint32_t N, uint32_t i, int32_t j, uint16_t value) {
    uint32_t index = get_symmetric_2D_array_index(N, i, j);
    array[index] = value;
}

uint16_t get_symmetric_2D_array_value(uint16_t *array, uint32_t N, uint32_t i, uint32_t j) {
    uint32_t index = get_symmetric_2D_array_index(N, i, j);
    uint16_t value = array[index];
    return value;
}

uint32_t get_symmetric_2D_array_index(uint32_t N, uint32_t i, int32_t j) {
        if (i > j) {
        // Because of symmetry: A[i][j] == A[j][i], so swap i and j
        uint32_t temp = i;
        i = j;
        j = temp;
    }

    // Compute the corresponding index in the 1D array
    uint32_t index = ((j * (j + 1)) / 2) + i;

    if (index >= (N * (N + 1) / 2)) {
        xprintf("index_error: index exceeds array size\r\n");
        exit(1);
    }


    return index;
}

uint32_t dot_prod_uint8_vect(uint8_t* pSrcA, uint8_t* pSrcB, uint32_t blockSize) {
    uint32_t result = 0;
    uint32_t num_of_whole_blocks = blockSize / 16;

    for (int i = 0; i < num_of_whole_blocks; i++) {
        uint8_t* pOne = &pSrcA[16*i];
        uint8_t* pTwo = &pSrcB[16*i];
        
        // Load the values from the array blocks
        uint8x16_t VectorOne = vld1q_u8(pOne);
        uint8x16_t VectorTwo = vld1q_u8(pTwo);

        result = vmladavaq_u8(result, VectorTwo, VectorOne);
    }

    return result;
}

// Function to shuffle array using Fisher-Yates algorithm
void shuffle(uint16_t *array, uint32_t size) {
    for (int i = size - 1; i > 0; i--) {

        // Generate a random index in range [0, i]
        int j = round( (rand() / (float)RAND_MAX) * i);

        // Swap array[i] with array[j]
        uint16_t temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

// Function to sample M random numbers from range [0, N-1] without replacement
void get_random_subset(uint32_t M, uint32_t N, uint16_t* subset_idxs) {
    // Create an array with numbers [0, N-1]
    uint16_t *idx_array = calloc(N, sizeof(uint16_t));
    if (idx_array == NULL) {
        xprintf("mem_error: memory allocation for idx_array failed\r\n");
		exit(1);
    }

    for (int i = 0; i < N; i++) {
        idx_array[i] = i;
    }

    // Shuffle the array
    shuffle(idx_array, N);

    // Set the first M elements of the shuffled idx_array as the random subset
    for (int i = 0; i < M; i++) {
        subset_idxs[i] = idx_array[i];
    }

    // Free the dynamically allocated memory
    free(idx_array);
}

// Comparator function for stable argsort
int compare_indices(void *arr, const void *a, const void *b) {
    uint16_t *array = (uint16_t *)arr;
    uint16_t idx1 = *(const uint16_t *)a;
    uint16_t idx2 = *(const uint16_t *)b;
    if (array[idx1] < array[idx2]) return -1;
    if (array[idx1] > array[idx2]) return 1;
    return idx1 - idx2;
}

uint8_t predict_label(uint16_t *sorting_indices, uint8_t *labels, uint8_t k) {
    uint8_t *label_counts = (uint8_t *)calloc(NUM_OF_CLASSES, sizeof(uint8_t));
    if (label_counts == NULL) {
        xprintf("mem_error: memory allocation for label_counts failed\r\n");
		exit(1);
    }
    
    for (int i = 0; i < k; i++) {
        uint8_t label = labels[sorting_indices[i]];
        label_counts[label]++;
    }

    uint8_t max_index = find_max_index(&label_counts[0], NUM_OF_CLASSES);

    // for (int i = 0; i < NUM_OF_CLASSES; i++) {
    //     xprintf("[label %d: %u], ", i, label_counts[i]);
    // }
    // xprintf("-- predicted_label: %u", max_index);
    // xprintf("\r\n");

    free(label_counts);
    return max_index;
}

// Function to find the index of the maximum value in an array
uint8_t find_max_index(uint8_t *array, size_t size) {
    uint8_t max_index = 0; // Initialize the maximum index to the first element
    for (size_t i = 1; i < size; i++) {
        if (array[i] > array[max_index]) {
            max_index = i; // Update the maximum index if a larger value is found
        }
    }

    return max_index;
}

void get_example_flash_addr(int example_num, int* flash_sector_num, uint32_t* flash_sector_start_addr, int* flash_sector_idx) {
    *flash_sector_num = example_num / EXAMPLES_PER_FLASH_SECTOR;

    *flash_sector_start_addr = FLASH_BASE_ADDRESS + (FLASH_SECTOR_SIZE * (uint32_t)(*flash_sector_num));
    *flash_sector_idx = (example_num % EXAMPLES_PER_FLASH_SECTOR) * BYTES_PER_IMG;
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
    int example_num = 0;
    int flash_sector_num = 0;
    uint32_t flash_sector_start_addr = FLASH_BASE_ADDRESS;
    int flash_sector_idx = 0;

    for (int i = 0; i < NUM_OF_IMGS_TOTAL; i++) {
        if (i < NUM_OF_IMGS_IN_RAM_BUFFER) {
            fun_args->labels[i] = fun_args->ram_buffer[i][BYTES_PER_IMG - 1];
        } else {
            example_num = i - NUM_OF_IMGS_IN_RAM_BUFFER;
            get_example_flash_addr(example_num, &flash_sector_num, &flash_sector_start_addr, &flash_sector_idx);

            // Read only the last byte that contains the label of the examples stored in EEPROM
            hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, flash_sector_start_addr + (uint32_t)flash_sector_idx + BYTES_PER_IMG - 1, &(fun_args->eeprom_buffer[0]), 1);
            fun_args->labels[i] = fun_args->eeprom_buffer[0];
        }
    }
}