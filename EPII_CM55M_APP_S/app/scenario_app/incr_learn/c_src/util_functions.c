#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

void get_random_bal_subset(uint8_t *labels, uint16_t* subset_idxs) {
    uint32_t target_label_count = 0;
    uint16_t *label_idxs;
    int idx = 0;

    // Get the same number of examples from every class
    uint8_t num_of_available_classes = get_num_of_available_classes(labels);
    int num_of_class_examples_in_subset = ceil(NUM_OF_IMGS_IN_EEPROM_BUFFER / (float) num_of_available_classes);
    
    // xprintf("num_of_available_classes: %u\n\r", num_of_available_classes);
    // xprintf("num_of_class_examples_in_subset: %d\n\r", num_of_class_examples_in_subset);

    for (uint8_t i = 0; i < NUM_OF_CLASSES; i++) {
        // Get the indices of the examples belonging to the target class
        label_idxs = find_label_indices(labels, NUM_OF_IMGS_TOTAL, i, &target_label_count);
         

         if (label_idxs != NULL) {
            // Shuffle the obtained indices
            shuffle(label_idxs, target_label_count);

            // Place the example indices to the subset
            // Check that you are not exceeding the size of the subset_idxs array
            for (int j = 0; (j < num_of_class_examples_in_subset) && (idx + j < NUM_OF_IMGS_IN_EEPROM_BUFFER); j++) {
                subset_idxs[idx + j] = label_idxs[j];
            }
            idx += num_of_class_examples_in_subset;
         }
    }

    // uint16_t *label_counts = (uint16_t *)calloc(NUM_OF_CLASSES, sizeof(uint16_t));
    // if (label_counts == NULL) {
    //     xprintf("mem_error: memory allocation for label_counts failed\r\n");
	// 	exit(1);
    // }
    
    // for (int i = 0; i < NUM_OF_IMGS_IN_EEPROM_BUFFER; i++) {
    //     label_counts[labels[subset_idxs[i]]]++;
    // }

    // for (int i = 0; i < NUM_OF_CLASSES; i++) {
    //     xprintf("label %d, count: %u\r\n", i, label_counts[i]);
    // }
    // free(label_counts);
}

// Comparator function for sorting subset indices in ascending order
int compare_subset_indices(const void *a, const void *b) {
    return (*(uint16_t*)a - *(uint16_t*)b);
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

uint16_t* find_label_indices(uint8_t *labels, uint16_t labels_array_size, uint8_t target_label, uint32_t *target_label_count) {
    // Initialize target_label_count to 0
    *target_label_count = 0;

    // First, count how many times the target label appears
    for (int i = 0; i < labels_array_size; i++) {
        if (labels[i] == target_label) {
            (*target_label_count)++;
        }
    }

    // If the target label is not found, return NULL
    if (*target_label_count == 0) {
        return NULL;
    }

    // Allocate memory to store the indices of the examples with the target label
    uint16_t* label_idxs = (uint16_t*)calloc(*target_label_count, sizeof(uint16_t));
    if (label_idxs == NULL) {
        xprintf("mem_error: memory allocation for label_idxs failed\r\n");
        exit(1);
    }

    // Collect the indices
    int idx = 0;
    for (int i = 0; i < labels_array_size; i++) {
        if (labels[i] == target_label) {
            label_idxs[idx++] = i;
        }
    }

    return label_idxs;
}

uint8_t get_num_of_available_classes(uint8_t *labels) {
    uint8_t num_of_available_classes = 0;
    
    uint16_t *label_counts = (uint16_t *)calloc(NUM_OF_CLASSES, sizeof(uint16_t));
    if (label_counts == NULL) {
        xprintf("mem_error: memory allocation for label_counts failed\r\n");
		exit(1);
    }
    
    for (int i = 0; i < NUM_OF_IMGS_TOTAL; i++) {
        label_counts[labels[i]]++;
    }
    
    for (int i = 0; i < NUM_OF_CLASSES; i++) {
        if (label_counts[i] > 0) {
            num_of_available_classes++;
        }
    }

    free(label_counts);
    return num_of_available_classes;
}


void classify_training_set(struct FunctionArguments *fun_args, uint16_t *subset_idxs, uint8_t* predicted_labels) {
    uint16_t* temp_dist_buf = calloc(NUM_OF_IMGS_IN_EEPROM_BUFFER, sizeof(uint16_t));
    uint16_t* indices = calloc(NUM_OF_IMGS_IN_EEPROM_BUFFER, sizeof(uint16_t));
    if (temp_dist_buf == NULL || indices == NULL) {
        xprintf("mem_error: memory allocation for temp_dist_buf or indices failed\r\n");
		exit(1);
    }


    for (int i = 0; i < NUM_OF_IMGS_TOTAL; i++) {
        // Load temporary buffer with the distances between the i-th examples and all the examples in the subset
        for (int j = 0; j < NUM_OF_IMGS_IN_EEPROM_BUFFER; j++) {
            temp_dist_buf[j] = get_symmetric_2D_array_value(fun_args->dist_matrix, NUM_OF_IMGS_TOTAL, i, subset_idxs[j]);
        }

        // Initialise indices buffer
        for (size_t i = 0; i < NUM_OF_IMGS_IN_EEPROM_BUFFER; i++) {
            indices[i] = i;
        }
        // Get the indices that short temp_dist_buf in ascending distance order
        qsort_r(indices, NUM_OF_IMGS_IN_EEPROM_BUFFER, sizeof(uint16_t), (void *) temp_dist_buf, compare_indices);

        // Convert the indices to the corresponding subset_idxs
        for (int j = 0; j < NUM_OF_IMGS_IN_EEPROM_BUFFER; j++) {
            indices[j] = subset_idxs[indices[j]];
        }

        // xprintf("Example %d, Nearest Neighbhours: [%u, %u, %u, %u, %u] \r\n", i, indices[0], indices[1], indices[2], indices[3], indices[4]);
        predicted_labels[i] = predict_label(indices, fun_args->labels, kNN_k);
    }

    free(temp_dist_buf);
    free(indices);
}

void move_subset_to_eeprom(uint16_t *subset_idxs, size_t subset_size, struct FunctionArguments *fun_args) {
    // Sort subset_idxs in ascending order
    qsort(subset_idxs, subset_size, sizeof(uint16_t), compare_subset_indices);

    // Get the indices of examples in eeprom that will be replaced by examples in RAM
    // Find the index of the first eeprom data example in sorted subset_idxs
    int first_eeprom_idx = 0;
    while (subset_idxs[first_eeprom_idx] < NUM_OF_IMGS_IN_RAM_BUFFER && first_eeprom_idx < subset_size) {
        first_eeprom_idx++;
    }

    // Find eeprom indices where data from RAM buffer will be placed.
    // These are the indices of eeprom exampels that are not in the subset
    uint16_t* eeprom_indices_not_in_subset  = calloc(first_eeprom_idx, sizeof(uint16_t));
    if (eeprom_indices_not_in_subset == NULL) {
        xprintf("mem_error: memory allocation for eeprom_indices_not_in_subset failed");
        exit(1);
    }

    int i = 0;
    int j = first_eeprom_idx;
    for (uint16_t idx = NUM_OF_IMGS_IN_RAM_BUFFER; idx < NUM_OF_IMGS_TOTAL; idx++) {
        if (idx == subset_idxs[j]) {
            j++;
        } else {
            // The index does not belong to the subset
            // Add it to eeprom_indices_not_in_subset
            eeprom_indices_not_in_subset[i] = idx;
            i++;
        }
    }

    int example_num = 0;
    int flash_sector_num = 0;
    uint32_t flash_sector_start_addr = FLASH_BASE_ADDRESS;
    int flash_sector_idx = 0;

    // Replace EEPROM examples that are not in the subset with examples from RAM
    for (int i = 0; i < first_eeprom_idx; i++) {
        example_num = eeprom_indices_not_in_subset[i] - NUM_OF_IMGS_IN_RAM_BUFFER; // Subtract NUM_OF_IMGS_IN_RAM_BUFFER to change index range to [0, NUM_OF_IMGS_IN_EEPROM - 1]

        // Determine flash_sector_num based on example_num
        get_example_flash_addr(example_num, &flash_sector_num, &flash_sector_start_addr, &flash_sector_idx);

        // Read contents from flash sector to eeprom_sector_buffer 
        hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, flash_sector_start_addr, &(fun_args->eeprom_sector_buffer[0]), FLASH_SECTOR_SIZE);

        // Erase flash sector
        hx_lib_spi_eeprom_erase_sector(USE_DW_SPI_MST_Q, flash_sector_start_addr, FLASH_SECTOR);

        // Copy the contents of ram_buffer[subset_idxs[i]] to eeprom_sector_buffer
        memcpy(&(fun_args->eeprom_sector_buffer[flash_sector_idx]),  fun_args->ram_buffer[subset_idxs[i]], BYTES_PER_IMG);

        // Write data in eeprom_sector_buffer to flash
        hx_lib_spi_eeprom_write(USE_DW_SPI_MST_Q, flash_sector_start_addr, &(fun_args->eeprom_sector_buffer[0]), FLASH_SECTOR_SIZE, 0);
    }

    free(eeprom_indices_not_in_subset);
}