#include <stdio.h>
#include <stdlib.h>
#include "xprintf.h"
#include "incr_learn.h"
#include "util_functions.h"
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
        int j = rand() % (i + 1);

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

    for (int i = 0; i < NUM_OF_CLASSES; i++) {
        xprintf("[label %d: %u], ", i, label_counts[i]);
    }
    xprintf("-- predicted_label: %u", max_index);
    xprintf("\r\n");

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