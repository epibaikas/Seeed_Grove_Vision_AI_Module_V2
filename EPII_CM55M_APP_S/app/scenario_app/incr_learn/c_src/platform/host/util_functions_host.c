#include <stdlib.h>
#include <string.h>
#include "xprintf.h"

#include "util_functions.h"
#include "protocol_functions.h"

uint32_t dot_prod_uint8_vect(uint8_t* pSrcA, uint8_t* pSrcB, uint32_t blockSize) {
    uint32_t result = 0;

    for (int i = 0; i < blockSize; i++) {
        result += pSrcA[i] * pSrcB[i];
    }

    return result;
}

void update_labels_buffer(struct FunctionArguments *fun_args) {
    int example_num = 0;

    for (int i = 0; i < fun_args->num_examples_total; i++) {
        if (i < fun_args->ram_buffer_size) {
            fun_args->labels[i] = fun_args->ram_buffer[i][fun_args->bytes_per_example - 1];
        } else {
            example_num = i - fun_args->ram_buffer_size;
            fun_args->labels[i] = fun_args->eeprom_buffer_host[example_num][fun_args->bytes_per_example - 1];
        }
    }
}

void copy_example_from_ram_to_eeprom(int ram_example_num, int eeprom_example_num, struct FunctionArguments *fun_args) {
    memcpy(fun_args->eeprom_buffer_host[eeprom_example_num], fun_args->ram_buffer[ram_example_num], fun_args->bytes_per_example * sizeof(uint8_t));
}

void move_subset_to_eeprom(uint16_t *subset_idxs, size_t subset_size, struct FunctionArguments *fun_args) {
    // Sort subset_idxs in ascending order
    qsort(subset_idxs, subset_size, sizeof(uint16_t), compare_subset_indices);

    // Get the indices of examples in eeprom that will be replaced by examples in RAM
    // Find the index of the first eeprom data example in sorted subset_idxs
    int first_eeprom_idx = 0;
    while (subset_idxs[first_eeprom_idx] < fun_args->ram_buffer_size && first_eeprom_idx < subset_size) {
        first_eeprom_idx++;
    }

    // Find eeprom indices where data from RAM buffer will be placed.
    // These are the indices of eeprom examples that are not in the subset
    uint16_t* eeprom_indices_not_in_subset  = calloc(first_eeprom_idx, sizeof(uint16_t));
    if (eeprom_indices_not_in_subset == NULL) {
        xprintf("mem_error: memory allocation for eeprom_indices_not_in_subset failed");
        exit(1);
    }

    int i = 0;
    int j = first_eeprom_idx;
    for (uint16_t idx = fun_args->ram_buffer_size; idx < fun_args->max_num_examples; idx++) {
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
    // Replace EEPROM examples that are not in the subset with examples from RAM
    for (int i = 0; i < first_eeprom_idx; i++) {
        example_num = eeprom_indices_not_in_subset[i] - fun_args->ram_buffer_size; // Subtract fun_args->ram_buffer_size to change index range to [0, NUM_OF_IMGS_IN_EEPROM - 1]

        // Copy the contents of ram_buffer[subset_idxs[i]] to eeprom_buffer_host
        memcpy(fun_args->eeprom_buffer_host[example_num],  fun_args->ram_buffer[subset_idxs[i]], fun_args->bytes_per_example);
    }

    free(eeprom_indices_not_in_subset);
}