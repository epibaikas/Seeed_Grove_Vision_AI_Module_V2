#include <stdlib.h>
#include "spi_eeprom_comm.h"
#include "arm_mve.h"
#include "xprintf.h"

#include "incr_learn.h"
#include "util_functions.h"
#include "protocol_functions.h"

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

void get_example_flash_addr(int example_num, int* flash_sector_num, uint32_t* flash_sector_start_addr, int* flash_sector_idx, struct FunctionArguments *fun_args) {
    *flash_sector_num = example_num / fun_args->examples_per_eeprom_sector;

    *flash_sector_start_addr = EEPROM_BASE_ADDRESS + (EEPROM_SECTOR_SIZE * (uint32_t)(*flash_sector_num));
    *flash_sector_idx = (example_num % fun_args->examples_per_eeprom_sector) * fun_args->bytes_per_example;
}

void update_labels_buffer(struct FunctionArguments *fun_args) {
    int example_num = 0;
    int flash_sector_num = 0;
    uint32_t flash_sector_start_addr = EEPROM_BASE_ADDRESS;
    int flash_sector_idx = 0;

    for (int i = 0; i < fun_args->num_examples_total; i++) {
        if (i < fun_args->ram_buffer_size) {
            fun_args->labels[i] = fun_args->ram_buffer[i][fun_args->bytes_per_example - 1];
        } else {
            example_num = i - fun_args->ram_buffer_size;
            get_example_flash_addr(example_num, &flash_sector_num, &flash_sector_start_addr, &flash_sector_idx, fun_args);

            // Read only the last byte that contains the label of the examples stored in EEPROM
            hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, flash_sector_start_addr + (uint32_t)flash_sector_idx + fun_args->bytes_per_example - 1, &(fun_args->eeprom_buffer[0]), 1);
            fun_args->labels[i] = fun_args->eeprom_buffer[0];
        }
    }
}

void copy_example_from_ram_to_eeprom(int ram_example_num, int eeprom_example_num, struct FunctionArguments *fun_args) {
    int flash_sector_num = 0;
    uint32_t flash_sector_start_addr = EEPROM_BASE_ADDRESS;
    int flash_sector_idx = 0;

    // Determine flash_sector_num based on eeprom_example_num
    get_example_flash_addr(eeprom_example_num, &flash_sector_num, &flash_sector_start_addr, &flash_sector_idx, fun_args);

    // Read contents from flash sector to eeprom_sector_buffer 
    hx_lib_spi_eeprom_4read(USE_DW_SPI_MST_Q, flash_sector_start_addr, &(fun_args->eeprom_sector_buffer[0]), EEPROM_SECTOR_SIZE);

    // Erase flash sector
    hx_lib_spi_eeprom_erase_sector(USE_DW_SPI_MST_Q, flash_sector_start_addr, FLASH_SECTOR);

    // Copy the contents of ram_buffer[i] to eeprom_sector_buffer
    memcpy(&(fun_args->eeprom_sector_buffer[flash_sector_idx]),  fun_args->ram_buffer[ram_example_num], fun_args->bytes_per_example);

    // Write data in eeprom_sector_buffer to flash
    hx_lib_spi_eeprom_write(USE_DW_SPI_MST_Q, flash_sector_start_addr, &(fun_args->eeprom_sector_buffer[0]), EEPROM_SECTOR_SIZE, 0);
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
        example_num = eeprom_indices_not_in_subset[i] - fun_args->ram_buffer_size; // Subtract fun_args->ram_buffer_size to change index range to [0, fun_args->eeprom_buffer_size - 1]
        copy_example_from_ram_to_eeprom(subset_idxs[i], example_num, fun_args);
    }

    free(eeprom_indices_not_in_subset);
}

TIMER_CFG_T setup_timer() {
    TIMER_CFG_T timer_cfg;
    timer_cfg.period = UINT32_MAX; // (ms)
    timer_cfg.mode = TIMER_MODE_PERIODICAL;
    timer_cfg.state = TIMER_STATE_DC;
    timer_cfg.ctrl = TIMER_CTRL_CPU;

    return timer_cfg;
}