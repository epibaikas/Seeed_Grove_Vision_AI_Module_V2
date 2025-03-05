#ifndef UTIL_FUNCTIONS_HOST_H
#define UTIL_FUNCTIONS_HOST_H

uint32_t dot_prod_uint8_vect(uint8_t* pSrcA, uint8_t* pSrcB, uint32_t blockSize);

void update_labels_buffer(struct FunctionArguments *fun_args);
void copy_example_from_ram_to_eeprom(int ram_example_num, int eeprom_example_num, struct FunctionArguments *fun_args);
void move_subset_to_eeprom(uint16_t *subset_idxs, size_t subset_size, struct FunctionArguments *fun_args);

#endif