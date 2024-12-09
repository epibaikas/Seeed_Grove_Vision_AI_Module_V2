#ifndef UTIL_FUNCTIONS_H
#define UTIL_FUNCTIONS_H

struct FunctionArguments;

uint16_t* allocate_symmetric_2D_array(uint32_t N);
void set_symmetric_2D_array_value(uint16_t *array, uint32_t N, uint32_t i, int32_t j, uint16_t value);
uint16_t get_symmetric_2D_array_value(uint16_t *array, uint32_t N, uint32_t i, uint32_t j);
uint32_t get_symmetric_2D_array_index(uint32_t N, uint32_t i, int32_t j);

uint32_t dot_prod_uint8_vect(uint8_t* pSrcA, uint8_t* pSrcB, uint32_t blockSize);

void shuffle(uint16_t *array, uint32_t size);
void get_random_subset(uint32_t M, uint32_t N, uint16_t* subset_idxs);
void get_random_bal_subset(uint8_t *labels, uint16_t* subset_idxs);

int compare_subset_indices(const void *a, const void *b);
int compare_indices(void *arr, const void *a, const void *b);
uint8_t predict_label(uint16_t *sorting_indices, uint8_t *labels, uint8_t k);
uint8_t find_max_index(uint8_t *array, size_t size);

void get_example_flash_addr(int example_num, int* flash_sector_num, uint32_t* flash_sector_start_addr, int* flash_sector_idx);

void write_buffer(uint8_t* buffer, uint32_t buffer_size, int num_per_line);
void read_buffer(void* buffer, uint32_t buffer_size, size_t element_size,  int num_per_line);
void update_labels_buffer(struct FunctionArguments *fun_args);
uint16_t* find_label_indices(uint8_t *labels, uint16_t labels_array_size, uint8_t target_label, uint32_t *target_label_count);
uint8_t get_num_of_available_classes(uint8_t *labels);
void classify_training_set(struct FunctionArguments *fun_args, uint16_t *subset_idxs, uint8_t* predicted_labels);
void move_subset_to_eeprom(uint16_t *subset_idxs, size_t subset_size, struct FunctionArguments *fun_args);
#endif