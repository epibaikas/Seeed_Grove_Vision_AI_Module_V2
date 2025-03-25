#ifndef UTIL_FUNCTIONS_H
#define UTIL_FUNCTIONS_H

struct FunctionArguments;

#ifdef HOST_PLATFORM
  #include "platform/host/util_functions_host.h"
#elif defined(GROVE_VISION_WE2)
  #include "platform/grove_vision_we2/util_functions_grove_vision_we2.h"
#endif

uint16_t** allocate_2D_array(uint32_t dim_0, uint32_t dim_1, char* array_name);
void free_2D_array(uint16_t** array, uint32_t dim_0);
uint16_t* allocate_symmetric_2D_array(uint32_t N);
void set_symmetric_2D_array_value(uint16_t *array, uint32_t N, uint32_t i, uint32_t j, uint16_t value);
uint16_t get_symmetric_2D_array_value(uint16_t *array, uint32_t N, uint32_t i, uint32_t j);
uint32_t get_symmetric_2D_array_index(uint32_t N, uint32_t i, uint32_t j);

void shuffle(uint16_t *array, uint32_t size);
void get_random_subset(uint32_t M, uint32_t N, uint16_t* subset_idxs);
void get_random_bal_subset(uint8_t *labels, uint16_t* subset_idxs, struct FunctionArguments *fun_args);

int compare_subset_indices(const void *a, const void *b);

#ifdef _GNU_SOURCE
int compare_indices(const void *a, const void *b, void *arr);
int compare_indices_uint8(const void *a, const void *b, void *arr);
int compare_indices_float_array(const void *a, const void *b, void *arr);
#else
int compare_indices(void *arr, const void *a, const void *b);
int compare_indices_uint8(void *arr, const void *a, const void *b);
int compare_indices_float_array(void *arr, const void *a, const void *b);
#endif

uint8_t predict_label(uint16_t *sorting_indices, uint8_t *labels, uint8_t k, struct FunctionArguments *fun_args);
uint8_t find_max_index(uint8_t *array, size_t size);

void write_buffer(uint8_t* buffer, uint32_t buffer_size, int num_per_line);
void read_buffer(void* buffer, uint32_t buffer_size, size_t element_size,  int num_per_line);
uint16_t* find_label_indices(uint8_t *labels, uint16_t labels_array_size, uint8_t target_label, uint32_t *target_label_count);
uint8_t get_num_of_available_classes(uint8_t *labels, struct FunctionArguments *fun_args);
void classify_training_set(struct FunctionArguments *fun_args, uint16_t *subset_idxs, uint8_t* predicted_labels);
float get_avg_class_acc(uint8_t *labels, uint8_t *predicted_labels, struct FunctionArguments *fun_args);
uint32_t get_num_correct_pred(uint8_t *labels, uint8_t *predicted_labels, struct FunctionArguments *fun_args);

void steady_state_parent_selection(uint16_t** population, uint32_t population_size, uint16_t** parents, uint32_t num_parents, uint8_t* max_fitness_idxs, struct FunctionArguments *fun_args);
void single_point_crossover(uint16_t* par_1, uint16_t* par_2, uint16_t* offspring, struct  FunctionArguments *fun_args);
void mutate_bal_subset(uint16_t* subset_idxs, uint8_t *labels, float mutation_rate, struct FunctionArguments *fun_args);

void float_to_string(float num, char *str, int precision);
#endif