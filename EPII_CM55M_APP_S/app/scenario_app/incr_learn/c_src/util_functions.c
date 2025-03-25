#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "xprintf.h"
#include "incr_learn.h"
#include "util_functions.h"
#include "protocol_functions.h"

uint16_t** allocate_2D_array(uint32_t dim_0, uint32_t dim_1, char* array_name) {
    uint16_t **array = (uint16_t **)calloc(dim_0, sizeof(uint16_t *));
	if (array == NULL) {
		xprintf("mem_error: memory allocation for %s failed\r\n", array_name);
		exit(1);
	}

	for (int i = 0; i < dim_0; i++) {
		array[i] = (uint16_t *)calloc(dim_1, sizeof(uint16_t));
		if (array[i] == NULL) {
			xprintf("mem_error: memory allocation for %s[%d] failed\r\n", array_name, i);
			exit(1);
		}
	}

    return array;
}

void free_2D_array(uint16_t** array, uint32_t dim_0) {
    if (array == NULL) return;
    
    // Free every row
    for (int i = 0; i < dim_0; i++) {
        free(array[i]);
    }

    // Free the array of pointers
    free(array);
}

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

void set_symmetric_2D_array_value(uint16_t *array, uint32_t N, uint32_t i, uint32_t j, uint16_t value) {
    uint32_t index = get_symmetric_2D_array_index(N, i, j);
    array[index] = value;
}

uint16_t get_symmetric_2D_array_value(uint16_t *array, uint32_t N, uint32_t i, uint32_t j) {
    uint32_t index = get_symmetric_2D_array_index(N, i, j);
    uint16_t value = array[index];
    return value;
}

uint32_t get_symmetric_2D_array_index(uint32_t N, uint32_t i, uint32_t j) {
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

void get_random_bal_subset(uint8_t *labels, uint16_t* subset_idxs, struct FunctionArguments *fun_args) {
    uint32_t target_label_count = 0;
    uint16_t *label_idxs;
    int sum = 0;
    int num_of_class_examples_to_be_retained;

    // Get the same number of examples from every class
    uint8_t num_of_available_classes = get_num_of_available_classes(labels, fun_args);    
    // xprintf("num_of_available_classes: %u\n\r", num_of_available_classes);

    for (uint8_t i = 0; i < fun_args->num_of_classes; i++) {
        // Get the indices of the examples belonging to the target class
        label_idxs = find_label_indices(labels, fun_args->num_examples_total, i, &target_label_count);
         
        num_of_class_examples_to_be_retained = ceil((fun_args->eeprom_buffer_size - sum) / (float) num_of_available_classes);

        if (label_idxs != NULL) {
            // Shuffle the obtained indices
            shuffle(label_idxs, target_label_count);

            // Place the example indices to the subset
            for (int j = 0; j < num_of_class_examples_to_be_retained; j++) {
                subset_idxs[sum + j] = label_idxs[j];
            }
            sum += num_of_class_examples_to_be_retained;
            num_of_available_classes -= 1;
        }

        free(label_idxs);
    }

    // uint16_t *label_counts = (uint16_t *)calloc(fun_args->num_of_classes, sizeof(uint16_t));
    // if (label_counts == NULL) {
    //     xprintf("mem_error: memory allocation for label_counts failed\r\n");
	// 	exit(1);
    // }
    
    // for (int i = 0; i < fun_args->eeprom_buffer_size; i++) {
    //     label_counts[labels[subset_idxs[i]]]++;
    // }

    // for (int i = 0; i < fun_args->num_of_classes; i++) {
    //     xprintf("label %d, count: %u\r\n", i, label_counts[i]);
    // }
    // free(label_counts);
}

// Comparator function for sorting subset indices in ascending order
int compare_subset_indices(const void *a, const void *b) {
    return (*(uint16_t*)a - *(uint16_t*)b);
}

#ifdef _GNU_SOURCE
    // Comparator function for stable argsort
    int compare_indices(const void *a, const void *b, void *arr) {
        uint16_t *array = (uint16_t *)arr;
        uint16_t idx1 = *(const uint16_t *)a;
        uint16_t idx2 = *(const uint16_t *)b;
        if (array[idx1] < array[idx2]) return -1;
        if (array[idx1] > array[idx2]) return 1;
        return idx1 - idx2;
    }

    // Comparator function for stable argsort - uint8_t - ascending
    int compare_indices_uint8(const void *a, const void *b, void *arr) {
        uint8_t *array = (uint8_t *)arr;
        uint16_t idx1 = *(const uint16_t *)a;
        uint16_t idx2 = *(const uint16_t *)b;
        if (array[idx1] < array[idx2]) return -1;
        if (array[idx1] > array[idx2]) return 1;
        return idx1 - idx2;
    }

    // Comparator function for stable argsort - float - descending
    int compare_indices_float_array(const void *a, const void *b, void *arr) {
        float *array = (float *)arr;
        uint8_t idx1 = *(const uint8_t *)a;
        uint8_t idx2 = *(const uint8_t *)b;
        if (array[idx1] < array[idx2]) return 1;
        if (array[idx1] > array[idx2]) return -1;
        return idx1 - idx2;
    }
#else
    // Comparator function for stable argsort
    int compare_indices(void *arr, const void *a, const void *b) {
        uint16_t *array = (uint16_t *)arr;
        uint16_t idx1 = *(const uint16_t *)a;
        uint16_t idx2 = *(const uint16_t *)b;
        if (array[idx1] < array[idx2]) return -1;
        if (array[idx1] > array[idx2]) return 1;
        return idx1 - idx2;
    }

    // Comparator function for stable argsort - uint8_t - ascending
    int compare_indices_uint8(void *arr, const void *a, const void *b) {
        uint8_t *array = (uint8_t *)arr;
        uint16_t idx1 = *(const uint16_t *)a;
        uint16_t idx2 = *(const uint16_t *)b;
        if (array[idx1] < array[idx2]) return -1;
        if (array[idx1] > array[idx2]) return 1;
        return idx1 - idx2;
    }

    // Comparator function for stable argsort - float - descending
    int compare_indices_float_array(void *arr, const void *a, const void *b) {
        float *array = (float *)arr;
        uint8_t idx1 = *(const uint8_t *)a;
        uint8_t idx2 = *(const uint8_t *)b;
        if (array[idx1] < array[idx2]) return 1;
        if (array[idx1] > array[idx2]) return -1;
        return idx1 - idx2;
    }
#endif

uint8_t predict_label(uint16_t *sorting_indices, uint8_t *labels, uint8_t k, struct FunctionArguments *fun_args) {
    uint8_t *label_counts = (uint8_t *)calloc(fun_args->num_of_classes, sizeof(uint8_t));
    if (label_counts == NULL) {
        xprintf("mem_error: memory allocation for label_counts failed\r\n");
		exit(1);
    }
    
    for (int i = 0; i < k; i++) {
        uint8_t label = labels[sorting_indices[i]];
        label_counts[label]++;
    }

    uint8_t max_index = find_max_index(&label_counts[0], fun_args->num_of_classes);

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
    char idx_str[20];
    char ack_str[20];

    uint8_t *buf8;
    uint16_t *buf16;
    float *buf_float;
    char float_str[10];

    size_t len = 0;

    if (element_size == sizeof(uint8_t)) {
        buf8 = (uint8_t *)buffer;
    } else if (element_size == sizeof(uint16_t)) {
        buf16 = (uint16_t *)buffer;
    } else if (element_size == sizeof(float)) {
        buf_float = (float *)buffer;
    }

    for (int i = 0; i < buffer_size; i++) {
        if (element_size == sizeof(uint8_t)) {
            xprintf("%3u ", buf8[i]);
        } else if (element_size == sizeof(uint16_t)) {
            xprintf("%5u ", buf16[i]);
        } else if (element_size == sizeof(float)) {
            float_to_string(buf_float[i], float_str, 4);
            xprintf("%6s ", float_str);
        }

        if ((i+1) % num_per_line == 0 || i+1 == buffer_size) {
            xprintf("\r\n");
            xgets(line_buf, LINE_BUFFER_LEN);

            len = strlen(line_buf);

            strcpy(ack_str, "ack ");
            sprintf(idx_str, "%d", i+1);
            strcat(ack_str, idx_str);
            if (strncmp(line_buf, ack_str, len) != 0) {
                xprintf("ack_error: read acknowledge not properly received\r\n");
                xprintf("line_buf: %s, ack_str: %s\r\n", line_buf, ack_str);
                exit(1);
            }
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

uint8_t get_num_of_available_classes(uint8_t *labels, struct FunctionArguments *fun_args) {
    uint8_t num_of_available_classes = 0;
    
    uint16_t *label_counts = (uint16_t *)calloc(fun_args->num_of_classes, sizeof(uint16_t));
    if (label_counts == NULL) {
        xprintf("mem_error: memory allocation for label_counts failed\r\n");
		exit(1);
    }
    
    for (int i = 0; i < fun_args->num_examples_total; i++) {
        label_counts[labels[i]]++;
    }
    
    for (int i = 0; i < fun_args->num_of_classes; i++) {
        if (label_counts[i] > 0) {
            num_of_available_classes++;
        }
    }

    free(label_counts);
    return num_of_available_classes;
}


void classify_training_set(struct FunctionArguments *fun_args, uint16_t *subset_idxs, uint8_t* predicted_labels) {
    uint16_t* temp_dist_buf = calloc(fun_args->eeprom_buffer_size, sizeof(uint16_t));
    uint16_t* indices = calloc(fun_args->eeprom_buffer_size, sizeof(uint16_t));
    if (temp_dist_buf == NULL || indices == NULL) {
        xprintf("mem_error: memory allocation for temp_dist_buf or indices failed\r\n");
		exit(1);
    }


    for (int i = 0; i < fun_args->num_examples_total; i++) {
        // Load temporary buffer with the distances between the i-th examples and all the examples in the subset
        for (int j = 0; j < fun_args->eeprom_buffer_size; j++) {
            temp_dist_buf[j] = get_symmetric_2D_array_value(fun_args->dist_matrix, fun_args->max_num_examples, i, subset_idxs[j]);
        }

        // Initialise indices buffer
        for (size_t i = 0; i < fun_args->eeprom_buffer_size; i++) {
            indices[i] = i;
        }
        // Get the indices that sort temp_dist_buf in ascending distance order
        #ifdef _GNU_SOURCE
            qsort_r(indices, fun_args->eeprom_buffer_size, sizeof(uint16_t), compare_indices, (void *) temp_dist_buf);
        #else
            qsort_r(indices, fun_args->eeprom_buffer_size, sizeof(uint16_t), (void *) temp_dist_buf, compare_indices);
        #endif

        // Convert the indices to the corresponding subset_idxs
        for (int j = 0; j < fun_args->eeprom_buffer_size; j++) {
            indices[j] = subset_idxs[indices[j]];
        }

        // xprintf("Example %d, Nearest Neighbhours: [%u, %u, %u, %u, %u] \r\n", i, indices[0], indices[1], indices[2], indices[3], indices[4]);
        predicted_labels[i] = predict_label(indices, fun_args->labels, kNN_k, fun_args);
    }

    free(temp_dist_buf);
    free(indices);
}

float get_avg_class_acc(uint8_t *labels, uint8_t *predict_labels, struct FunctionArguments *fun_args) {
    uint16_t *label_idxs;
    uint32_t target_label_count = 0;
    uint32_t num_correct = 0;

    uint8_t num_of_available_classes = 0;
    float acc_sum = 0.0;
    float avg_acc = 0.0;

    for (uint8_t i = 0; i < fun_args->num_of_classes; i++) {
        // Get the indices of the examples belonging to the target class
        label_idxs = find_label_indices(labels, fun_args->num_examples_total, i, &target_label_count);
        
        num_correct = 0;
        for (uint32_t j = 0; j < target_label_count; j++) {
            if (predict_labels[label_idxs[j]] == labels[label_idxs[j]]) {
                num_correct++;
            }
        }
        
        if (target_label_count > 0) {
            num_of_available_classes++;
            acc_sum += num_correct / (float) target_label_count;
        }
        
        free(label_idxs);
    }

    avg_acc = acc_sum / (float) num_of_available_classes;
    return avg_acc;
}

uint32_t get_num_correct_pred(uint8_t *labels, uint8_t *predicted_labels, struct FunctionArguments *fun_args) {
    uint32_t num_correct = 0;
    for (int i = 0; i < fun_args->num_examples_total; i++) {
        if (labels[i] == predicted_labels[i]) {
            num_correct++;
        }
    }

    return num_correct;
}

void steady_state_parent_selection(uint16_t** population, uint32_t population_size, uint16_t** parents, uint32_t num_parents, uint8_t* max_fitness_idxs, struct FunctionArguments *fun_args) {
    for (int i = 0; i < num_parents; i++) {
        memcpy(parents[i], population[max_fitness_idxs[i]], fun_args->eeprom_buffer_size * sizeof(uint16_t));
    }
}

void single_point_crossover(uint16_t* par_1, uint16_t* par_2, uint16_t* offspring, struct  FunctionArguments *fun_args) {
    int crossover_idx = rand() % fun_args->eeprom_buffer_size;

    if  (par_1[crossover_idx] < par_2[crossover_idx]) {
        memcpy(offspring, par_1, crossover_idx * sizeof(uint16_t));
        memcpy(&offspring[crossover_idx], &par_2[crossover_idx], (fun_args->eeprom_buffer_size - crossover_idx) * sizeof(uint16_t));
    } else {
        memcpy(offspring, par_2, crossover_idx * sizeof(uint16_t));
        memcpy(&offspring[crossover_idx], &par_1[crossover_idx], (fun_args->eeprom_buffer_size - crossover_idx) * sizeof(uint16_t));
    }

    // xprintf("Crossover index: %d\r\n", crossover_idx);
    // xprintf("par1: ");
    // for (int i = 0; i < fun_args->eeprom_buffer_size; i++) {
    //     xprintf("%u ", par_1[i]);
    // }
    // xprintf("\r\n");

    // xprintf("par2: ");
    // for (int i = 0; i < fun_args->eeprom_buffer_size; i++) {
    //     xprintf("%u ", par_2[i]);
    // }
    // xprintf("\r\n");

    // xprintf("offs: ");
    // for (int i = 0; i < fun_args->eeprom_buffer_size; i++) {
    //     xprintf("%u ", offspring[i]);
    // }
    // xprintf("\r\n");
}

void mutate_bal_subset(uint16_t* subset_idxs, uint8_t *labels, float mutation_rate, struct FunctionArguments *fun_args) {    
    int idx = 0;
    uint32_t num_of_idxs_to_be_mutated = floor(mutation_rate * fun_args->eeprom_buffer_size);
    uint32_t num_of_available_idxs = fun_args->num_examples_total - fun_args->eeprom_buffer_size;

    if (num_of_idxs_to_be_mutated > num_of_available_idxs) {
        xprintf("mutation error: num_of_idxs_to_be_mutated is greater than available indices");
        exit(1);
    }

    uint8_t* in_subset = calloc(fun_args->num_examples_total, sizeof(uint8_t));
    uint16_t* idxs_not_in_subset = calloc(num_of_available_idxs, sizeof(uint16_t));
    uint8_t* idxs_mutated = calloc(fun_args->eeprom_buffer_size, sizeof(uint8_t));

    if (in_subset == NULL || idxs_not_in_subset == NULL || idxs_mutated == NULL) {
        xprintf("mem_error: memory allocation for in_subset, idxs_not_in_subset or idxs_mutated failed\r\n");
		exit(1);
    }
    
    shuffle(subset_idxs, fun_args->eeprom_buffer_size);

    // Turn subset_idxs into 0-1 bitstream format
    for (int i = 0; i < fun_args->eeprom_buffer_size; i++) {
        in_subset[subset_idxs[i]] = 1;
    }

    // Get the example idxs that are not in the subset
    for (int i = 0; i < fun_args->num_examples_total; i++) {
        if (in_subset[i] == 0) {
            idxs_not_in_subset[idx] = i;
            idx++;
        }
    }
    
    shuffle(idxs_not_in_subset, num_of_available_idxs);
    
    uint8_t label = 0;
    bool mutation_complete = false;
    // Apply mutation
    for (int i = 0; i < num_of_idxs_to_be_mutated; i++) {
      // Get label of i-th example from idxs_not_in_subset
      label = labels[idxs_not_in_subset[i]];

      idx = 0;
      mutation_complete = false;
      // Find the first idx within subset_idxs that has the same label and hasn't already been mutated 
      while (mutation_complete == false && idx < fun_args->eeprom_buffer_size) {
        if (label == labels[subset_idxs[idx]] && idxs_mutated[idx] != 1) {
          subset_idxs[idx] = idxs_not_in_subset[i];
          idxs_mutated[idx] = 1;
          mutation_complete = true;
        }
        
        idx++;
      }

    }

    // xprintf("mutated offs: ");
    // for (int i = 0; i < fun_args->eeprom_buffer_size; i++) {
    //     xprintf("%u ", subset_idxs[i]);
    // }
    // xprintf("\r\n\r\n");

    free(in_subset);
    free(idxs_not_in_subset);
    free(idxs_mutated);
}

void float_to_string(float num, char *str, int precision) {
    // Handle negative numbers
    int is_negative = 0;
    if (num < 0) {
        is_negative = 1;
        num = -num;
    }

    // Extract integer part
    int int_part = (int)num;
    float fraction_part = num - int_part;

    // Scale fractional part to desired precision
    for (int i = 0; i < precision; i++) {
        fraction_part *= 10;
    }
    int fractional_int = (int)(fraction_part + 0.5); // Round to nearest integer

    // Convert integer part to string
    char int_str[20], frac_str[20];
    sprintf(int_str, "%d", int_part);

    // Convert fractional part to string
    sprintf(frac_str, "%0*d", precision, fractional_int);

    // Combine integer and fractional parts
    if (is_negative) {
        sprintf(str, "-%s.%s", int_str, frac_str);
    } else {
        sprintf(str, "%s.%s", int_str, frac_str);
    }
}