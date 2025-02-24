#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "xprintf.h"
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
    write_buffer(&(fun_args->ram_buffer[example_num][0]), fun_args->bytes_per_example, num_per_line);
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

    read_buffer(&(fun_args->ram_buffer[example_num][0]), fun_args->bytes_per_example, sizeof(uint8_t), num_per_line);
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
    read_buffer(&(fun_args->labels[0]), fun_args->num_examples_total, sizeof(uint8_t), num_per_line);
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

    uint32_t N = fun_args->num_examples_total;
    uint32_t size = (N * (N + 1)) / 2;

    read_buffer(fun_args->dist_matrix, size, sizeof(uint16_t), num_per_line);
}

void rand_subset_selection(struct FunctionArguments *fun_args) {
    int balanced_subset = 0;
    int num_per_line = 8;
    int sscanf_ret_value = 0;

    sscanf_ret_value = sscanf(fun_args->param, "%d %d",  &balanced_subset, &num_per_line);
    if (sscanf_ret_value <= 0) {
        xprintf("ack_error: rand_subset_selection() parameters not parsed correctly\r\n");
        exit(1);
    }
    
    xprintf("ack_begin %d\r\n", fun_args->seq_num);

    // Generate subset
    uint16_t* subset_idxs = calloc(fun_args->eeprom_buffer_size, sizeof(uint16_t));
    uint8_t* predicted_labels = calloc(fun_args->num_examples_total, sizeof(uint8_t));
    if (subset_idxs == NULL || predicted_labels == NULL) {
        xprintf("mem_error: memory allocation for subset_idxs or predicted_labels failed\r\n");
		exit(1);
    }

    // Update the labels buffer
    update_labels_buffer(fun_args);
    
    if (balanced_subset == 1) {
        get_random_bal_subset(fun_args->labels, subset_idxs, fun_args);
    } else {
        get_random_subset(fun_args->eeprom_buffer_size, fun_args->num_examples_total, subset_idxs);
    }

    // Classify all examples using the subset
    classify_training_set(fun_args, subset_idxs, predicted_labels);

    // Output generated subset and label predictions
    read_buffer(subset_idxs, fun_args->eeprom_buffer_size, sizeof(uint16_t), num_per_line);
    xprintf("subset_idxs_read_done\r\n");
    read_buffer(predicted_labels, fun_args->num_examples_total, sizeof(uint8_t), num_per_line);
    xprintf("predicted_labels_read_done\r\n");

    // Move subset data examples located in RAM to EEPROM
    move_subset_to_eeprom(subset_idxs, fun_args->eeprom_buffer_size, fun_args);

    free(subset_idxs);
    free(predicted_labels);
}

void rand_greedy_subset_selection(struct FunctionArguments *fun_args) {
    int num_iter = 1; 
    int num_per_line = 8;
    int sscanf_ret_value = 0;

    float max_avg_class_acc = 0.0;
    float avg_class_acc = 0.0;
    char max_avg_class_acc_str[10];
    char avg_class_acc_str[10];

    float mutation_rate = 0.2;

    sscanf_ret_value = sscanf(fun_args->param, "%d %d", &num_iter, &num_per_line);
    if (sscanf_ret_value <= 0) {
        xprintf("ack_error: rand_subset_selection() parameters not parsed correctly\r\n");
        exit(1);
    }
    
    xprintf("ack_begin %d\r\n", fun_args->seq_num);

    // Allocate memory
    uint16_t* candidate_subset_idxs = calloc(fun_args->eeprom_buffer_size, sizeof(uint16_t));
    uint16_t* subset_idxs = calloc(fun_args->eeprom_buffer_size, sizeof(uint16_t));
    uint8_t* predicted_labels = calloc(fun_args->num_examples_total, sizeof(uint8_t));
    float* optim_func_buffer = calloc(num_iter, sizeof(float));
    if (candidate_subset_idxs == NULL || subset_idxs == NULL || predicted_labels == NULL || optim_func_buffer == NULL) {
        xprintf("mem_error: memory allocation for candidate_subset_idxs, subset_idxs, predicted_labels or optim_func_buffer failed\r\n");
		exit(1);
    }

    // Update the labels buffer
    update_labels_buffer(fun_args);

    // Generate initial random balanced subset
    get_random_bal_subset(fun_args->labels, subset_idxs, fun_args);

    // Classify all examples using the subset
    classify_training_set(fun_args, subset_idxs, predicted_labels);

    // uint32_t num_correct = get_num_correct_pred(fun_args->labels, predicted_labels);
    // xprintf("num_correct = %u\r\n", num_correct);

    max_avg_class_acc = get_avg_class_acc(fun_args->labels, predicted_labels, fun_args);
    float_to_string(max_avg_class_acc, max_avg_class_acc_str, 4);

    for (int i = 0; i < num_iter; i++) {
        memcpy(candidate_subset_idxs, subset_idxs, fun_args->eeprom_buffer_size * sizeof(uint16_t));
        mutate_bal_subset(candidate_subset_idxs, fun_args->labels, mutation_rate, fun_args);

        classify_training_set(fun_args, candidate_subset_idxs, predicted_labels);

        avg_class_acc = get_avg_class_acc(fun_args->labels, predicted_labels, fun_args);
        float_to_string(avg_class_acc, avg_class_acc_str, 4);

        xprintf("iter = %d, avg_class_acc = %s, max_avg_class_acc = %s \r\n", i, avg_class_acc_str, max_avg_class_acc_str);

        if (avg_class_acc > max_avg_class_acc) {
            memcpy(subset_idxs, candidate_subset_idxs, fun_args->eeprom_buffer_size * sizeof(uint16_t));
            max_avg_class_acc = avg_class_acc;
            float_to_string(max_avg_class_acc, max_avg_class_acc_str, 4);
        }

        optim_func_buffer[i] = max_avg_class_acc;
    }

    // Get label predictions using the latest subset
    classify_training_set(fun_args, subset_idxs, predicted_labels);

    // Output generated subset, label predictions and max_avg_class_acc_buffer
    read_buffer(subset_idxs, fun_args->eeprom_buffer_size, sizeof(uint16_t), num_per_line);
    xprintf("subset_idxs_read_done\r\n");
    read_buffer(predicted_labels, fun_args->num_examples_total, sizeof(uint8_t), num_per_line);
    xprintf("predicted_labels_read_done\r\n");
    read_buffer(optim_func_buffer, num_iter, sizeof(float), num_per_line);
    xprintf("optim_func_buffer_read_done\r\n");

    // Move subset data examples located in RAM to EEPROM
    move_subset_to_eeprom(subset_idxs, fun_args->eeprom_buffer_size, fun_args);

    free(subset_idxs);
    free(candidate_subset_idxs);
    free(predicted_labels);
}

void evo_subset_selection(struct FunctionArguments *fun_args) {
    int num_gen = 1; 
    int num_per_line = 8;
    int sscanf_ret_value = 0;

    int population_size = 100;
    int num_parents = 20;

    float mutation_rate = 0.1;
    int keep_elite = 1;

    float best_fitness = 0.0;

    char fitness_str[10];

    sscanf_ret_value = sscanf(fun_args->param, "%d %d", &num_gen, &num_per_line);
    if (sscanf_ret_value <= 0) {
        xprintf("ack_error: evo_subset_selection() parameters not parsed correctly\r\n");
        exit(1);
    }

    xprintf("ack_begin %d\r\n", fun_args->seq_num);

    int break_gen = num_gen;

    // Allocate memory
    uint16_t **population = allocate_2D_array(population_size, fun_args->eeprom_buffer_size, "population");
    uint16_t **parents = allocate_2D_array(num_parents, fun_args->eeprom_buffer_size, "parents");

    float *fitness = calloc(population_size, sizeof(float));
    uint8_t* max_fitness_idxs = calloc(population_size, sizeof(uint8_t));
    uint8_t* predicted_labels = calloc(fun_args->num_examples_total, sizeof(uint8_t));
    float* optim_func_buffer = calloc(num_gen, sizeof(float));

    uint16_t* subset_idxs = calloc(fun_args->eeprom_buffer_size, sizeof(uint16_t));

    if (fitness == NULL || max_fitness_idxs == NULL || predicted_labels == NULL || optim_func_buffer == NULL || subset_idxs == NULL) {
        xprintf("mem_error: memory allocation for fitness, max_fitness_idxs, predicted_labels, optim_func_buffer or subset_idxs failed\r\n");
        exit(1);
    }


    // Initialise max_fitness_idxs buffer
    for (int i = 0; i < population_size; i++) {
        max_fitness_idxs[i] = i;
    }

    // Update the labels buffer
    update_labels_buffer(fun_args);

    // Generate initial population and compute fitness scores
    for (int i = 0; i < population_size; i++) {
        get_random_bal_subset(fun_args->labels, population[i], fun_args);

        classify_training_set(fun_args, population[i], predicted_labels);
        fitness[i] = get_avg_class_acc(fun_args->labels, predicted_labels, fun_args);
    }

    // Get the indices that sort fitness scores in descending order
    qsort_r(max_fitness_idxs, population_size, sizeof(uint8_t), (void *) fitness, compare_indices_float_array);

    best_fitness = fitness[max_fitness_idxs[0]];
    memcpy(subset_idxs, population[max_fitness_idxs[0]], fun_args->eeprom_buffer_size * sizeof(uint16_t));
    optim_func_buffer[0] = best_fitness;
    float_to_string(best_fitness, fitness_str, 4);
    xprintf("Gen 0, max fitness score: %s\r\n", fitness_str);

    for (int n = 1; n < num_gen; n++) {
        // Parent selection
        steady_state_parent_selection(population, population_size, parents, num_parents, max_fitness_idxs, fun_args);

        for (int i = 0; i < num_parents; i++) {
            // Sort the subset_idxs in a parent chromosome first in ascending order and then in ascending class label order
            // The purpose of the double sorting is to avoid duplicates when combining chromosomes with single-point crossover
            qsort(parents[i], fun_args->eeprom_buffer_size, sizeof(uint16_t), compare_subset_indices);
            qsort_r(parents[i], fun_args->eeprom_buffer_size, sizeof(uint16_t), fun_args->labels , compare_indices_uint8);
        }

        // Place the elite solutions directly to the new population
        for (int i = 0; i < keep_elite; i++) {  
            memcpy(population[i], parents[i], fun_args->eeprom_buffer_size * sizeof(uint16_t));
            fitness[i] = fitness[max_fitness_idxs[i]];
        }

        // Generate the rest of the offsprings
        for (int i = keep_elite; i < population_size; i++) {
            // Apply single-point crossover
            single_point_crossover(parents[(i - keep_elite) % num_parents], parents[(i - keep_elite + 1) % num_parents], population[i], fun_args);

            // Mutate generated offspring
            mutate_bal_subset(population[i], fun_args->labels, mutation_rate, fun_args);

            // Compute offspring's fitness score
            classify_training_set(fun_args, population[i], predicted_labels);
            fitness[i] = get_avg_class_acc(fun_args->labels, predicted_labels, fun_args);
        }

        // Get the indices that sort fitness scores in descending order
        qsort_r(max_fitness_idxs, population_size, sizeof(uint8_t), (void *) fitness, compare_indices_float_array);

        if (fitness[max_fitness_idxs[0]] > best_fitness) {
            best_fitness = fitness[max_fitness_idxs[0]];
            memcpy(subset_idxs, population[max_fitness_idxs[0]], fun_args->eeprom_buffer_size * sizeof(uint16_t));
            float_to_string(best_fitness, fitness_str, 4);
            xprintf("Gen %d, max fitness score: %s, new max\r\n", n, fitness_str);
        } else {
            xprintf("Gen %d, max fitness score: %s\r\n", n, fitness_str);
        }
        optim_func_buffer[n] = best_fitness;

        if (best_fitness >= 1.0) {
            break_gen = n + 1;
            break;
        } 
    }

    // Dummy loop for printing remaining new lines 
    for (int i = break_gen; i < num_gen; i++) {
        xprintf("\r\n");
    }

    // Get label predictions using the latest subset
    classify_training_set(fun_args, subset_idxs, predicted_labels);

    // Output generated subset, label predictions and max_avg_class_acc_buffer
    read_buffer(subset_idxs, fun_args->eeprom_buffer_size, sizeof(uint16_t), num_per_line);
    xprintf("subset_idxs_read_done\r\n");
    read_buffer(predicted_labels, fun_args->num_examples_total, sizeof(uint8_t), num_per_line);
    xprintf("predicted_labels_read_done\r\n");
    read_buffer(optim_func_buffer, num_gen, sizeof(float), num_per_line);
    xprintf("optim_func_buffer_read_done\r\n");

    // Move subset data examples located in RAM to EEPROM
    move_subset_to_eeprom(subset_idxs, fun_args->eeprom_buffer_size, fun_args);


    free_2D_array(population, population_size);
    free_2D_array(parents, num_parents);
    free(fitness);
    free(max_fitness_idxs);
    free(predicted_labels);
    free(optim_func_buffer);
}

void set_data_buffer_parameters(struct FunctionArguments *fun_args) {
    uint32_t ram_buffer_size;
    uint32_t eeprom_buffer_size;
    uint32_t bytes_per_example;
    uint32_t num_of_classes;
    int sscanf_ret_value = 0;

    sscanf_ret_value = sscanf(fun_args->param, "%u %u %u %u", &ram_buffer_size, &eeprom_buffer_size, &bytes_per_example, &num_of_classes);
    if (sscanf_ret_value <= 0) {
        xprintf("ack_error: set_data_buffer_parameters() parameters not parsed correctly\r\n");
        exit(1);
    }

    xprintf("ack_begin %d\r\n", fun_args->seq_num);

    fun_args->ram_buffer_size = ram_buffer_size;
    fun_args->eeprom_buffer_size = eeprom_buffer_size;
    fun_args->num_examples_total = ram_buffer_size + eeprom_buffer_size;
    fun_args->bytes_per_example = bytes_per_example;
    fun_args->data_bytes_per_example = bytes_per_example - 1;
    fun_args->examples_per_eeprom_sector = EEPROM_SECTOR_SIZE / bytes_per_example;
    fun_args->num_of_classes = num_of_classes;

    // Allocate memory for RAM buffer
	uint8_t **ram_buffer = (uint8_t **)calloc(fun_args->ram_buffer_size, sizeof(uint8_t *));
	if (ram_buffer == NULL) {
		xprintf("mem_error: memory allocation for ram_buffer failed\r\n");
		exit(1);
	}

	for (int i = 0; i < fun_args->ram_buffer_size; i++) {
		ram_buffer[i] = (uint8_t *)calloc(fun_args->bytes_per_example, sizeof(uint8_t));
		if (ram_buffer[i] == NULL) {
			xprintf("mem_error: memory allocation for ram_buffer[%d] failed\r\n", i);
			exit(1);
		}
	}

    #ifdef HOST_PLATFORM
        // Allocate memory for EEPROM buffer
        uint8_t **eeprom_buffer_host = (uint8_t **)calloc(fun_args->eeprom_buffer_size, sizeof(uint8_t *));
        if (eeprom_buffer_host == NULL) {
            xprintf("mem_error: memory allocation for eeprom_buffer_host failed\r\n");
            exit(1);
        }

        for (int i = 0; i < fun_args->eeprom_buffer_size; i++) {
            eeprom_buffer_host[i] = (uint8_t *)calloc(fun_args->bytes_per_example, sizeof(uint8_t));
            if (eeprom_buffer_host[i] == NULL) {
                xprintf("mem_error: memory allocation for eeprom_buffer_host[%d] failed\r\n", i);
                exit(1);
            }
        }

        fun_args->eeprom_buffer_host = eeprom_buffer_host;
    #endif

	// Allocate memory for distance matrix
	uint16_t* dist_matrix = allocate_symmetric_2D_array(fun_args->num_examples_total);

	// Create eeprom buffers;
	uint8_t* labels = (uint8_t *)calloc(fun_args->num_examples_total, sizeof(uint8_t));
    
    if (labels == NULL) {
		xprintf("mem_error: memory allocation for labels buffer failed\r\n");
		exit(1);
	}

	fun_args->ram_buffer = ram_buffer;
	fun_args->dist_matrix = dist_matrix;
	fun_args->labels = labels;

    xprintf("ram_buffer_size: %u\r\n", fun_args->ram_buffer_size);
    xprintf("eeprom_buffer_size: %u\r\n", fun_args->eeprom_buffer_size);
    xprintf("num_examples_total: %u\r\n", fun_args->num_examples_total);
    xprintf("bytes_per_example: %u\r\n", fun_args->bytes_per_example);
    xprintf("data_bytes_per_example: %u\r\n", fun_args->data_bytes_per_example);
    xprintf("examples_per_eeprom_sector: %u\r\n", fun_args->examples_per_eeprom_sector);
    xprintf("num_of_classes: %u\r\n", fun_args->num_of_classes);

	xprintf("Addr of dist_matrix: 0x%08x\r\n", fun_args->dist_matrix);
	xprintf("Addr of eeprom_buffer: 0x%08x\r\n", fun_args->eeprom_buffer);
	xprintf("Addr of eeprom_buffer_2: 0x%08x\r\n", fun_args->eeprom_buffer_2);
	xprintf("Addr of eeprom_sector_buffer: 0x%08x\r\n", fun_args->eeprom_sector_buffer);
	xprintf("Addr of labels buffer: 0x%08x\r\n", fun_args->labels);
	xprintf("RAND_MAX: 0x%08x\r\n", RAND_MAX);    
}

void set_random_seed(struct FunctionArguments *fun_args) {
    unsigned int random_seed = 1;
    int sscanf_ret_value = 0;

    sscanf_ret_value = sscanf(fun_args->param, "%u", &random_seed);
    if (sscanf_ret_value <= 0) {
        xprintf("ack_error: set_random_seed() parameters not parsed correctly\r\n");
        exit(1);
    }
    
    xprintf("ack_begin %d\r\n", fun_args->seq_num);
    srand(random_seed);

    fun_args->random_seed = random_seed;

    xprintf("random seed set to: %u\r\n", fun_args->random_seed);
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
    } else if (strncmp(command_name, "rand_greedy_subset_selection", 29) == 0) {
        return &rand_greedy_subset_selection;
    } else if (strncmp(command_name, "evo_subset_selection", 21) == 0) {
        return &evo_subset_selection;
    } else if (strncmp(command_name, "set_random_seed", 16) == 0) {
        return &set_random_seed;
    } else if (strncmp(command_name, "set_data_buffer_parameters", 27) == 0) {
        return &set_data_buffer_parameters;
    } else {
        xprintf("ack_error: command_name not recognised\r\n");
        xprintf("command_name %s\r\n", command_name);
		exit(1);
    }
}