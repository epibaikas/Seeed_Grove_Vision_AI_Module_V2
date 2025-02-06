#ifndef PROTOCOL_FUNCTIONS_H
#define PROTOCOL_FUNCTIONS_H

struct FunctionArguments {
  int seq_num;
  char *param;
  uint32_t ram_buffer_size;
  uint32_t eeprom_buffer_size;
  uint32_t num_examples_total;
  uint32_t bytes_per_example;
  uint32_t data_bytes_per_example;
  int examples_per_eeprom_sector;
  uint32_t num_of_classes;

  uint8_t **ram_buffer;
  uint8_t **eeprom_buffer_host;
  uint8_t *eeprom_buffer;
  uint8_t *eeprom_buffer_2;
  uint8_t *eeprom_sector_buffer;
  uint16_t *dist_matrix;
  uint8_t *labels;
  unsigned int random_seed;
};

#ifdef HOST_PLATFORM
  #include "platform/host/protocol_functions_host.h"
#elif defined(GROVE_VISION_WE2)
  #include "platform/grove_vision_we2/protocol_functions_grove_vision_we2.h"
#endif

typedef void (*function_pointer)(struct FunctionArguments *);

void write_ram_buffer(struct FunctionArguments *fun_args);
void read_ram_buffer(struct FunctionArguments *fun_args);
void read_labels_buffer(struct FunctionArguments *fun_args);

void read_dist_matrix(struct FunctionArguments *fun_args);
void rand_subset_selection(struct FunctionArguments *fun_args);
void rand_greedy_subset_selection(struct FunctionArguments *fun_args);

void set_data_buffer_parameters(struct FunctionArguments *fun_args);
void set_random_seed(struct FunctionArguments *fun_args);

function_pointer lookup_function(char *command_name);
#endif