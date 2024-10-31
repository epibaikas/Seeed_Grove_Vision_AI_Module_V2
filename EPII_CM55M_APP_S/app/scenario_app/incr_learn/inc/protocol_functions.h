#ifndef PROTOCOL_FUNCTIONS_H
#define PROTOCOL_FUNCTIONS_H

struct FunctionArguments {
  int seq_num;
  char *param;
  uint8_t **ram_buffer;
  uint8_t *eeprom_buffer;
  uint8_t *eeprom_buffer_2;
  uint8_t *eeprom_sector_buffer;
  uint16_t *dist_matrix;
  uint8_t *labels;
  unsigned int random_seed;
};

typedef void (*function_pointer)(struct FunctionArguments *);

void write_ram_buffer(struct FunctionArguments *fun_args);
void read_ram_buffer(struct FunctionArguments *fun_args);
void write_eeprom(struct FunctionArguments *fun_args);
void read_eeprom(struct FunctionArguments *fun_args);
void read_labels_buffer(struct FunctionArguments *fun_args);

void compute_dist_matrix(struct FunctionArguments *fun_args);
void read_dist_matrix(struct FunctionArguments *fun_args);
void rand_subset_selection(struct FunctionArguments *fun_args);

void set_random_seed(struct FunctionArguments *fun_args);

function_pointer lookup_function(char *command_name);
#endif