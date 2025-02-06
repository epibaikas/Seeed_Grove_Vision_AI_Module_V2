#ifndef PROTOCOL_FUNCTIONS_HOST_H
#define PROTOCOL_FUNCTIONS_HOST_H

    void write_eeprom(struct FunctionArguments *fun_args);
    void read_eeprom(struct FunctionArguments *fun_args);

    void compute_dist_matrix(struct FunctionArguments *fun_args);
    
#endif