uint16_t* allocate_symmetric_2D_array(uint32_t N);
void set_symmetric_2D_array_value(uint16_t *array, uint32_t N, uint32_t i, int32_t j, uint16_t value);
uint16_t get_symmetric_2D_array_value(uint16_t *array, uint32_t N, uint32_t i, uint32_t j);
uint32_t get_symmetric_2D_array_index(uint32_t N, uint32_t i, int32_t j);

uint32_t dot_prod_uint8_vect(uint8_t* pSrcA, uint8_t* pSrcB, uint32_t blockSize);

void shuffle(uint16_t *array, uint32_t size);
void get_random_subset(uint32_t M, uint32_t N, uint16_t* subset_idxs);

int compare_indices(void *arr, const void *a, const void *b);
uint8_t predict_label(uint16_t *sorting_indices, uint8_t *labels, uint8_t k);
uint8_t find_max_index(uint8_t *array, size_t size);