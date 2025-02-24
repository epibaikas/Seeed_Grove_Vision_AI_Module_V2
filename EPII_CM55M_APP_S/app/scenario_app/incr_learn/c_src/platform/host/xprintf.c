#include <stdio.h>
#include <stdarg.h>
#include <string.h>

void xprintf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    fflush(stdout);
    va_end(args);
}

char* xgets(char *str, size_t size) {
    if (fgets(str, size, stdin) != NULL) {
        // Remove trailing characters from string including \r and \n characters
        size_t len = strlen(str);

        size_t end_of_line_idx = 0;

        if (len > 0) {
            for (size_t i = 0; i < size; i++) {
                if (str[i] == '\n' || str[i] ==  '\r') {
                    end_of_line_idx = i;
                    break;
                }
            }

            for (size_t i = end_of_line_idx; i < size; i++) {
                str[i] = '\0';
            }
        }
        return str;
    }
    return NULL;  // Handle EOF or error
}