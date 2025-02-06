#include <stdio.h>
#include <stdarg.h>
#include <string.h>

void xprintf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

char *xgets(char *str, size_t size) {
    if (fgets(str, size, stdin) != NULL) {
        // Remove trailing newline (if present)
        size_t len = strlen(str);
        if (len > 0 && str[len - 1] == '\n') {
            str[len - 1] = '\0';
        }
        return str;
    }
    return NULL;  // Handle EOF or error
}