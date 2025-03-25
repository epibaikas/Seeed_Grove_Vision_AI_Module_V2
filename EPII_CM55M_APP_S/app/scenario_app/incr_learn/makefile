CC = gcc

INCDIR = inc
INCDIR += inc/platform/host

CFLAGS = -g -Wall
CFLAGS += -DHOST_PLATFORM 
CFLAGS += -D_GNU_SOURCE
CFLAGS += $(addprefix -I, $(INCDIR))

BUILD_DIR = build
TARGET = incr_learn

CSRC_DIR = c_src
SRC = $(wildcard $(CSRC_DIR)/*.c) $(wildcard $(CSRC_DIR)/platform/host/*.c)

# Convert .c files to .o object files
OBJ = $(patsubst %.c, $(BUILD_DIR)/%.o, $(SRC))

# Build the executable
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $^ -o $(BUILD_DIR)/$@ -lm

# Rule to compile all object files
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Ensure the build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

show:
	echo $(SRC)
	echo $(OBJ)

run:
	./$(BUILD_DIR)/$(TARGET)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)