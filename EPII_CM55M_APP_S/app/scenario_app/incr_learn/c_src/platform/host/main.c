#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <string.h>

#include "incr_learn.h"

/** main entry */
int main(void)
{
	int serial_fd;
    struct termios tty;

    // Open the serial port
    serial_fd = open(SERIAL_PORT, O_RDWR | O_NOCTTY);
    if (serial_fd == -1) {
        perror("Error opening serial port");
        return 1;
    }

    // Get current serial port settings
    if (tcgetattr(serial_fd, &tty) != 0) {
        perror("Error getting terminal attributes");
        close(serial_fd);
        return 1;
    }

    // Configure serial port
    cfsetospeed(&tty, BAUDRATE);
    cfsetispeed(&tty, BAUDRATE);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8; // 8-bit chars
    tty.c_cflag |= CLOCAL | CREAD;              // Enable receiver
    tty.c_cflag &= ~(PARENB | PARODD);          // No parity
    tty.c_cflag &= ~CSTOPB;                     // 1 stop bit
    tty.c_cflag &= ~CRTSCTS;                    // No hardware flow control

    tty.c_lflag = 0; // No canonical mode, no echo
    tty.c_iflag &= ~(IXON | IXOFF | IXANY); // Disable software flow control
    tty.c_oflag = 0; // Raw output

    // Apply the settings
    if (tcsetattr(serial_fd, TCSANOW, &tty) != 0) {
        perror("Error setting terminal attributes");
        close(serial_fd);
        return 1;
    }

    // Redirect stdin and stdout to serial port
    dup2(serial_fd, STDIN_FILENO);  // Redirect stdin
    dup2(serial_fd, STDOUT_FILENO); // Redirect stdout
    dup2(serial_fd, STDERR_FILENO); // Redirect stderr (optional)


	app_main();
	return 0;
}