//
// Created by Jiarui She on 2021/11/11.
//

#include "Error.h"

using namespace std;

Error::Error() {
    error_code = 0;
}

Error::Error(int error) {
    error_code = error;
}

int Error::get_code() const {
    return error_code;
}

void Error::handle(Error e) {
    int error = e.get_code();
    switch (error) {
        case SUCCESS:
            cout << "Operation success!" << endl;
            break;
        case NO_SUCH_FILE:
            cout << "No such file or directory" << endl;
            break;
        case FILE_EXISTS:
            cout << "File or directory already exits" << endl;
            break;
        case END_OF_FILE:
            cout << "End of file" << endl;
            break;
        case NO_AUTH:
            cout << "No authority" << endl;
            break;
        case INVALID_ARG:
            cout << "Invalid argument" << endl;
            break;
        case UNKNOWN:
            cout << "Unknown Error" << endl;
            break;
        case NO_MORE_INODE:
            cout << "No more inode in the system" << endl;
            break;
        case INODE_FULL:
            cout << "Inode is full" << endl;
            break;
        case EMPTY:
            cout << "File or directory is empty" << endl;
            break;
        case NO_MORE_BLOCK:
            cout << "No more block in disk" << endl;
            break;
        case IO_EXCEPTION:
            cout << "I/O exception" << endl;
            break;
        case LINK_DIR:
            cout << "Hard Link to a directory" << endl;
            break;
        case WR_DIR:
            cout << "Cannot write a directory" << endl;
            break;
        default:
            break;
    }
}