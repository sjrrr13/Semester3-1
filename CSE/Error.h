//
// Created by Jiarui She on 2021/11/11.
//

#ifndef SMARTFILESYSTEM_ERROR_H
#define SMARTFILESYSTEM_ERROR_H

// 成功
#define SUCCESS 0
// 文件或目录不存在
#define NO_SUCH_FILE 1001
// 文件或目录已经存在
#define FILE_EXISTS 1002
// End of File
#define END_OF_FILE 1003
// 没有权限
#define NO_AUTH 1004
// 不合法的参数
#define INVALID_ARG 1005
// 未知错误
#define UNKNOWN 1006
// 磁盘上的 inode list 已经满了
#define NO_MORE_INODE 1007
// 磁盘上的某个 inode 不能再加入 block 了
#define INODE_FULL 1008
// 空的文件或者目录
#define EMPTY 1009
// 磁盘上没有空闲的 block 了
#define NO_MORE_BLOCK 1010
// IO Exception
#define IO_EXCEPTION 1011
// Link to a Directory
#define LINK_DIR 1012
// Write a directory
#define WR_DIR 1013

#include "iostream"

class Error {
private:
    int error_code;

public:
    Error();

    explicit Error(int);

    int get_code() const;

    static void handle(Error);
};


#endif //SMARTFILESYSTEM_ERROR_H
