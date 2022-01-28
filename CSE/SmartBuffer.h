//
// Created by Jiarui She on 2021/11/11.
//

#ifndef SMARTFILESYSTEM_SMARTBUFFER_H
#define SMARTFILESYSTEM_SMARTBUFFER_H

// 模拟内存中最多可以缓存多少个block和inode
#define BLOCK_NUM 32
#define INODE_NUM 10

#include <string>
#include "Error.h"
#include "SmartDataStructure.h"
#include "SmartDiskDriver.h"

using namespace std;

class SmartBuffer {
private:
    // 分别是Block的缓存和Inode的缓存，因为Inode存储为.inode文件
    Block *block_buffer = new Block[BLOCK_NUM];
    Inode *inode_buffer = new Inode[INODE_NUM];
    int block_head = 0;
    int block_tail = 0;
    int inode_head = 0;
    int inode_tail = 0;

    int change_block();

    int change_inode();

public:
    SmartBuffer();

    ~SmartBuffer();

    // 初始化buffer
    Error initial(int, int, int, int);

    // 根据id获取block
    Block *get_block(int);

    // 根据id获取inode
    Inode *get_inode(int);

    // 得到一个新的block，加入到buffer中
    int get_new_block();

    // 创建一个新的inode，加入到buffer中
    int get_new_inode(int);

    // 分配一个block给一个inode
    int allocate_block(int);

    // buffer的信息，供测试
    void info();

    // 获取inode的一个没有被写满的块
    int get_inode_free_block(int);

    // 将buffer的内容写回disk中
    Error write_back();

    // 重启file system时初始化buffer
    Error reboot();
};


#endif //SMARTFILESYSTEM_SMARTBUFFER_H
