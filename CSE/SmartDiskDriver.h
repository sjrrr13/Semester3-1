//
// Created by Jiarui She on 2021/11/12.
//

#ifndef SMARTFILESYSTEM_SMARTDISKDRIVER_H
#define SMARTFILESYSTEM_SMARTDISKDRIVER_H

#include "SmartDataStructure.h"
#include <ctime>

using namespace std;

class SmartDiskDriver {
private:
    static string smart_disk_path;
    static string smart_inode_path;
    static int smart_block_num;
    static int smart_block_size;
    static int smart_inode_num;
    static int smart_inode_size;

public:
    SmartDiskDriver();

    static void set_metadata(const string &, const string &);

    // block_num, block_size, inode_num, inode_size
    static Error format(int, int, int, int);

    static Error save_block(Block*);

    static Error load_block(Block*, int);

    static Error save_inode(Inode *);

    static Error load_inode(Inode *, int);

    static Error save_metadata();

    static Error load_metadata();

    static int getSmartBlockNum();

    static int getSmartBlockSize();

    static int getSmartInodeNum();

    static int getSmartInodeSize();
};

#endif //SMARTFILESYSTEM_SMARTDISKDRIVER_H
