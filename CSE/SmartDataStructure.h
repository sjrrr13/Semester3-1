//
// Created by Jiarui She on 2021/11/11.
//

#ifndef SMARTFILESYSTEM_SMARTDATASTRUCTURE_H
#define SMARTFILESYSTEM_SMARTDATASTRUCTURE_H

#define DIR 1
#define OTHER 2

#include <string>
#include <iostream>
#include <fstream>
#include "Error.h"
#include "Tool.h"

using namespace std;

struct DirRecord {
    char name[12]; // File or Dir name
    int id; // Inode ID
};

class Block {
private:
    static int block_size;
    // 存储目录时，每个block可以存储的记录数是有限的
    static int record_limit;
    int block_id;
    // 存储文件内容时的偏移量
    int offset;
    // 当前block已经存储了多少条目录记录
    int record_num;
    char *data;

public:
    Block();

    ~Block();

    // 设置block size
    static void set_metadata(int);

    // 初始化data
    void initial(int);

    void set_id(int);

    int get_id() const;

    // 设置block的数据
    int set_data(const string &);

    // 只会在load block时用到
    int set_char_data(const char *);

    char *get_data() const;

    // 判断能否在添加目录信息
    bool can_add_record();

    // block存储目录的信息
    void set_record(DirRecord *);

    static int get_limit();

    static int get_block_size();

    int get_offset() const;

    int get_record_num();
//    // 将该block的内容写到disk上
//    Error save();
//
//    Error load();
};

class SuperBlock : public Block {
private:
    static int capacity;
    static int block_size;
    static int inode_num;
    static int inode_size;
    static int next_inode;

public:
    SuperBlock() : Block() {}

    // block_num, block_size, inode_num, inode_size
    static void set_metadata(int, int, int, int);

    static int get_next();

    static void set_next(int);

    static string serialize();

    static int getCapacity();

    static int getBlockSize();

    static int getInodeNum();

    static int getInodeSize();
};

class Bitmap : public Block {
private:
    static int block_num;
    static string usage;

public:
    Bitmap();

    static void set_metadata(int);

    static void set_used(int);

    static int get_free_block();

    static string serialize();

    static void set_usage(const string&);
};


class Inode {
    static int inode_size;
    int inode_id;
    int *block_array;
    int ptr; // 下一个block的下标
    // 文件的一些元数据
    int file_size;
    int file_type;
    time_t create;
    time_t last_access;
    time_t last_modify;

public:
    Inode();

    // Inode大小，每个Inode最多可以有几个block
    static void set_metadata(int);

    // 最多可以有几个block
    static int get_limit();

    void initial(int);

    int get_id() const;

    int *get_blocks() const;

    void set_file_size(int);

    void set_type(int);

    int add_block(int);

    void set_create(time_t);

    void set_access(time_t);

    void set_modify(time_t);

    string serialize();

    int get_type() const;

    int get_file_size() const;

    string get_create() const;

    string get_access() const;

    string get_modify() const;

//
//    Error save();
//
//    Error load();
};

#endif //SMARTFILESYSTEM_SMARTDATASTRUCTURE_H
