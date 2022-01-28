//
// Created by Jiarui She on 2021/11/11.
//

#ifndef SMARTFILESYSTEM_SMARTFS_H
#define SMARTFILESYSTEM_SMARTFS_H

#define O_RDONLY 1
#define O_WRONLY 2
#define O_RDWR 3

#include <iostream>
#include <string>
#include <vector>
#include "SmartDiskDriver.h"
#include "SmartBuffer.h"
#include "Tool.h"
#include "Error.h"

using namespace std;

class FDTableEntry {
private:
    int file_descriptor;
    int file_table_index;

public:
    FDTableEntry();

    FDTableEntry(int, int);

    int get_fd() const;

    int get_index() const;
};


class FDTable {
private:
    vector<FDTableEntry> fd_table;
    int next_fd;

public:
    FDTable();

    // fd, index
    void add_fd_entry(int, int);

    int get_index(int);

    int get_fd() const;

    void check() {
        cout << "File Descriptor Table" << endl;
        for (const auto &entry: fd_table) {
            cout << "fd: " << entry.get_fd() << "; index: " << entry.get_index() << endl;
        }
        cout << endl;
    }
};


class FileTableEntry {
private:
    int inode_id;
    int file_cursor;
    int ref_cnt;
    int authority;

public:
    FileTableEntry();

    FileTableEntry(int, int, int, int);

    int get_inode_id() const;

    int get_auth() const;

    int get_cnt() const;

    int get_cursor() const;

    void set_cursor(int);
};


class FileTable {
private:
    vector<FileTableEntry> file_table;

public:
    FileTable();

    // Inode id, authority
    int add_file_entry(int, int);

    int get_file_cursor(int);

    int get_auth(int);

    int get_inode(int) const;

    void set_file_cursor(int, int);

    void check() {
        cout << "File Table" << endl;
        for (const auto &entry: file_table) {
            cout << "inode: " << entry.get_inode_id() << "; cursor: " << entry.get_cursor()
                 << "; ref_cnt: " << entry.get_cnt() << "; auth: ";
            if (entry.get_auth() == O_RDONLY)
                cout << "O_RDONLY" << endl;
            else if (entry.get_auth() == O_WRONLY)
                cout << "O_WRONLY" << endl;
            else
                cout << "O_RDWR" << endl;
        }
        cout << endl;
    }
};


class InodeTableEntry {
private:
    int inode_id;
    int ref_cnt;
    string path;

public:
    InodeTableEntry();

    InodeTableEntry(int, int, const string &);

    int get_id() const;

    int get_cnt() const;

    string get_path() const;

    void add_cnt();
};


class InodeTable {
private:
    vector<InodeTableEntry> inode_table;

public:
    InodeTable();

    void add_inode_entry(int, const string &);

    int get_inode_id(int);

    int match(const string &);

    void add_ref_cnt(int);

    void check() {
        cout << "Inode Table" << endl;
        for (const auto &entry: inode_table) {
            cout << "inode: " << entry.get_id() << "; ref_cnt: " << entry.get_cnt()
                 << "; path: " << entry.get_path() << endl;
        }
        cout << endl;
    }
};


class SmartFS {
private:
    string disk_path;
    string inode_path;
    SmartBuffer buffer;
    FDTable fdTable;
    FileTable fileTable;
    InodeTable inodeTable;
    vector<string> path_array;

    int find_inode(int, const string &);

    // 创建一级目录或文件，参数为 inode id、文件名、类型
    int single_create(int, const string &, int);

    int write_block(int, const string&);

public:
    SmartFS();

    void set_metadata(const string &, const string &);

    // block_num, block_size, inode_num, inode_size
    Error format(int, int, int, int);

    Error initial_buffer(int, int, int, int);

    Error create(const string &path);

    void info_buffer();

    void info_table();

    Error open(const string &path, int);

    Error read(int, int);

    Error write(int, const string &data);

    Error link(const string &o_name, const string &n_name);

    Error symbolic_link(const string &o_name, const string &n_name);

    Error save();

    Error reboot(const string &, const string &);

    Error info_file(const string &path);

    Error info_block(int);

    static Error info_super();

    static Error info_bitmap();

    void run();
};


#endif //SMARTFILESYSTEM_SMARTFS_H
