//
// Created by Jiarui She on 2021/11/11.
//

#include "SmartFS.h"

using namespace std;

// 以下是 FDTable 类相关的方法
FDTableEntry::FDTableEntry() {
    file_descriptor = 0;
    file_table_index = 0;
}

FDTableEntry::FDTableEntry(int fd, int index) {
    file_descriptor = fd;
    file_table_index = index;
}

int FDTableEntry::get_fd() const {
    return file_descriptor;
}

int FDTableEntry::get_index() const {
    return file_table_index;
}


// 以下是 FDTable 类相关的方法
FDTable::FDTable() {
    next_fd = 3;
}

void FDTable::add_fd_entry(int fd, int index) {
    FDTableEntry entry = FDTableEntry(fd, index);
    fd_table.push_back(entry);
    next_fd++;
}

int FDTable::get_index(int fd) {
    for (const FDTableEntry &entry: fd_table) {
        if (entry.get_fd() == fd)
            return entry.get_index();
    }
    return -1;
}

int FDTable::get_fd() const {
    return next_fd;
}


// 以下是 FileTableEntry 类相关的方法
FileTableEntry::FileTableEntry() {
    inode_id = 0;
    file_cursor = 0;
    ref_cnt = 0;
    authority = 0;
}

FileTableEntry::FileTableEntry(int inode, int cursor, int ref, int auth) {
    inode_id = inode;
    file_cursor = cursor;
    ref_cnt = ref;
    authority = auth;
}

int FileTableEntry::get_inode_id() const {
    return inode_id;
}

int FileTableEntry::get_auth() const {
    return authority;
}

int FileTableEntry::get_cnt() const {
    return ref_cnt;
}

int FileTableEntry::get_cursor() const {
    return file_cursor;
}

void FileTableEntry::set_cursor(int index) {
    file_cursor = index;
}


// 以下是 FileTable 类相关的方法
FileTable::FileTable() = default;

int FileTable::add_file_entry(int inode, int auth) {
    FileTableEntry entry = FileTableEntry(inode, 0, 1, auth);
    file_table.push_back(entry);
    return (int) (file_table.size() - 1);
}

int FileTable::get_file_cursor(int index) {
    return file_table[index].get_cursor();
}

void FileTable::set_file_cursor(int index, int cursor) {
    return file_table[index].set_cursor(cursor);

}

int FileTable::get_inode(int index) const {
    return file_table[index].get_inode_id();
}

int FileTable::get_auth(int index) {
    return file_table[index].get_auth();
}


// 以下是 InodeTableEntry 类相关的方法
InodeTableEntry::InodeTableEntry() {
    inode_id = 0;
    ref_cnt = 0;
    path = "";
}

InodeTableEntry::InodeTableEntry(int id, int ref, const string &path) {
    inode_id = id;
    ref_cnt = ref;
    this->path = path;
}

int InodeTableEntry::get_id() const {
    return inode_id;
}

int InodeTableEntry::get_cnt() const {
    return ref_cnt;
}

string InodeTableEntry::get_path() const {
    return path;
}

void InodeTableEntry::add_cnt() {
    ref_cnt++;
}


// 以下是 InodeTable 类相关的方法
InodeTable::InodeTable() = default;

void InodeTable::add_inode_entry(int id, const string &path) {
    InodeTableEntry entry = InodeTableEntry(id, 1, path);
    inode_table.push_back(entry);
}

int InodeTable::get_inode_id(int index) {
    return inode_table[index].get_id();
}

// 返回路径对应的 inode id
int InodeTable::match(const string &path) {
    for (auto &i: inode_table) {
        if (i.get_path() == path)
            return i.get_id();
    }
    return -1;
}

void InodeTable::add_ref_cnt(int index) {
    inode_table[index].add_cnt();
}


int SmartFS::find_inode(int id, const string &sub_dir) {
    // id所对应inode的block数组
    int *blocks = buffer.get_inode(id)->get_blocks();
    Block *block;
    for (int i = 0; i < Inode::get_limit(); ++i) { // 遍历每个block
        block = buffer.get_block(blocks[i]);
        if (block->get_id() == 0) { // 找到未被分配的block，说明没有找到子目录
            return -1;
        }
        char *data = block->get_data();

        DirRecord record{};
        // 解析DirRecord
        int record_num = block->get_record_num();
        for (int j = 0; j < record_num; ++j) {
            memcpy(&(record.name), data + 16 * j, 12 * sizeof(char));
            if (record.name == sub_dir) {
                memcpy(&(record.id), data + 16 * j + 12 * sizeof(char), sizeof(int));
                return record.id;
            }
        }
    }
    return -1;
}

int SmartFS::single_create(int id, const string &name, int type) {
    // 先检查参数是否符合要求
    if (name.length() > 12) {
        cout << "Error when creating " << name << endl;
        cout << "File name cannot be larger than 12" << endl;
        return INVALID_ARG;
    }

    // 参数符合要求的话获取一个新的inode，如果inode已经全部被用了，会创建失败
    int new_id = buffer.get_new_inode(type);
    if (new_id > 1000) {
        return new_id;
    }

    // 创建一条目录记录
    DirRecord record{};
    for (int i = 0; i < name.length(); ++i) {
        record.name[i] = name[i];
    }
    record.id = new_id;

    // 写数据到buffer
    Inode *inode = buffer.get_inode(id);
    int *blocks = inode->get_blocks();
    for (int i = 0; i < Inode::get_limit(); ++i) {
        if (blocks[i] == 0) { // 新分配一个block
            int new_block = buffer.allocate_block(id);
            if (new_block > 1000) {
                cout << "Error when adding a new block to inode " << id << endl;
                return new_block;
            }

            buffer.get_block(new_block)->set_record(&record);
            return new_id;
        }
        if (buffer.get_block(blocks[i])->can_add_record()) { // 如果遍历到的块可以写入记录的话
            buffer.get_block(blocks[i])->set_record(&record);
            return new_id;
        } else {
            continue;
        }
    }

    // inode上的所有block都满了
    cout << "Error when creating " << name << endl;
    return INODE_FULL;
}

int SmartFS::write_block(int inode_id, const string &data) {
    int new_block_id = buffer.get_inode_free_block(inode_id);
    if (new_block_id > 1000) {
        cout << "Error when allocating a new block to inode " << inode_id << endl;
        return new_block_id;
    }
    int result = buffer.get_block(new_block_id)->set_data(data);
    if (result == 0) // 全部写完
        return SUCCESS;
    int len = (int) data.length();
    string left_data = data.substr(len - result, result);
    return write_block(inode_id, left_data);
}

SmartFS::SmartFS() = default;

void SmartFS::set_metadata(const string &disk, const string &inode) {
    disk_path = disk;
    inode_path = inode;
}

Error SmartFS::format(int block_num, int block_size, int inode_num, int inode_size) {
    cout << "Formatting block and inode..." << endl;
    if (block_size < 16) {
        cout << "Error when formatting: block size should be larger than 16 (recommended N * 16)";
        return Error(INVALID_ARG);
    }
    Block::set_metadata(block_size);
    Inode::set_metadata(inode_size);
    cout << "Loading SmartDiskDriver..." << endl;
    SmartDiskDriver::set_metadata(disk_path, inode_path);

    return SmartDiskDriver::format(block_num, block_size, inode_num, inode_size);
}

Error SmartFS::initial_buffer(int block_num, int block_size, int inode_num, int inode_size) {
    cout << "Initializing buffer..." << endl;
    return buffer.initial(block_num, block_size, inode_num, inode_size);
}

// 需要初始化三个时间戳
Error SmartFS::create(const string &path) {
    // 如果文件已经存在， 返回相应的错误
    if (inodeTable.match(path) > 0) {
        return Error(FILE_EXISTS);
    } else if (Tool::path_resolution(path, path_array) != -1) {
        int file_type;
        string file_name = path_array.back();
        int len = (int) file_name.length();
        // 解析最后要创建的是文件还是目录
        if (len == 0) { // 创建目录，否则是创建文件
            path_array.pop_back();
            file_type = 1;
        } else {
            file_type = 2;
        }
        int path_num = (int) path_array.size();

        int new_id;
        int current_id = 1;
        int next_id;
        for (int i = 0; i < path_num - 1; ++i) {
            next_id = find_inode(current_id, path_array[i]);
            if (next_id < 0) { // 没有找到 inode
                next_id = single_create(current_id, path_array[i], DIR);
                if (next_id > 1000) { // 单次创建失败
                    return Error(next_id);
                }
            }
            current_id = next_id;
        }
        new_id = single_create(current_id, path_array[path_num - 1], file_type);
        if (new_id > 1000) { // 单次创建失败
            return Error(new_id);
        }

        // 修改三张表的表项
        inodeTable.add_inode_entry(new_id, path);
        int index = fileTable.add_file_entry(new_id, O_RDWR);
        int fd = fdTable.get_fd();
        fdTable.add_fd_entry(fdTable.get_fd(), index);

        path_array.clear();
        return Error(fd);
    } else {
        cout << "Error when creating file" << endl;
        return Error(INVALID_ARG);
    }
}

void SmartFS::info_buffer() {
    buffer.info();
}

void SmartFS::info_table() {
    fdTable.check();
    fileTable.check();
    inodeTable.check();
}

// 需要修改last_access
Error SmartFS::open(const string &path, int auth) {
    int current_id = inodeTable.match(path);
    // 要打开的文件的inode已经在inode table中了
    if (current_id > 0) {
        inodeTable.add_ref_cnt(current_id); // ref_cnt++
        goto end;
    } else if (Tool::path_resolution(path, path_array) != -1) { // 将要打开的文件的inode添加到inode table中
        int next_id;
        // 打开文件夹的情况
        if (path_array.back().empty()) {
            path_array.pop_back();
        }
        int path_num = (int) path_array.size();

        current_id = 1;
        for (int i = 0; i < path_num; ++i) {
            next_id = find_inode(current_id, path_array[i]);
            // 要求的路径中有不存在的 inode，即有目录或文件不存在
            if (next_id < 0) {
                cout << "Error when searching " << path_array[i] << endl;
                path_array.clear();
                return Error(NO_SUCH_FILE);
            }
            current_id = next_id;
        }
        inodeTable.add_inode_entry(current_id, path);
        path_array.clear();
        goto end;
    } else { // 输入的路径不正确
        return Error(INVALID_ARG);
    }

    end:
    buffer.get_inode(current_id)->set_access(time(nullptr)); // 设置last access
    int index = fileTable.add_file_entry(current_id, auth);

    int fd = fdTable.get_fd();
    fdTable.add_fd_entry(fd, index);
    return Error(fd);
}

Error SmartFS::read(int fd, int byte_num) {
    int index = fdTable.get_index(fd);
    if (index == -1) { // 不存在该fd
        cout << "Error when searching file descriptor " << fd << endl;
        return Error(NO_SUCH_FILE);
    }
    int auth = fileTable.get_auth(index);
    if (auth == O_WRONLY) { // 文件为只写
        cout << "Error when reading file descriptor " << fd << endl;
        return Error(NO_AUTH);
    }
    int inode_id = fileTable.get_inode(index);
    Inode *inode = buffer.get_inode(inode_id);
    int *blocks = inode->get_blocks();
    char *data;

    // 读文件夹的情况，应该被解析为DirRecord
    if (inode->get_type() == DIR) {
        DirRecord record{};
        vector<string> names;

        for (int i = 0; i < Inode::get_limit(); ++i) {
            if (blocks[i] == 0) // 没有更多的记载信息的block了
                goto out;
            data = buffer.get_block(blocks[i])->get_data(); // 得到当前block的信息
            for (int j = 0; j < Block::get_limit(); ++j) { // 解析DirRecord
                memcpy(&(record.name), data + 16 * j, 12 * sizeof(char));
                memcpy(&(record.id), data + 16 * j + 12 * sizeof(char), sizeof(int));
                if (record.id == 0) { // 说明已经没有更多的记录了
                    goto out;
                } else {
                    names.emplace_back(record.name); // 记录下文件夹下的文件名
                }
            }
        }
        out:
        if (names.empty()) { // 文件夹为空
            cout << "Error when reading file descriptor " << fd << endl;
            return Error(EMPTY);
        } else { // 输出
            for (const string &name: names) {
                cout << name << endl;
            }
        }
        return Error(SUCCESS);
    }

    // 读文件的情况
    int cursor = fileTable.get_file_cursor(index);
    int increase = byte_num;
    if (cursor == inode->get_file_size()) {
        if (cursor > 0) { // EOF异常
            cout << "Error when reading file descriptor " << fd << endl;
            return Error(END_OF_FILE);
        } else { // 读的文件为空
            cout << "Error when reading file descriptor " << fd << endl;
            return Error(EMPTY);
        }
    }
    int start_block = cursor / Block::get_block_size(); // 起始块
    int start_index = cursor % Block::get_block_size(); // 起始下标

    string buf;
    int i = start_block;
    while (byte_num > 0) {
        if (blocks[i] == 0) // 要读的长度超过了可以读的长度
            break;
        data = buffer.get_block(blocks[i])->get_data();
        string str_data(data);
        int len = (int) str_data.length();
        str_data = str_data.substr(start_index, len - start_index);
        if (byte_num > len) {
            buf += str_data;
            byte_num -= len;
        } else {
            buf += str_data.substr(0, byte_num);
            byte_num = 0;
        }
        start_index = 0;
        i++;
    }

    cursor += increase;
    if (cursor > inode->get_file_size())
        cursor = inode->get_file_size();
    fileTable.set_file_cursor(index, cursor);
    cout << buf << endl;
    return Error(SUCCESS);
}

// 修改last_modify
Error SmartFS::write(int fd, const string &data) {
    int index = fdTable.get_index(fd);
    if (index == -1) {
        cout << "Error when searching file descriptor " << fd << endl;
        return Error(NO_SUCH_FILE);
    }
    int auth = fileTable.get_auth(index);
    if (auth == O_RDONLY) { // 文件为只读
        cout << "Error when writing file descriptor " << fd << endl;
        return Error(NO_AUTH);
    }
    int inode_id = fileTable.get_inode(index);
    if (buffer.get_inode(inode_id)->get_type() == DIR) { // 写目录会报错
        cout << "Error when writing file descriptor " << fd << endl;
        return Error(WR_DIR);
    }
    int result = write_block(inode_id, data);
    if (result > 1000) {
        cout << "Error when writing file descriptor " << fd << endl;
        return Error(result);
    }
    int size = buffer.get_inode(inode_id)->get_file_size() + (int) data.length();
    buffer.get_inode(inode_id)->set_file_size(size);
    buffer.get_inode(inode_id)->set_modify(time(nullptr));
    return Error(SUCCESS);
}

Error SmartFS::link(const string &o_name, const string &n_name) {
    int o_inode = inodeTable.match(o_name); // old name对应的inode
    if (o_inode > 0) {
        if (buffer.get_inode(o_inode)->get_type() != DIR) {
            inodeTable.add_inode_entry(o_inode, n_name);
            goto end;
        } else { // 链接目录会报错
            cout << "Error when linking " << o_name << " to " << n_name << endl;
            return Error(LINK_DIR);
        }
    } else if (Tool::path_resolution(o_name, path_array) != -1) { // inode table中没有对应项
        int next_id;
        if (path_array.back().empty()) { // 打开文件夹的情况
            cout << "Error when linking " << o_name << " to " << n_name << endl;
            return Error(LINK_DIR);
        }
        int path_num = (int) path_array.size();

        int current_id = 1;
        for (int i = 0; i < path_num; ++i) {
            next_id = find_inode(current_id, path_array[i]);

            if (next_id < 0) { // 要求的路径中有不存在的 inode，即有目录或文件不存在
                cout << "Error when linking " << o_name << " to " << n_name << endl;
                return Error(NO_SUCH_FILE);
            }
            current_id = next_id;
        }

        inodeTable.add_inode_entry(current_id, n_name);
        path_array.clear();
        goto end;
    } else {
        cout << "Error when linking " << o_name << " to " << n_name << endl;
        return Error(INVALID_ARG);
    }
    end:
    int index = fileTable.add_file_entry(o_inode, O_RDWR);
    int fd = fdTable.get_fd();
    fdTable.add_fd_entry(fd, index);

    return Error(fd);
}

Error SmartFS::symbolic_link(const string &o_name, const string &n_name) {
    Error e = create(n_name);
    if (e.get_code() > 0 && e.get_code() < 1000) {
        Error e2 = write(e.get_code(), o_name); // 将old name写到新文件的block中去
        return e2;
    }
    return e;
}

Error SmartFS::save() {
    cout << "Writing data from buffer to disk" << endl;
    Error error = buffer.write_back();
    if (error.get_code() > 1000) {
        cout << "Error when saving file system" << endl;
        return error;
    }
    return Error(SUCCESS);
}

Error SmartFS::reboot(const string &disk, const string &inode) {
    // 设置路径
    set_metadata(disk, inode);
    // load Disk Driver
    cout << "Loading SmartDiskDriver" << endl;
    SmartDiskDriver::set_metadata(disk_path, inode_path);
    // load metadata，例如block number，block size，inode number，inode size
    SmartDiskDriver::load_metadata();
    Block::set_metadata(SmartDiskDriver::getSmartBlockSize());
    Inode::set_metadata(SmartDiskDriver::getSmartInodeSize());
    cout << "Initializing buffer" << endl;
    Error error = buffer.reboot();

    if (error.get_code() > 1000) {
        cout << "Error when rebooting file system" << endl;
        return error;
    }
    return Error(SUCCESS);
}

Error SmartFS::info_file(const string &path) {
    int id = inodeTable.match(path);
    if (id < 0) {
        cout << "Error when searching file " << path << endl;
        return Error(NO_SUCH_FILE);
    } else {
        Inode *inode = buffer.get_inode(id);
        int *blocks = inode->get_blocks();
        cout << "Blocks: ";
        for (int i = 0; i < Inode::get_limit(); ++i) {
            if (blocks[i] != 0)
                cout << "block" << blocks[i] << "; ";
        }
        cout << endl;
        cout << "Size: " << inode->get_file_size() << endl;
        cout << "Type: " << (inode->get_type() == 1 ? "Directory" : "File") << endl;
        cout << "Create time: " << inode->get_create();
        cout << "Last access time: " << inode->get_access();
        cout << "Last modify time: " << inode->get_modify();
        return Error(SUCCESS);
    }
}

void SmartFS::run() {
    bool tag = true;
    string start = "Smart File System started!\n"
                   "Use -h to get help\n";
    cout << start << endl;
    string help = "\tTo set the disk file and the inode table directory, use:\n"
                  "\t\tset [disk file] [inode table directory]\n"
                  "\n\tTo format the disk, use:\n"
                  "\t\tformat [block number] [block size] [inode number] [inode size]\n"
                  "\n\tTo reboot the system from a file, use:\n"
                  "\t\treboot [disk file] [inode table directory]\n"
                  "\n\tTo create a file or a directory, use:\n"
                  "\t\tcreate [file name]\n"
                  "\n\tTo open a file, use:\n"
                  "\t\topen [file name] [authority: O_RDONLY; O_WRONLY; O_RDWR]\n"
                  "\n\tTo read a file or list a directory, use:\n"
                  "\t\tread [file descriptor] [byte number]\n"
                  "\n\tTo write a file, use:\n"
                  "\t\twrite [file descriptor] \"[content]\"\n"
                  "\n\tTo link a file to a new name, use:\n"
                  "\t\tlink [old name] [new name]\n"
                  "\n\tTo symbolic link a file to a new name, use:\n"
                  "\t\tlink -s [old name] [new name]\n"
                  "\n\tTo check a file, use:\n"
                  "\t\tcheck -f [file name]\n"
                  "\n\tTo check a block, use:\n"
                  "\t\tcheck -bk [block id]\n"
                  "\n\tTo check super block, use:\n"
                  "\t\tcheck -s\n"
                  "\n\tTo check bitmap, use:\n"
                  "\t\tcheck -bm\n"
                  "\n\tTo check three tables, use:\n"
                  "\t\tcheck -t\n"
                  "\n\tTo check buffer, use:\n"
                  "\t\tcheck -b\n"
                  "\n\tTo exit without save data, use:\n"
                  "\t\texit\n"
                  "\n\tTo exit and save data, use:\n"
                  "\t\texit -s\n";
    string instruction;
    Error e;
    string arg1;
    vector<string> args;
    while (tag) {
        cout << "SmartFileSystem >> ";
        getline(cin, instruction);
        if (instruction == "-h") {
            cout << help;
            continue;
        } else if (instruction.substr(0, 3) == "set") {
            arg1 = instruction.substr(4, instruction.length() - 4) + " ";
            args = Tool::my_split(arg1, ' ');
            set_metadata(args[0], args[1]);
        } else if (instruction.substr(0, 6) == "format") {
            arg1 = instruction.substr(7, instruction.length() - 7) + " ";
            args = Tool::my_split(arg1, ' ');
            if (args.size() != 4) {
                e = Error(INVALID_ARG);
                goto end;
            } else {
                e = format(atoi(args[0].c_str()), atoi(args[1].c_str()), atoi(args[2].c_str()), atoi(args[3].c_str()));
                if (e.get_code() == SUCCESS) {
                    e = initial_buffer(atoi(args[0].c_str()), atoi(args[1].c_str()), atoi(args[2].c_str()),
                                       atoi(args[3].c_str()));
                }
                goto end;
            }
        } else if (instruction.substr(0, 6) == "create") {
            arg1 = instruction.substr(7, instruction.length() - 7);
            e = create(arg1);
            if (e.get_code() > 1 && e.get_code() < 1000)
                cout << "file descriptor: " << e.get_code() << endl;
            else
                goto end;
        } else if (instruction.substr(0, 4) == "open") {
            arg1 = instruction.substr(5, instruction.length() - 5) + " ";
            args = Tool::my_split(arg1, ' ');
            if (args.size() != 2)
                e = Error(INVALID_ARG);
            if (args[1] == "O_RDONLY")
                e = open(args[0], O_RDONLY);
            else if (args[1] == "O_WRONLY")
                e = open(args[0], O_WRONLY);
            else if (args[1] == "O_RDWR")
                e = open(args[0], O_RDWR);
            else {
                e = Error(INVALID_ARG);
                goto end;
            }
            if (e.get_code() > 1 && e.get_code() < 1000)
                cout << "file descriptor: " << e.get_code() << endl;
            else
                goto end;
        } else if (instruction.substr(0, 4) == "read") {
            arg1 = instruction.substr(5, instruction.length() - 5) + " ";
            args = Tool::my_split(arg1, ' ');
            if (args.size() != 2) {
                e = Error(INVALID_ARG);
                goto end;
            }
            e = read(atoi(args[0].c_str()), atoi(args[1].c_str()));
            if (e.get_code() > 1000)
                goto end;
            continue;
        } else if (instruction.substr(0, 5) == "write") {
            int index = (int) instruction.find_first_of('\"');
            arg1 = instruction.substr(6, index - 7);
            int fd = atoi(arg1.c_str());
            arg1 = instruction.substr(index + 1, instruction.length() - index - 2);
            e = write(fd, arg1);
            goto end;
        } else if (instruction.substr(0, 7) == "link -s") {
            arg1 = instruction.substr(8, instruction.length() - 8) + " ";
            args = Tool::my_split(arg1, ' ');
            if (args.size() != 2) {
                e = Error(INVALID_ARG);
                goto end;
            }
            e = symbolic_link(args[0], args[1]);
            if (e.get_code() > 1 && e.get_code() < 1000)
                cout << "file descriptor: " << e.get_code() << endl;
            else
                goto end;
        } else if (instruction.substr(0, 4) == "link") {
            arg1 = instruction.substr(5, instruction.length() - 5) + " ";
            args = Tool::my_split(arg1, ' ');
            if (args.size() != 2) {
                e = Error(INVALID_ARG);
                goto end;
            }
            e = link(args[0], args[1]);
            if (e.get_code() > 1 && e.get_code() < 1000)
                cout << "file descriptor: " << e.get_code() << endl;
            else
                goto end;
        } else if (instruction.substr(0, 6) == "reboot") {
            arg1 = instruction.substr(7, instruction.length() - 7) + " ";
            args = Tool::my_split(arg1, ' ');
            if (args.size() != 2) {
                e = Error(INVALID_ARG);
                goto end;
            }
            e = reboot(args[0], args[1]);
            goto end;
        } else if (instruction.substr(0, 7) == "exit -s") {
            e = save();
            if (e.get_code() != SUCCESS)
                goto end;
            e = SmartDiskDriver::save_metadata();
            if (e.get_code() != SUCCESS)
                goto end;
            cout << "Successfully saved" << endl;
            cout << "Goodbye :)" << endl;
            tag = false;
            continue;
        } else if (instruction == "exit") {
            cout << "Goodbye :)" << endl;
            tag = false;
            continue;
        } else if (instruction.substr(0, 5) == "check") {
            arg1 = instruction.substr(6, instruction.length() - 6);
            if (arg1 == "-t") {
                info_table();
                continue;
            } else if (arg1 == "-b") {
                info_buffer();
                continue;
            } else if (arg1 == "-s") {
                e = info_super();
                continue;
            } else if (arg1 == "-bm") {
                e = info_bitmap();
                continue;
            } else if (arg1.substr(0, 3) == "-bk") {
                arg1 = arg1.substr(4, arg1.length() - 4);
                e = info_block(atoi(arg1.c_str()));
                continue;
            } else if (arg1.substr(0, 2) == "-f") {
                arg1 = arg1.substr(3, arg1.length() - 3);
                e = info_file(arg1);
                if (e.get_code() > 1000) {
                    cout << "Error when getting file information" << endl;
                    goto end;
                }
                continue;
            } else {
                e = Error(INVALID_ARG);
                goto end;
            }
        } else {
            cout << "Invalid instruction, please input again or use -h to see more information" << endl;
            continue;
        }

        end:
        Error::handle(e);
    }
}

Error SmartFS::info_block(int id) {
    Block *block = buffer.get_block(id);
    char *data = block->get_data();
    cout << "Block " << id << ": " << endl;
    for (int i = 0; i < Block::get_block_size(); ++i) {
        cout << data[i] << "; ";
    }
    cout << endl;
    return Error(SUCCESS);
}

Error SmartFS::info_super() {
    cout << "SuperBlock: " << endl;
    cout << "capacity: " << SuperBlock::getCapacity() << endl;
    cout << "block size: " << SuperBlock::getBlockSize() << endl;
    cout << "inode number: " << SuperBlock::getInodeNum() << endl;
    cout << "inode size: " << SuperBlock::getInodeSize() << endl;
    cout << "free inode number: " << (SuperBlock::getInodeNum() - SuperBlock::get_next()) << endl;
    return Error(SUCCESS);
}

Error SmartFS::info_bitmap() {
    cout << "Bitmap: " << endl;
    string data = Bitmap::serialize();
    for (char i: data) {
        bitset<8> bits = bitset<8>(i);
        cout << bits << " ";
    }
    int left = SmartDiskDriver::getSmartBlockSize() - (int) data.length();
    for (int i = 0; i < left; ++i) {
        bitset<8> bits = bitset<8>();
        cout << bits << " ";
    }
    cout << endl;
    return Error(SUCCESS);
}
