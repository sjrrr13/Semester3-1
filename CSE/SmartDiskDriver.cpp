//
// Created by Jiarui She on 2021/11/12.
//

#include "SmartDiskDriver.h"

string SmartDiskDriver::smart_disk_path;
string SmartDiskDriver::smart_inode_path;
int SmartDiskDriver::smart_block_num = 0;
int SmartDiskDriver::smart_block_size = 0;
int SmartDiskDriver::smart_inode_num = 0;
int SmartDiskDriver::smart_inode_size = 0;

SmartDiskDriver::SmartDiskDriver() = default;

void SmartDiskDriver::set_metadata(const string &block_path, const string &inode_path) {
    smart_disk_path = block_path;
    smart_inode_path = inode_path;
}

Error SmartDiskDriver::format(int block_num, int block_size, int inode_num, int inode_size) {
    smart_block_num = block_num;
    smart_block_size = block_size;
    smart_inode_num = inode_num;
    smart_inode_size = inode_size;
    Block block = Block();
    Block *block_ptr = &block;
    Inode inode = Inode();
    Inode *inode_ptr = &inode;
    Error error;

    // 初始化Boot Block
    block_ptr->initial(0);
    block_ptr->set_data("BootBlock");
    error = save_block(block_ptr);
    if (error.get_code() != SUCCESS) {
        cout << "Error in SmartDiskDriver::format()" << endl;
        return error;
    }

    // 初始化SuperBlock
    SuperBlock::set_metadata(block_num, block_size, inode_num, inode_size);
    block_ptr->initial(1);
    block_ptr->set_data(SuperBlock::serialize());
    error = save_block(block_ptr);
    if (error.get_code() != SUCCESS) {
        cout << "Error in SmartDiskDriver::format()" << endl;
        return error;
    }

    // 初始化Bitmap，此时只有block0、block1、block2、block3（root占用）被用到
    Bitmap::set_metadata(block_num);
    Bitmap::set_used(0);
    Bitmap::set_used(1);
    Bitmap::set_used(2);
    Bitmap::set_used(3);
    block_ptr->initial(2);
    block_ptr->set_data(Bitmap::serialize());
    error = save_block(block_ptr);
    if (error.get_code() != SUCCESS) {
        cout << "Error in SmartDiskDriver::format()" << endl;
        return error;
    }

    // 初始化其他block
    for (int i = 3; i < block_num; ++i) {
        block_ptr->initial(i);
        save_block(block_ptr);
        if (error.get_code() != SUCCESS) {
            cout << "Error in SmartDiskDriver::format()" << endl;
            return error;
        }
    }

    // 初始化root
    inode_ptr->initial(1);
    time_t now = time(nullptr);
    inode_ptr->set_create(now);
    inode_ptr->set_access(now);
    inode_ptr->set_modify(now);
    inode_ptr->add_block(3);
    error = save_inode(inode_ptr);
    if (error.get_code() != SUCCESS) {
        cout << "Error in SmartDiskDriver::format()" << endl;
        return error;
    }

    return Error(SUCCESS);
}

Error SmartDiskDriver::save_block(Block *block) {
    int disk_offset = smart_block_size * block->get_id();
    fstream stream(smart_disk_path, ios::binary | ios::out | ios::in);
    if (!stream) {
        cout << "Error when writing data to block" << endl;
        return Error(IO_EXCEPTION);
    }
    // 跳转到disk中该block起始位置进行写入
    stream.seekp(disk_offset, ios::beg);
    stream.write(block->get_data(), smart_block_size);
    stream.close();
    return Error(SUCCESS);
}

Error SmartDiskDriver::load_block(Block *block, int id) {
    int disk_offset = smart_block_size * id;
    fstream in(smart_disk_path, ios::in);
    if (!in) {
        cout << "Error when reading data to block" << endl;
        return Error(IO_EXCEPTION);
    }

    char data[smart_block_size];
    in.seekg(disk_offset, ios::beg);
    in.read(data, smart_block_size);
    block->set_id(id);
    block->set_char_data(data);
    in.close();
    return Error(SUCCESS);
}

Error SmartDiskDriver::save_inode(Inode *inode) {
    string path = smart_inode_path + to_string(inode->get_id()) + ".inode";
    ofstream out(path, ios::out);
    if (!out) {
        cout << "Error when writing data to inode" << endl;
        return Error(IO_EXCEPTION);
    }
    out << inode->serialize();
    out.close();
    return Error(SUCCESS);
}

Error SmartDiskDriver::load_inode(Inode *inode, int id) {
    string path = smart_inode_path + to_string(id) + ".inode";
    ifstream in(path, ios::in);
    if (!in) {
        cout << "Error when reading data to inode" << endl;
        return Error(IO_EXCEPTION);
    }
    string data;
    in >> data;

    inode->initial(id);
    vector<string> items = Tool::my_split(data, '|');
    inode->set_file_size(atoi(items[0].c_str()));
    inode->set_type(atoi(items[1].c_str()));
    inode->set_create(atoi(items[2].c_str()));
    inode->set_access(atoi(items[3].c_str()));
    inode->set_modify(atoi(items[4].c_str()));

    vector<string> block_str = Tool::my_split(items[5], '-');
    int limit = Inode::get_limit();
    int *blocks = inode->get_blocks();
    for (int i = 0; i < limit; ++i) {
        blocks[i] = atoi(block_str[i].c_str());
    }
    in.close();
    return Error(SUCCESS);
}

Error SmartDiskDriver::save_metadata() {
    ofstream out("../metadata.txt", ios::out);
    if (!out) {
        cout << "Error when saving metadata" << endl;
        return Error(IO_EXCEPTION);
    }
    string data = to_string(smart_block_size) + ";" + to_string(smart_block_num) + ";"
            + to_string(smart_inode_size) + ";" + to_string(smart_inode_num) + ";";
    out << data;
    out.close();
    return Error(SUCCESS);
}

Error SmartDiskDriver::load_metadata() {
    ifstream in("../metadata.txt", ios::in);
    if (!in) {
        cout << "Error when loading metadata" << endl;
        return Error(IO_EXCEPTION);
    }
    string data;
    in >> data;
    vector<string> metadata = Tool::my_split(data, ';');
    smart_block_size = atoi(metadata[0].c_str());
    smart_block_num = atoi(metadata[1].c_str());
    smart_inode_size = atoi(metadata[2].c_str());
    smart_inode_num = atoi(metadata[3].c_str());
    in.close();
    return Error(SUCCESS);
}

int SmartDiskDriver::getSmartBlockNum() {
    return smart_block_num;
}

int SmartDiskDriver::getSmartBlockSize() {
    return smart_block_size;
}

int SmartDiskDriver::getSmartInodeNum() {
    return smart_inode_num;
}

int SmartDiskDriver::getSmartInodeSize() {
    return smart_inode_size;
}
