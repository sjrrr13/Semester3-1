//
// Created by Jiarui She on 2021/11/11.
//

#include "SmartDataStructure.h"

int Block::block_size = 0;
int Block::record_limit = 0;

Block::Block() {
    block_id = 0;
    offset = 0;
    record_num = 0;
    data = nullptr;
}

Block::~Block() {
    delete[] data;
}

void Block::set_metadata(int size) {
    block_size = size;
    record_limit = size / 16;
}

void Block::initial(int id) {
    block_id = id;
    offset = 0;
    data = new char[block_size];
    for (int i = 0; i < block_size; ++i) {
        data[i] = char(0);
    }
}

void Block::set_id(int id) {
    block_id = id;
}

int Block::get_id() const {
    return block_id;
}

int Block::set_data(const string &new_data) {
    int len = (int) new_data.length();
    if (offset + len <= block_size) {
        for (int i = 0; i < len; ++i) {
            data[i + offset] = new_data[i];
        }
        offset += len;
        return 0;
    } else {
        int left_len = len - (block_size - offset);
        for (int i = 0; i < block_size - offset; ++i) {
            data[i + offset] = new_data[i];
        }
        offset = block_size;
        return left_len;
    }
}

char *Block::get_data() const {
    return data;
}

void Block::set_record(DirRecord *record) {
    for (int i = 0; i < 12; ++i) {
        data[i + record_num * 16] = record->name[i];
    }
    memcpy(data + record_num * 16 + 12 * sizeof(char), &(record->id), sizeof(int));
    record_num++;
}

bool Block::can_add_record() {
    record_num = get_record_num();
    return record_num < record_limit;
}

int Block::get_limit() {
    return record_limit;
}

int Block::get_block_size() {
    return block_size;
}

int Block::get_offset() const {
    return offset;
}

int Block::get_record_num() {
    record_num = 0;
    for (int i = 0; i < block_size; i += 16) {
        if (data[i] != char(0))
            record_num++;
    }
    return record_num;
}

int Block::set_char_data(const char *data_p) {
    for (int i = 0; i < block_size; ++i) {
        data[i] = data_p[i];
    }
    return 0;
}


int SuperBlock::capacity = 0;
int SuperBlock::block_size = 0;
int SuperBlock::inode_num = 0;
int SuperBlock::inode_size = 0;
int SuperBlock::next_inode = 0;

void SuperBlock::set_metadata(int b_num, int b_size, int i_num, int i_size) {
    capacity = b_num * b_size;
    block_size = b_size;
    inode_num = i_num;
    inode_size = i_size;
    // inode0为空，inode1为root，所以下一个inode从开始
    next_inode = 2;
}

int SuperBlock::get_next() {
    return next_inode <= inode_num ? next_inode : -1;
}

void SuperBlock::set_next(int id) {
    next_inode = id;
}

string SuperBlock::serialize() {
    string super_data = to_string(capacity) + "-" + to_string(block_size) + "-" + to_string(inode_size) + "-" +
                        to_string(inode_num) + "-" + to_string(inode_num - (next_inode - 1)) + "-";
    return super_data;
}

int SuperBlock::getCapacity() {
    return capacity;
}

int SuperBlock::getBlockSize() {
    return block_size;
}

int SuperBlock::getInodeNum() {
    return inode_num;
}

int SuperBlock::getInodeSize() {
    return inode_size;
}


int Bitmap::block_num = 0;
string Bitmap::usage;

Bitmap::Bitmap() = default;

void Bitmap::set_metadata(int num) {
    block_num = num;
    for (int i = 0; i < block_num / 8 + 1; ++i) {
        usage += (char) 0;
    }
}

void Bitmap::set_used(int index) {
    int order = index / 8;
    char target = usage[order];
    bitset<8> bit_mask = bitset<8>(1);
    bit_mask <<= (7 - index % 8);
    bitset<8> bit_target = bitset<8>(target);
    bit_target = bit_target | bit_mask;
    usage[order] = (char) bit_target.to_ulong();
}

int Bitmap::get_free_block() {
    for (int i = 0; i < block_num; ++i) {
        bitset<8> bit = bitset<8>(usage[i]);
        if (bit.count() < 8) { // 返回新block的同时已经将该位置设置成了used
            bitset<8> mask = bitset<8>(1);
            mask <<= (7 - bit.count());
            bit = bit | mask;
            usage[i] = (char) bit.to_ulong();
            return (int) (i * 8 + bit.count() - 1);
        }
    }
    // 没有空闲的 block
    return -1;
}

string Bitmap::serialize() {
    return usage;
}

void Bitmap::set_usage(const string &data) {
    usage = data;
}


int Inode::inode_size = 0;

Inode::Inode() {
    inode_id = 0;
    file_size = 0;
    file_type = 0;
    block_array = nullptr;
    ptr = 0;
    create = 0;
    last_access = 0;
    last_modify = 0;
}

void Inode::set_metadata(int size) {
    inode_size = size;
}

int Inode::get_limit() {
    return inode_size;
}

void Inode::initial(int id) {
    inode_id = id;
    block_array = new int[inode_size];
    for (int i = 0; i < inode_size; ++i) {
        block_array[i] = 0;
    }
}

int Inode::get_id() const {
    return inode_id;
}

void Inode::set_file_size(int size) {
    file_size = size;
}

void Inode::set_type(int type) {
    file_type = type;
}

int Inode::add_block(int id) {
    for (int i = 0; i < inode_size; ++i) {
        if (block_array[i] == 0) {
            block_array[i] = id;
            return 0;
        }
    }
    return -1;
}

void Inode::set_create(time_t time) {
    create = time;
}

void Inode::set_access(time_t time) {
    last_access = time;
}

void Inode::set_modify(time_t time) {
    last_modify = time;
}

string Inode::serialize() {
    string data = to_string(file_size) + "|" + to_string(file_type) + "|" + to_string(create) + "|"
                  + to_string(last_access) + "|" + to_string(last_modify) + "|";
    string blocks;
    for (int i = 0; i < inode_size; ++i) {
        blocks += to_string(block_array[i]) + "-";
    }
    blocks += "|";
    return data + blocks;
}

int *Inode::get_blocks() const {
    return block_array;
}

int Inode::get_type() const {
    return file_type;
}

int Inode::get_file_size() const {
    return file_size;
}

string Inode::get_create() const {
    char *time = ctime(&create);
    return time;
}

string Inode::get_access() const {
    char *time = ctime(&last_access);
    return time;
}

string Inode::get_modify() const {
    char *time = ctime(&last_modify);
    return time;
}
