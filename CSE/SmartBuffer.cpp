//
// Created by Jiarui She on 2021/11/11.
//

#include "SmartBuffer.h"

SmartBuffer::SmartBuffer() = default;

SmartBuffer::~SmartBuffer() {
    delete[] block_buffer;
    delete[] inode_buffer;
}

Error SmartBuffer::initial(int block_num, int block_size, int inode_num, int inode_size) {
    for (int i = 0; i < BLOCK_NUM; ++i) {
        block_buffer[i].initial(0);
    }
    for (int i = 0; i < INODE_NUM; ++i) {
        inode_buffer[i].initial(0);
    }
    // load super block
    SuperBlock::set_metadata(block_num, block_size, inode_num, inode_size);
    // load bitmap
    Bitmap::set_metadata(block_num);
    Bitmap::set_used(0);
    Bitmap::set_used(1);
    Bitmap::set_used(2);
    Bitmap::set_used(3);

    // 加载root的block
    Error error = SmartDiskDriver::load_block(&(block_buffer[0]), 3);
    if (error.get_code() != SUCCESS) {
        cout << "Error when initializing buffer";
        return error;
    }

    // 加载root
    error = SmartDiskDriver::load_inode(&(inode_buffer[0]), 1);
    if (error.get_code() != SUCCESS) {
        cout << "Error when initializing buffer";
        return error;
    }

    block_head = 0;
    block_tail = 1;
    inode_head = 0;
    inode_tail = 1;

    return Error(SUCCESS);
}

Block *SmartBuffer::get_block(int id) {
    for (int i = 0; i < BLOCK_NUM; ++i) { // 先查找缓存里是否有这个block
        if (block_buffer[i].get_id() == id) {
            return &(block_buffer[i]);
        }
    }
    if (block_tail - block_head == BLOCK_NUM) { // 缓存满了，将head换出去
        int index = change_block();
        SmartDiskDriver::load_block(&(block_buffer[index]), id);
        return &(block_buffer[index]);
    } else { // 缓存没有满，直接将填充到tail
        SmartDiskDriver::load_block(&(block_buffer[block_tail % BLOCK_NUM]), id);
        block_tail++;
        return &(block_buffer[(block_tail - 1) % BLOCK_NUM]);
    }
}

Inode *SmartBuffer::get_inode(int id) {
    for (int i = 0; i < INODE_NUM; ++i) { // 先查找缓存里是否有这个inode
        if (inode_buffer[i].get_id() == id) {
            return &(inode_buffer[i]);
        }
    }
    if (inode_tail - inode_head == INODE_NUM) { // 缓存满了，将head换出去
        int index = change_inode();
        SmartDiskDriver::load_inode(&(inode_buffer[index]), id);
        return &(inode_buffer[index]);
    } else { // 缓存没有满，直接将填充到tail
        SmartDiskDriver::load_inode(&(inode_buffer[inode_tail % INODE_NUM]), id);
        inode_tail++;
        return &(inode_buffer[(inode_tail - 1) % INODE_NUM]);
    }
}

int SmartBuffer::get_new_block() {
    int new_id = Bitmap::get_free_block(); // 得到一个空的block id
    if (new_id < 0) { // 如果block都非空，报错
        cout << "Error when allocating a new block" << endl;
        return NO_MORE_BLOCK;
    }
    int index = 0;
    if (block_tail - block_head == BLOCK_NUM) {
        index = change_block(); // 换一个block出去
    } else {
        index = block_tail;
        block_tail++;
    }
    block_buffer[index].initial(new_id);
    return new_id;
}

int SmartBuffer::get_new_inode(int type) {
    int new_id = SuperBlock::get_next(); // 得到一个新的inode id
    if (new_id < 0) { // 如果inode都被用完了，报错
        cout << "Error when creating a new inode" << endl;
        return NO_MORE_INODE;
    }
    SuperBlock::set_next(new_id + 1);
    int index = 0;
    if (inode_tail - inode_head == INODE_NUM) {
        index = change_inode(); // 换一个inode出去
    } else {
        index = inode_tail;
        inode_tail++;
    }
    inode_buffer[index].initial(new_id); // 初始化新的inode
    inode_buffer[index].set_type(type);
    time_t now = time(nullptr);
    inode_buffer[index].set_create(now);
    inode_buffer[index].set_access(now);
    inode_buffer[index].set_modify(now);
    int new_block = get_new_block();
    inode_buffer[index].add_block(new_block); // 给新的inode分配一个block
    return new_id;
}

int SmartBuffer::change_block() {
    SmartDiskDriver::save_block(&(block_buffer[block_head % BLOCK_NUM]));
    block_head++;
    block_tail++;
    return block_tail % BLOCK_NUM;
}

int SmartBuffer::change_inode() {
    SmartDiskDriver::save_inode(&(inode_buffer[inode_head % INODE_NUM]));
    inode_head++;
    inode_tail++;
    return inode_tail % INODE_NUM;
}

int SmartBuffer::allocate_block(int id) {
    int new_id = get_new_block();
    if (new_id > 1000) {
        return new_id;
    }
    int result = get_inode(id)->add_block(new_id);
    return result < 0 ? INODE_FULL : new_id;
}

void SmartBuffer::info() {
    cout << "Block Buffer: ";
    for (int i = 0; i < BLOCK_NUM; ++i) {
        if (block_buffer[i].get_id() > 0) {
            cout << block_buffer[i].get_id() << "; ";
        } else {
            cout << "empty; ";
        }
    }
    cout << endl;
    cout << "Inode Buffer: ";
    for (int i = 0; i < INODE_NUM; ++i) {
        if (inode_buffer[i].get_id() > 0) {
            cout << inode_buffer[i].get_id() << "; ";
        } else {
            cout << "empty; ";
        }
    }
    cout << endl;
}

int SmartBuffer::get_inode_free_block(int inode_id) {
    int *blocks = get_inode(inode_id)->get_blocks();
    for (int i = 0; i < Inode::get_limit(); ++i) {
        if (blocks[i] == 0) // 需要重新分配一个block
            break;
        if (get_block(blocks[i])->get_offset() < Block::get_block_size()) // 有一个block未被写满
            return blocks[i];
    }
    int new_block = Bitmap::get_free_block();
    if (new_block < 0) // 没有空闲的 block
        return NO_MORE_BLOCK;
    if (get_inode(inode_id)->add_block(new_block) < 0) //  block array 满了
        return INODE_FULL;
    else
        return new_block;
}

Error SmartBuffer::write_back() {
    Error error;
    auto *block = new Block;
    // 记录SuperBlock信息
    block->initial(1);
    block->set_data(SuperBlock::serialize());
    error = SmartDiskDriver::save_block(block);
    if (error.get_code() > 1000) {
        cout << "Error when saving block" << endl;
        delete block;
        return error;
    }

    // 记录Bitmap信息
    block->initial(2);
    block->set_data(Bitmap::serialize());
    error = SmartDiskDriver::save_block(block);
    if (error.get_code() > 1000) {
        cout << "Error when saving block" << endl;
        delete block;
        return error;
    }

    delete block;

    // 记录其他块信息
    for (int i = 0; i < BLOCK_NUM; ++i) {
        if (block_buffer[i].get_id() == 0) {
            break;
        }
        error = SmartDiskDriver::save_block(&(block_buffer[i]));
        if (error.get_code() > 1000) {
            cout << "Error when saving block" << endl;
            return error;
        }
    }
    for (int i = 0; i < INODE_NUM; ++i) {
        if (inode_buffer[i].get_id() == 0) {
            break;
        }
        error = SmartDiskDriver::save_inode(&(inode_buffer[i]));
        if (error.get_code() > 1000) {
            cout << "Error when saving inode" << endl;
            return error;
        }
    }
    return Error(SUCCESS);
}

Error SmartBuffer::reboot() {
    // 初始化
    Error error = initial(SmartDiskDriver::getSmartBlockNum(), SmartDiskDriver::getSmartBlockSize(),
                          SmartDiskDriver::getSmartInodeNum(), SmartDiskDriver::getSmartInodeSize());
    if (error.get_code() > 1000) { // 初始化时出现问题
        cout << "Error when rebooting buffer" << endl;
        return error;
    }

    auto *super_block = new Block();
    auto *bitmap = new Block();
    super_block->initial(0);
    bitmap->initial(0);
    SmartDiskDriver::load_block(super_block, 1);
    SmartDiskDriver::load_block(bitmap, 2);
    string temp = super_block->get_data();

    vector<string> temp_data = Tool::my_split(temp, '-');

    int free_inode_num = atoi(temp_data[4].c_str());

    // load super block
    SuperBlock::set_next(SmartDiskDriver::getSmartInodeNum() - free_inode_num + 1);

    // load bitmap
    temp = bitmap->get_data();
    Bitmap::set_metadata(SmartDiskDriver::getSmartBlockNum());
    Bitmap::set_usage(temp);

    delete super_block;
    delete bitmap;
    return Error(SUCCESS);
}
