#include "SmartFS.h"

using namespace std;

int main() {
    SmartFS fs = SmartFS();
    fs.run();

//    fs.reboot("../SmartDisk.txt", "../InodeTable/");
//    fs.info_block(3);
//    Error e = fs.open("/test.txt", O_RDONLY);
//    Error::handle(e);
//    e = fs.open("/cse/cse.txt", O_RDONLY);
//    Error::handle(e);

////    fs.info_buffer();
//    fs.create("/cse/");
//    fs.create("/cse/test.txt");
//    fs.create("/cse.txt");
//    Error e = fs.read(3, 0);
//    Error::handle(e);
//    fs.write(3, "This is Dir");
//    fs.info_file("/cse/");
//    fs.write(4, "This is a text file");
//    fs.info_file("/cse/text.txt");
//    fs.info_buffer();
////    fs.info_table();
//    fs.save();
//    SmartDiskDriver::save_metadata();

//    fs.reboot("../SmartDisk.txt", "../InodeTable/");
//    Error fd = fs.open("/cse/cse.txt", O_RDWR);
//    fs.read(fd.get_code(), 15);
//    fs.read(fd.get_code(), 3);
//    fs.info_buffer();
//    fs.info_table();
}
