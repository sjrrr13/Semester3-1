//
// Created by Jiarui She on 2021/11/11.
//

#include "Tool.h"

// 解析 path 的方法
int Tool::path_resolution(const string & path, vector<string>& path_array) {
    if (path.find('/', 0) != 0) // 如果不是以 root 开头的路径
        return -1;
    else {
        size_t index1 = 1;
        while(path.find('/', index1) != string::npos) {
            size_t index2 = path.find('/', index1);
            path_array.push_back(path.substr(index1, index2 - index1));
            index1 = index2 + 1;
        }
        path_array.push_back(path.substr(index1, path.length() - index1));
        return 1;
    }
}

vector<string> Tool::my_split(const string & data, char t) {
    vector<string> result;
    string temp;
    int i = 0;
    int len = (int)data.length();
    while (i < len) {
        if (data[i] != t)
            temp += data[i];
        else {
            result.push_back(temp);
            temp = "";
        }
        i++;
    }
    return result;
}