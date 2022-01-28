//
// Created by Jiarui She on 2021/11/11.
//

#ifndef SMARTFILESYSTEM_TOOL_H
#define SMARTFILESYSTEM_TOOL_H


#include <string>
#include "vector"
#include <iostream>

using namespace std;

class Tool {
public:
    static int path_resolution(const string &, vector<string> &);

    static vector<string> my_split(const string &, char);
};


#endif //SMARTFILESYSTEM_TOOL_H
