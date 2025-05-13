//
// Created by Federico on 01/12/2018.
//

#include "csvio.h"
#include <iostream>
#include <fstream>
#include <iomanip>
//#include <boost/algorithm/string.hpp>
#include <stdlib.h>
#include <string>
#include <sstream>

void split(std::vector<std::string>& vec, const std::string& line, const std::string& delimiter) {
    vec.clear(); // 清空结果容器
    size_t start = 0; // 当前子字符串的起始位置

    for (size_t i = 0; i < line.size(); ++i) {
        // 检查当前字符是否是分隔符
        if (delimiter.find(line[i]) != std::string::npos) {
            // 提取子字符串并添加到结果中
            vec.push_back(line.substr(start, i - start));
            start = i + 1; // 更新起始位置为下一个字符
        }
    }
    // 处理最后一个子字符串
    vec.push_back(line.substr(start));
}

void write2VecTo(std::string filename, std::string delimiter, std::vector<float>& vec){
    std::cout << "Saving to: " + filename << std::endl;

    std::ostringstream valStream;
    std::ofstream myfile(filename);
    std::cout << std::setprecision(7);
    myfile << std::setprecision(7);
    for(int i = 0; i < vec.size(); i+=3){
        valStream << std::fixed << vec[i] << delimiter << vec[i+1]<< delimiter << vec[i+2] << "\n";
    }
    myfile << valStream.str();
    myfile << std::endl;
    myfile.close();
}

void read2VecFrom(std::string filename, std::string delimiter, std::vector<float>& dest){
    std::cout << "Reading from: " + filename << std::endl;
    std::ifstream file(filename);

    std::string line = ",";
    // Iterate through each line and split the content using delimeter
    while (getline(file, line))
    {
        std::vector<std::string> vec;
        split(vec, line, (delimiter));
        std::vector<float> xy;
        for(auto i = vec.begin(); i != vec.end(); i++){
            float value = strtof(i->c_str(), NULL);
            if(value)
                dest.push_back( value );
        }
    }

    file.close();
    bool read = false;
    if(read) {
        for (int i = 0; i < dest.size(); i++) {
            std::cout << dest[i] << ", ";
            if ((i % 2))
                std::cout << std::endl;
        }
    }
}
