//
// Created by A on 11/2/17.
//

#ifndef NEURALNETWORK_CONFIGFILEPARSER_H
#define NEURALNETWORK_CONFIGFILEPARSER_H

#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>

using namespace std;
// Reads in a configuration file for the neural network architecture
class ConfigFileParser {

private:
    string inputFile;

public:
    ConfigFileParser();
    ConfigFileParser(string newInputFile);
    void setInputFile(string newInputFile);
    vector<vector<string>> readFile();

};


#endif //NEURALNETWORK_CONFIGFILEPARSER_H
