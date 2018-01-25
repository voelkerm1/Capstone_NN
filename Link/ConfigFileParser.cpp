//
// Created by andrew on 11/2/17.
//

#include "Include/ConfigFileParser.h"

using namespace std;

ConfigFileParser::ConfigFileParser() {
    inputFile = "";
}

ConfigFileParser::ConfigFileParser(string newInputFile) {
    inputFile = newInputFile;
}

void ConfigFileParser::setInputFile(string newInputFile) {
    inputFile = newInputFile;
}

vector<vector<string>> ConfigFileParser::readFile() {
    vector<vector<string>> layerVector; //Sets up file parse vector
    ifstream infile; //Sets up container for file
    infile.open(inputFile); //Opens Config File

    while (infile) {
        string s;
        if (!getline(infile, s)) break; //Breaks loop on empty file

        istringstream ss(s);
        vector<string> record; // Sets up container for lines of the file

        while (ss)
        {
            string s;
            if (!getline(ss, s, ' ')) break; //Grabs the lines of the file, broken up by spaces
            record.push_back(s); //Stores line
        }

        layerVector.push_back(record); //Stores broken lines (as strings) into the file parse vector
    }
    if (!infile.eof())
    {
        cerr << "No File Found\n"; //If no file is found to parse
    }
    return layerVector;

}