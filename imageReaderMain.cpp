/*********************************************************************
*Patrick's Solution to the Patwa
*
*
*Brute Force
*
*********************************************************************/
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <queue>
#include <fstream>
#include <cstddef>
#include "ImageReader.h"
using namespace std;

string getInput(string prompt){

  string response;

  cout << prompt << endl;

  cin >> response;

}

double getTrainingBit(){

    string trainingBitResponse = getInput("Are the files in this folder for positive training or negative training? Type 1 for positive and 0 for negative or 2 for test images");
    double returnBit = std::atof(trainingBitResponse.c_str());

    cout << "returnBit = " << returnBit << endl;

    return returnBit;
}

string getInputFolderPath(){

    string inputFilePath = getInput("What is the file path of the folder that should be converted?");

}

int main()
{

  ImageReader* kevin = new ImageReader();

  string inputFolder = getInputFolderPath();

  kevin -> setInputFolder(inputFolder);

  vector < vector < uchar > > imageData = kevin -> readFolder();

  string outFileName = getInput("What should be the name of the output file?");

  system(("touch " + outFileName).c_str());

  system(("rm " + outFileName).c_str());

  system(("touch " + outFileName).c_str());

  cout << "touched things" <<endl;

  ofstream outFile;

  outFile.open(outFileName.c_str());


    int trainingBit = getTrainingBit();

  int i=0;
  int j;
  vector < uchar > temp; 

  int iFinal=imageData.size();

  cout<< "iFinal = " << iFinal<<endl;

  int jFinal;
  
  while(i<iFinal){

    temp = imageData[i];

    jFinal=temp.size();

    cout<<jFinal<<endl;
    
    j=0;

    while(j<(jFinal-1)){

      outFile << (unsigned int) temp[j] << " ";

      cout<< (unsigned int) temp[j] << "";

      j++;
    }

    outFile<<(int) temp[jFinal-1] << " " << trainingBit <<endl;

    cout <<(int) temp[jFinal-1] <<endl;

    i++;
  }
    
  outFile.flush();

  outFile.close();

	return 0;
}